package openacc.codegen.llvmBackend;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Stack;

import openacc.codegen.llvmBackend.SrcQualifiedType.SrcTypeQualifier;
import cetus.hir.Case;
import cetus.hir.Declarator;
import cetus.hir.Default;
import cetus.hir.GotoStatement;
import cetus.hir.Label;
import cetus.hir.Procedure;
import cetus.hir.Statement;
import cetus.hir.SwitchStatement;
import cetus.hir.Traversable;

/**
 * The LLVM backend's class for checking constraints on scopes of goto,
 * switch, label, case, and default statements from the C source.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public class JumpScopeTable {
  /**
   * The {@link SrcSymbolTable} being maintained for the same C source.
   */
  private SrcSymbolTable srcSymbolTable;
  /**
   * The top of the stack of symbols that are currently in scope. For the sake
   * of efficiency, only symbols whose exact scopes need to be tracked for
   * some purpose in {@link JumpScopeTable} are recorded here. See
   * {@link #addSymbol}'s documentation and implementation for details.
   */
  private SymbolStackNode symbolStackTop = null;
  /**
   * A stack in which each entry represents a scope nested in the scopes
   * represented by the entries below on the stack, and the top entry
   * represents the current innermost scope. Each entry records
   * {@link #symbolStackTop} at the start of the entry's scope.
   */
  private Stack<ScopeStart> scopeStartStack = new Stack<>();
  /**
   * A list containing an entry for every goto statement encountered so far in
   * the current function. Each entry records {@link #symbolStackTop} at the
   * entry's goto statement.
   */
  private List<StatementScope> gotoScopes = new ArrayList<>();
  /**
   * A map containing an entry for every goto label statement encountered so
   * far in the current function. Each entry's key is the entry's goto label
   * statement. Each entry's value is the {@link #symbolStackTop} at the
   * entry's goto label statement.
   */
  private Map<Statement,SymbolStackNode> gotoLabelScopes
    = new IdentityHashMap<>();
  /**
   * A stack in which each entry represents a switch statement nested in the
   * switch statements represented by the entries below on the stack, and the
   * top entry represents the current innermost switch statement. Each stack
   * entry is a list containing an entry for every default statement or case
   * statement encountered so far that is a target for the stack entry's
   * switch statement. Each list entry records {@link #symbolStackTop} at the
   * list entry's default statement or case statement.
   */
  private Stack<List<StatementScope>> switchLabelScopesStack
    = new Stack<>();

  /**
   * Create a new jump scope table.
   * 
   * @param srcSymbolTable
   *          the {@link SrcSymbolTable} being maintained for the same C
   *          source
   */
  public JumpScopeTable(SrcSymbolTable srcSymbolTable) {
    this.srcSymbolTable = srcSymbolTable;
  }

  /**
   * Add a symbol to the current scope.
   * 
   * <p>
   * This method is intended to be called only from {@link SrcSymbolTable}.
   * Currently, only local symbols are added, so every entry here has an entry
   * retrievable via {@link SrcSymbolTable#getLocal} called on
   * {@link #srcSymbolTable}. Moreover, the implementation here discards some
   * local symbols as uninteresting for our purposes.
   * </p>
   *
   * @param declarator
   *          the {@link Declarator} node at the top of the symbol's
   *          declarator's tree. It must have been declared in the source
   *          after other symbols already passed to {@link #addSymbol} called
   *          on this.
   * @param valueAndType
   *          the symbol's value and type as stored in {@link #srcSymbolTable}
   * @param isLocalAlloc
   *          whether this symbol is being added for a declaration that
   *          allocates storage locally. For example, it must be false for a
   *          function type, an extern declaration, or a static local variable
   *          declaration (which allocates and initializes storage globally),
   *          but it must be true for a local non-extern non-static variable
   *          declaration.
   */
  public void addSymbol(Declarator declarator, ValueAndType valueAndType,
                        boolean isLocalAlloc)
  {
    assert(!isLocalAlloc
           || !valueAndType.getSrcType().iso(SrcFunctionType.class));
    // Skip some symbols we don't care about here.
    if (!isLocalAlloc
        || !valueAndType.getSrcType()
            .storageHasPointerToQualifier(SrcTypeQualifier.NVL,
                                          SrcTypeQualifier.NVL_WP))
      return;
    // A symbol might be added twice in a row: once before its initializer,
    // and once after. Skip it the second time to save space.
    if (symbolStackTop != null && symbolStackTop.declarator == declarator)
      return;
    symbolStackTop = new SymbolStackNode(declarator, isLocalAlloc,
                                         symbolStackTop);
  }

  /**
   * Record the start of a new scope.
   * 
   * @param scope
   *          the {@link Traversable} representing that scope
   */
  public void startScope(Traversable scope) {
    scopeStartStack.push(new ScopeStart(scope, symbolStackTop));
    if (scope instanceof SwitchStatement)
      switchLabelScopesStack.push(new ArrayList<StatementScope>());
  }

  /**
   * Record the end of a scope, and verify constraints on contained jumps.
   * 
   * @param scope
   *          the {@link Traversable} representing that scope. It must be the
   *          {@link Traversable} most recently passed to {@link #startScope}
   *          but not yet passed to {@link #endScope} (that is, scopes must be
   *          properly nested).
   */
  public void endScope(Traversable scope) {
    final ScopeStart scopeStart = scopeStartStack.pop();
    assert(scopeStart.scope == scope);
    symbolStackTop = scopeStart.symbolStackTopAtStart;
    if (scope instanceof SwitchStatement) {
      assert(!switchLabelScopesStack.isEmpty());
      checkJumpScopes(new StatementScope((SwitchStatement)scope,
                                         scopeStart.symbolStackTopAtStart),
                      switchLabelScopesStack.pop());
    }
    else if (scope instanceof Procedure) {
      for (StatementScope gotoScope : gotoScopes) {
        final Statement label = ((GotoStatement)gotoScope.statement)
                                .getTarget();
        final List<StatementScope> labelScopeList = new ArrayList<>();
        labelScopeList.add(
          new StatementScope(label, gotoLabelScopes.get(label)));
        checkJumpScopes(gotoScope, labelScopeList);
      }
      gotoScopes.clear();
      gotoLabelScopes.clear();
    }
  }

  /**
   * Set the scope of a goto statement. Must be called at the moment when
   * other methods in this class have recorded exactly the set of symbols that
   * are in scope at that statement in the source.
   * 
   * @param node
   *          the node representing the goto statement
   */
  public void setGotoScope(GotoStatement node) {
    gotoScopes.add(new StatementScope(node, symbolStackTop));
  }

  /**
   * Set the scope of a goto label statement. Must be called at the moment
   * when other methods in this class have recorded exactly the set of symbols
   * that are in scope at that statement in the source.
   * 
   * @param node
   *          the node representing the goto label statement
   */
  public void setGotoLabelScope(Label node) {
    gotoLabelScopes.put(node, symbolStackTop);
  }

  /**
   * Set the scope of a case or default statement. Must be called at the
   * moment when other methods in this class have recorded exactly the set of
   * symbols that are in scope at that statement in the source.
   * 
   * @param node
   *          the node representing the case or default statement
   */
  public void setSwitchLabelScope(Statement node) {
    assert(!switchLabelScopesStack.isEmpty());
    switchLabelScopesStack.peek().add(
      new StatementScope(node, symbolStackTop));
  }

  private void checkJumpScopes(StatementScope jumpScope,
                               List<StatementScope> labelScopeList)
  {
    for (SymbolStackNode node = jumpScope.symbolStackTop;
         node != null; node = node.predecessor)
      node.mark = true;
    for (StatementScope labelScope : labelScopeList) {
      for (SymbolStackNode node = labelScope.symbolStackTop;
           node != null; node = node.predecessor)
      {
        final ValueAndType valueAndType = srcSymbolTable
                                          .getLocal(node.declarator);
        // Skip symbols that do not store pointers to NVM.
        if (!node.isLocalAlloc
            || !valueAndType.getSrcType()
                .storageHasPointerToQualifier(SrcTypeQualifier.NVL,
                                              SrcTypeQualifier.NVL_WP))
          continue;
        // We have a local pointer to NVM in the scope of the jump target. If
        // it's not also in the scope of the jump statement, then the jump
        // bypasses its initialization.
        if (!node.mark) {
          final String jumpKind;
          if (jumpScope.statement instanceof GotoStatement)
            jumpKind = "goto";
          else {
            assert(jumpScope.statement instanceof SwitchStatement);
            if (labelScope.statement instanceof Case)
              jumpKind = "switch case";
            else {
              assert(labelScope.statement instanceof Default);
              jumpKind = "switch default";
            }
          }
          throw new SrcRuntimeException(
            jumpKind+" bypasses initialization of pointer to NVM: "
            +node.declarator.getID().getName());
        }
        break;
      }
    }
    for (SymbolStackNode node = jumpScope.symbolStackTop;
         node != null; node = node.predecessor)
      node.mark = false;
  }

  private static final class SymbolStackNode {
    private final Declarator declarator;
    private final boolean isLocalAlloc;
    private final SymbolStackNode predecessor;
    private boolean mark;
    public SymbolStackNode(Declarator declarator, boolean isLocalAlloc,
                           SymbolStackNode predecessor)
    {
      this.declarator = declarator;
      this.isLocalAlloc = isLocalAlloc;
      this.predecessor = predecessor;
      this.mark = false;
    }
  }
  private static final class ScopeStart {
    private final Traversable scope;
    private final SymbolStackNode symbolStackTopAtStart;
    public ScopeStart(Traversable scope,
                      SymbolStackNode symbolStackTopAtStart)
    {
      this.scope = scope;
      this.symbolStackTopAtStart = symbolStackTopAtStart;
    }
  }
  private static final class StatementScope {
    private final Statement statement;
    private final SymbolStackNode symbolStackTop;
    public StatementScope(Statement statement, SymbolStackNode symbolStackTop)
    {
      this.statement= statement;
      this.symbolStackTop = symbolStackTop;
    }
  }
}
