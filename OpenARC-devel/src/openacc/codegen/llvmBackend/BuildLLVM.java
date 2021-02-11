package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcPrimitiveFloatingType.SrcDoubleType;
import static openacc.codegen.llvmBackend.SrcPrimitiveFloatingType.SrcFloatType;
import static openacc.codegen.llvmBackend.SrcPrimitiveFloatingType.SrcLongDoubleType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SRC_CHAR_CONST_TYPE;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SRC_ENUM_CONST_TYPE;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SRC_SIZE_T_TYPE;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SRC_WCHAR_TYPE;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcBoolType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcCharType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcIntType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcLongLongType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcLongType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedIntType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedLongLongType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedLongType;
import static openacc.codegen.llvmBackend.SrcPrimitiveNonNumericType.SrcVoidType;
import static org.jllvm.bindings.LLVMAttribute.*;
import static org.jllvm.bindings.LLVMLinkage.*;
import static org.jllvm.bindings.LLVMTypeKind.*;

import java.io.File;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.IdentityHashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Stack;
import java.util.Vector;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import openacc.codegen.BuildLLVMDelegate;
import openacc.codegen.llvmBackend.SrcQualifiedType.SrcTypeQualifier;
import openacc.codegen.llvmBackend.SrcSymbolTable.InlineDefState;
import openacc.codegen.llvmBackend.SrcType.AndXorOr;
import openacc.codegen.llvmBackend.ValueAndType.AssignKind;

import org.jllvm.LLVMAndInstruction;
import org.jllvm.LLVMArgument;
import org.jllvm.LLVMBasicBlock;
import org.jllvm.LLVMBitCast;
import org.jllvm.LLVMBranchInstruction;
import org.jllvm.LLVMCallInstruction;
import org.jllvm.LLVMConstant;
import org.jllvm.LLVMConstantArray;
import org.jllvm.LLVMConstantExpression;
import org.jllvm.LLVMConstantInlineASM;
import org.jllvm.LLVMConstantInteger;
import org.jllvm.LLVMConstantReal;
import org.jllvm.LLVMContext;
import org.jllvm.LLVMExecutionEngine;
import org.jllvm.LLVMExtendCast;
import org.jllvm.LLVMExtendCast.ExtendType;
import org.jllvm.LLVMFunction;
import org.jllvm.LLVMFunctionType;
import org.jllvm.LLVMGetElementPointerInstruction;
import org.jllvm.LLVMGlobalVariable;
import org.jllvm.LLVMIdentifiedStructType;
import org.jllvm.LLVMInstruction;
import org.jllvm.LLVMInstructionBuilder;
import org.jllvm.LLVMIntegerType;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMOrInstruction;
import org.jllvm.LLVMPhiNode;
import org.jllvm.LLVMPointerType;
import org.jllvm.LLVMReturnInstruction;
import org.jllvm.LLVMSelectInstruction;
import org.jllvm.LLVMStackAllocation;
import org.jllvm.LLVMStoreInstruction;
import org.jllvm.LLVMSwitchInstruction;
import org.jllvm.LLVMTargetData;
import org.jllvm.LLVMType;
import org.jllvm.LLVMUser;
import org.jllvm.LLVMValue;
import org.jllvm.bindings.LLVMLinkage;
import org.jllvm.bindings.LLVMVerifierFailureAction;

import cetus.hir.*;

/**
 * Builds LLVM IR from a program.
 * 
 * <p>
 * This class is compiled if and only if OpenARC is built with LLVM support
 * enabled.  However, {@link BuildLLVMDelegate} is always compiled, so use it
 * instead.
 * </p>
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class BuildLLVM extends BuildLLVMDelegate {
  /**
   * The single {@link LLVMContext} used by this {@link BuildLLVM} instance.
   * The global LLVM context is never used.
   * 
   * <p>
   * Only one thread should access any given LLVM context at a time, and LLVM
   * contexts are not intended to interact with one another. Thus, multiple
   * LLVM contexts are useful to permit different threads of a program to
   * separately run LLVM. See {@link LLVMContext}'s header comments for more
   * details.
   * </p>
   * 
   * <p>
   * For several reasons, we choose to use a different LLVM context per
   * {@link BuildLLVM} object instead of, for example, using the global LLVM
   * context for all {@link BuildLLVM} objects. First, it is convenient to
   * have a {@link #finalize} method that automatically frees LLVM memory
   * associated with a {@link BuildLLVM} object when the {@link BuildLLVM}
   * object is no longer used. However, the JVM calls {@code finalize} methods
   * in separate threads, so, if there were multiple {@link BuildLLVM} objects
   * associated with the same context, such a {@code finalize} method could
   * produce concurrent accesses to that context. Moreover, because each
   * {@link BuildLLVM} object has its own context, cleaning up LLVM memory for
   * a {@link BuildLLVM} object is very simple: {@link #finalize} just tells
   * LLVM to dispose of the entire context, which in turn disposes of all
   * associated modules, values, types, etc. (see discussion at
   * {@link #getLLVMModules}). The cleanup provided by our {@link #finalize}
   * method is helpful in the JUnit test suite, for example, so that we can
   * automatically free LLVM memory in between test cases.
   * </p>
   * 
   * <p>
   * Another reason we do not use the global context instead is so that, in
   * the JUnit test suite, the {@link LLVMIdentifiedStructType}s from one test
   * case are not seen in another. Thus, we avoid false positives when an
   * expected struct type was actually constructed by a different test case,
   * and we avoid false negatives when LLVM assigns a struct type a different
   * tag because another test case already constructed a struct type with the
   * original tag. Moreover, when renaming struct types, LLVM uses a
   * per-context counter rather than a per-struct-tag counter, and JUnit test
   * case execution order is by default unpredictable, so separate contexts
   * for separate test cases enable us to predict renames within a single test
   * case.
   * </p>
   * 
   * <p>
   * If using separate LLVM contexts proves undesirable, perhaps because we
   * decide we want to be able to compare aspects of the LLVM IR from several
   * {@link BuildLLVM} objects, then the initialization of
   * {@link #llvmContext} can be changed to the global context, or the caller
   * could provide a context, which it could also provide to other
   * {@link BuildLLVM} objects. For the former case, the test suite will
   * likely fail as described above, so selecting the global context should
   * probably be made optional. For the latter case, {@link BuildLLVM}'s
   * finalize method must be altered to skip disposal of the context when the
   * context is provided by the caller.
   * </p>
   * 
   * <p>
   * Generally in LLVM, in what context a derived type ends up depends on its
   * element types. For example, a function type's context is implied by its
   * return type's context. Generally in LLVM, an {@link LLVMFunction} has the
   * context of its {@link LLVMModule}.
   * </p>
   */
  private final LLVMContext llvmContext = new LLVMContext();
  private LLVMModule[] llvmModules = null;
  private String[] llvmModuleIdentifiers = null;
  private final String llvmTargetTriple;
  private final LLVMTargetData llvmTargetData;
  private final boolean traceASTTraversal;
  private final boolean dumpLLVMModules;
  private final boolean verifyModule;
  private final boolean debugTypeChecksums;
  private final boolean reportLine;
  private final boolean warningsAsErrors;
  private final boolean enableFaultInjection;

  /**
   * Should be called only by {@link BuildLLVMDelegate#make}. It is called via
   * Java reflection, so Eclipse will not find the call.
   */
  public BuildLLVM(String llvmTargetTriple, String llvmTargetDataLayout,
                   Program program, boolean debugOutput, boolean reportLine,
                   boolean warningsAsErrors, boolean enableFaultInjection)
  {
    super(program);
    this.llvmTargetTriple = llvmTargetTriple;
    this.llvmTargetData = new LLVMTargetData(llvmTargetDataLayout);
    this.traceASTTraversal = debugOutput;
    this.dumpLLVMModules = debugOutput;
    this.verifyModule = debugOutput;
    this.debugTypeChecksums = debugOutput;
    this.reportLine = reportLine;
    this.warningsAsErrors = warningsAsErrors;
    this.enableFaultInjection = enableFaultInjection;
  }

  /**
   * Get the {@link LLVMModule}s constructed by this pass.
   * 
   * <p>
   * WARNING: The {@link LLVMModule}s returned here will become invalid after
   * {@link #finalize} is called on this, so maintain a reference to this
   * until you're done with the {@link LLVMModule}s. Also, be sure to dispose
   * of any associated {@link LLVMExecutionEngine}s before {@link #finalize}
   * is called on this.
   * </p>
   * 
   * @return the {@link LLVMModule}s constructed by this pass, or null if the
   *         pass has not yet been run
   */
  public LLVMModule[] getLLVMModules() {
    return llvmModules;
  }

  /**
   * Get the identifiers for the {@link LLVMModule}s constructed by this pass.
   * 
   * @return the identifiers for the {@link LLVMModule}s constructed by this
   *         pass, or null if the pass has not yet been run
   */
  public String[] getLLVMModuleIdentifiers() {
    return llvmModuleIdentifiers;
  }

  @Override
  public void printLLVM(String outdir) {
    assert(llvmModules != null);
    final File dir = new File(outdir);
    if (!dir.exists() && !dir.mkdir()) {
      System.err.println("cetus: could not create LLVM output directory: "
                         + outdir);
      Tools.exit(1);
    }
    for (int i = 0; i < llvmModules.length; ++i) {
      final String inFile = new File(llvmModuleIdentifiers[i]).getName();
      final int dot = inFile.lastIndexOf('.');
      if (dot == -1)
        throw new SrcRuntimeException(
          "trying to print LLVM module to \".bc\" file but current name"
          +" contains no \".\": "+inFile);
      final String outFile = outdir+"/"+inFile.substring(0, dot)+".bc";
      if (!llvmModules[i].writeBitcodeToFile(outFile)) {
        System.err.println("cetus: could not write LLVM output file: "
                           + outFile);
        Tools.exit(1);
      }
    }
  }

  @Override
  public void dumpLLVM() {
    for (LLVMModule llvmModule : llvmModules)
      llvmModule.dump();
  }

  @Override
  public void start() {
    new Visitor(traceASTTraversal ? "" : null).visitTree(program);
  }

  protected static void warnOrError(boolean warn, boolean warningsAsErrors,
                                    String msg)
  {
    if (!warn)
      throw new SrcRuntimeException(msg);
    warn(warningsAsErrors, msg);
  }

  protected static void warn(boolean warningsAsErrors, String msg) {
    if (warningsAsErrors)
      throw new SrcRuntimeException("warning treated as error: "
                                    + msg);
    else
      PrintTools.println("\n[WARNING in BuildLLVM] " + msg, 0);
  }

  /**
   * For a pre-order traversal of a {@link Procedure},
   * {@link VariableDeclaration}, {@link SizeofExpression}'s type operand,
   * {@link OffsetofExpression}'s type operand, {@link Typecast}'s type
   * operand, {@link VaArgExpression}'s type operand, or type operand for
   * any NVL-C built-in function call node, this contains attributes of the
   * declaration that must be seen by the descendant {@link Declarator}s.
   */
  protected static class DeclarationAttributes {
    public DeclarationAttributes(Traversable parent, boolean hasExtern,
                                 boolean hasStatic, boolean hasInline,
                                 boolean hasThread, boolean isTypedef,
                                 boolean isProcedureDefinition,
                                 Declarator topDeclarator, SrcType srcType)
    {
      this.isGlobal = parent instanceof TranslationUnit;
      this.getNameAndType = parent instanceof ProcedureDeclarator // param
                            || parent instanceof NestedDeclarator // param
                            || (parent instanceof DeclarationStatement
                                && parent.getParent()
                                     instanceof ClassDeclaration) // member
                            || parent instanceof SizeofExpression
                            || parent instanceof OffsetofExpression
                            || parent instanceof Typecast
                            || parent instanceof VaArgExpression
                            || parent instanceof NVLGetRootExpression
                            || parent instanceof NVLAllocNVExpression;
      this.hasExtern = hasExtern;
      this.hasStatic = hasStatic;
      this.hasInline = hasInline;
      this.hasThread = hasThread;
      this.isTypedef = isTypedef;
      this.isProcedureDefinition = isProcedureDefinition;
      this.topDeclarator = topDeclarator;
      this.srcType = srcType;
    }
    /** True iff the declaration is at file scope. */
    public final boolean isGlobal;
    /**
     * True iff the declaration is for a procedure parameter, a member of a
     * struct or union, a sizeof's type operand, an offsetof's type operand,
     * an explicit cast's type operand, a va_arg's type operand, or a type
     * operand of any NVL-C built-in function.
     */
    public final boolean getNameAndType;
    /** True iff extern was specified at the declaration level. */
    public final boolean hasExtern;
    /** True iff static was specified at the declaration level. */
    public final boolean hasStatic;
    /** True iff inline was specified at the declaration level. */
    public final boolean hasInline;
    /** True iff __thread was specified at the declaration level. */
    public final boolean hasThread;
    /** True iff the declaration is a typedef. */
    public final boolean isTypedef;
    /**
     * True iff the declaration is a {@link #Procedure} and thus has a
     * procedure definition.
     */
    public final boolean isProcedureDefinition;
    /**
     * The {@link Declarator} node at the top of a declarator's tree, set here
     * before the traversal of each declarator's tree is started. Any
     * initializer is attached to this node. Moreover, it is the node that
     * Cetus uses as the {@link Symbol} for the declarator in symbol tables,
     * and so it is also the node that {@link BuildLLVM} uses in its symbol
     * tables. ({@link NestedDeclarator} and {@link VariableDeclarator} have
     * equals and hashCode methods so that you can supposedly look up symbols
     * by any {@link Declarator} in a declarator's tree. That is not true for
     * a {@link ProcedureDeclarator}, and it doesn't help with finding the
     * {@link Initializer} node. For consistency, we just assume you need the
     * topmost {@link Declarator} in all cases.)
     */
    public Declarator topDeclarator;
    /**
     * The type, set at the declaration level before the traversal of each
     * declarator is started, and modified as each declarator's tree is
     * descended.
     */
    public SrcType srcType;
  }

  /**
   * Data used to tell the {@link Initializer} visitor what the LLVM IR it
   * generates must initialize. See {@link Visitor#initDestinationStack}.
   */
  protected static class InitDestination {
    /** Specify that constant must be generated. */
    public InitDestination(SrcType srcType) {
      this.srcType = srcType;
      this.stackAllocName = null;
      this.stackAddr = null;
    }
    /** Specify a stack allocation to generate. */
    public InitDestination(SrcType srcType, String stackAllocName) {
      this.srcType = srcType;
      this.stackAllocName = stackAllocName;
      this.stackAddr = null;
    }
    /** Specify an existing address to store to. */
    public InitDestination(SrcType srcType, LLVMValue stackAddr)
    {
      this.srcType = srcType;
      this.stackAllocName = null;
      this.stackAddr = stackAddr;
    }
    /** Specify an lvalue to store to. */
    public InitDestination(ValueAndType valueAndType) {
      srcType = valueAndType.getSrcType();
      stackAllocName = null;
      stackAddr = valueAndType.getLLVMValue();
      assert(valueAndType.isLvalue());
    }

    /** Fill in size for an array type with unspecified size. */
    public void updateArrayType(long numElements) {
      final SrcArrayType srcArrayType = srcType.toIso(SrcArrayType.class);
      assert(srcArrayType != null);
      assert(!srcArrayType.numElementsIsSpecified());
      srcType = SrcArrayType.get(srcArrayType.getElementType(), numElements);
    }

    /**
     * Generate specified stack allocation, update with the resulting lvalue,
     * and return it.
     */
    public ValueAndType generateStackAlloc(
      LLVMContext llvmContext, LLVMInstructionBuilder allocaBuilder)
    {
      assert(stackAllocName != null);
      stackAddr = new LLVMStackAllocation(allocaBuilder, stackAllocName,
                                          srcType.getLLVMType(llvmContext),
                                          null);
      stackAllocName = null;
      return new ValueAndType(stackAddr, srcType, true);
    }

    /**
     * The declared type the visitor must use for the initializer. If it's an
     * array with an unspecified number of elements, then its number of
     * elements must be taken from the initializer.
     */
    public SrcType srcType;
    /**
     * If non-null, then {@link #stackAddr} must be null, and using
     * {@link Visitor#getLLVMAllocaBuilder} the visitor must generate an
     * {@link LLVMStackAllocation} with this name. That
     * {@link LLVMStackAllocation} should then be used in place of
     * {@link #stackAddr}, and the resulting lvalue must be pushed to
     * {@link Visitor#postOrderValuesAndTypes}.
     */
    public String stackAllocName;
    /**
     * If non-null, then {@link #stackAllocName} must be null, and at the
     * current location in the current LLVM function, the visitor must
     * generate {@link LLVMStoreInstruction}s to store the initializer to this
     * address. If both {@link #stackAllocName} and {@link #stackAddr} are
     * null, then the visitor must push an {@link LLVMConstant} to
     * {@link Visitor#postOrderValuesAndTypes} instead.
     */
    public LLVMValue stackAddr;
  }

  protected class Visitor implements TraversableVisitor {
    final LLVMContext llvmContext;
    final LLVMTargetData llvmTargetData;
    final boolean warningsAsErrors;

    /**
     * Construct a visitor.
     * 
     * @param tracePrefix the text to print to stderr before each object in a
     *                    trace of the traversal, or null if tracing should be
     *                    disabled
     */
    public Visitor(String tracePrefix) {
      this.tracePrefix = tracePrefix;
      this.llvmContext = BuildLLVM.this.llvmContext;
      this.llvmTargetData = BuildLLVM.this.llvmTargetData;
      this.warningsAsErrors = BuildLLVM.this.warningsAsErrors;
      if (enableFaultInjection)
        pragmaTranslators.add(new FITL(this));
      pragmaTranslators.add(new NVL(this));
    }

    /**
     * Visit all nodes in a tree.
     * 
     * @param root
     *          the root of the tree to traverse
     */
    public void visitTree(Traversable root) {
      final String tracePrefixOld = tracePrefix;
      DepthFirstIterator<Traversable> itr
        = new DepthFirstIterator<>(root, true, tracePrefixOld);
      itr.setDefaultOrderToPost();
      itr.pruneOn(Program.class);
      itr.pruneOn(TranslationUnit.class);
      itr.pruneOn(Procedure.class);
      itr.pruneOn(VariableDeclaration.class);
      itr.pruneOn(VariableDeclarator.class);
      itr.pruneOn(NestedDeclarator.class);
      itr.pruneOn(ClassDeclaration.class);
      itr.reverseOrderFor(Enumeration.class);
      itr.pruneOn(SizeofExpression.class);
      itr.pruneOn(Initializer.class);
      itr.pruneOn(BinaryExpression.class);
      itr.pruneOn(ConditionalExpression.class);
      itr.pruneOn(FunctionCall.class);
      itr.pruneOn(WhileLoop.class);
      itr.pruneOn(DoLoop.class);
      itr.pruneOn(ForLoop.class);
      itr.pruneOn(IfStatement.class);
      itr.pruneOn(SwitchStatement.class);
      itr.pruneOn(Case.class);
      itr.reverseOrderFor(StatementExpression.class);
      itr.pruneOn(CompoundStatement.class);
      itr.pruneOn(DeclarationStatement.class);
      itr.pruneOn(ExpressionStatement.class);
      itr.pruneOn(ReturnStatement.class);
      while (itr.hasNext()) {
        Traversable node = itr.next();
        tracePrefix = itr.getChildTracePrefix();
        if (reportLine && node instanceof Statement) {
          final Statement stat = (Statement)node;
          try {
            node.accept(this);
          }
          catch (SrcRuntimeException e) {
            throw new SrcRuntimeException(e, stat.where());
          }
        }
        else
          node.accept(this);
      }
      tracePrefix = tracePrefixOld;
    }

    /**
     * When tracing is enabled (it's not null), this is used by
     * {@link #visitTree} to track the trace prefix for
     * {@link DepthFirstIterator}s.
     */
    private String tracePrefix;

    /** The translation unit currently being translated to LLVM IR. */
    protected TranslationUnit tu;
    /** The module currently being constructed (for {@link #tu}). */
    protected LLVMModule llvmModule;
    /**
     * The index of {@link #llvmModule} within the array of modules being
     * constructed. This index is also the index of {@link #tu} in the array
     * of translation units being translated.
     */
    protected int llvmModuleIndex = -1;
    /**
     * The procedure currently being translated to LLVM IR, or null if none.
     */
    protected Procedure procedure = null;
    /**
     * Either (1) the {@link SrcFunctionType} for {@link #procedure}, (2) the
     * {@link SrcFunctionType} for a temporary function created for examining
     * an expression (see {@link FakeFunctionForEval}), or (3) null if
     * neither of those exist.
     */
    protected SrcFunctionType srcFunctionType = null;
    /**
     * Either (1) the {@link LLVMFunction} currently being constructed for
     * {@link #procedure}, (2) the temporary {@link LLVMFunction} created for
     * examining an expression (see {@link FakeFunctionForEval}), or (3) null
     * if neither of those exist.
     */
    protected LLVMFunction llvmFunction = null;
    /**
     * The IR builder currently in use for all but alloca instructions
     * ({@link LLVMStackAllocation}). Use {@link #getLLVMAllocaBuilder} instead
     * for alloca instructions.
     */
    protected LLVMInstructionBuilder llvmBuilder
      = new LLVMInstructionBuilder(BuildLLVM.this.llvmContext);

    /**
     * Get an IR builder for generating alloca instructions
     * ({@link LLVMStackAllocation}) at the beginning of the current
     * function's ({@link #llvmFunction}) entry block. The caller must ensure
     * that there is a current function (we're not at file scope).
     * 
     * <p>
     * To insert other kinds of instructions, use {@link #llvmBuilder}
     * instead. Afterwards, if more allocas need to be inserted, be careful to
     * call this method again rather than using the builder it previously
     * returned. Otherwise, the new allocas might not be inserted at the
     * correct position. The reverse problem does not seem to happen. (If both
     * builders are pointing at the end of the entry block, it's fine for
     * {@link #llvmBuilder} to remain at the end while an alloca builder
     * inserts before it, but the alloca builder should remain at the end of
     * only the leading alloca instructions, so {@link #llvmBuilder} should
     * not be allowed to insert before it.)
     * </p>
     */
    protected LLVMInstructionBuilder getLLVMAllocaBuilder() {
      assert(llvmFunction != null);
      // http://llvm.org/docs/tutorial/LangImpl7.html#memory-in-llvm gives
      // requirements for mem2reg to operate on a variable. One condition is
      // that its alloca must be in the entry block.
      final LLVMInstructionBuilder res
        = new LLVMInstructionBuilder(llvmContext);
      // We could just insert before the first instruction, but that would
      // insert allocas in the reverse of declaration order, which would be
      // less readable.
      LLVMInstruction insn;
      for (insn = llvmFunction.getEntryBasicBlock().getFirstInstruction();
           insn.getInstance() != null && insn instanceof LLVMStackAllocation;
           insn = insn.getNextInstruction())
        ;
      res.positionBuilder(llvmFunction.getEntryBasicBlock(), insn);
      return res;
    }

    /**
     * The top of the stack stores the target for any continue statement
     * encountered.
     */
    protected Stack<LLVMBasicBlock> continueTargetStack = new Stack<>();
    /**
     * The top of the stack stores the target for any break statement
     * encountered.
     */
    protected Stack<LLVMBasicBlock> breakTargetStack = new Stack<>();

    /** Index of symbols used in the C source. */
    protected SrcSymbolTable srcSymbolTable = null;
    /**
     * Table for checking constraints on the scopes of goto, switch, label,
     * case, and default statements that are currently in scope.
     */
    protected JumpScopeTable jumpScopeTable = null;

    /**
     * Stack of basic block sets, in each of which we attach metadata to basic
     * blocks (at their terminator instructions) to identify those sets.
     */
    protected BasicBlockSetStack basicBlockSetStack = null;

    /** Translators for pragmas. */
    protected List<PragmaTranslator> pragmaTranslators
      = new ArrayList<PragmaTranslator>();

    /**
     * Within a pre-order traversal of a {@link Procedure},
     * {@link VariableDeclaration}, {@link SizeofExpression}'s type operand,
     * {@link OffsetofExpression}'s type operand, {@link Typecast}'s type
     * operand, {@link VaArgExpression}'s type operand, or type operand of
     * any NVL-C built-in function call node, there can be other
     * {@link VariableDeclaration}s (for example, parameter declarations, or
     * struct members). Thus, we need this stack of declaration attribute
     * sets, which the visitor for each of the above node types pushes
     * before starting a traversal of its descendants and pops afterward.
     */
    protected Stack<DeclarationAttributes> declarationAttributesStack
      = new Stack<>();
    /**
     * The initializer for the next enumerator declaration. Set to zero at
     * each {@link Enumeration}, and then overwritten or incremented at each
     * {@link VariableDeclarator} within it.
     * 
     * <p>
     * Within an {@link Enumeration}, there can be other {@link Enumeration}s
     * (for example, in a sizeof) within an enumerator's initializer.
     * Nevertheless, we don't need a stack here because, so far, Cetus always
     * moves the nested {@link Enumeration} before the enclosing
     * {@link Enumeration}. Even if it didn't, the visitor always writes the
     * enumerator's initializer value here after the initializer and thus
     * after any nested {@link Enumeration}s have been fully traversed.
     * </p>
     */
    protected LLVMConstant enumeratorLLVMInit;
    /**
     * Stack of initializer destinations.
     * 
     * <p>
     * Before an {@link Initializer} node is visited, its parent pushes the
     * initializer destination for it. Thus, the remaining elements on the
     * stack are the initializer destinations within which that initializer
     * destination is nested (for example, a member within a struct).
     * </p>
     */
    protected Stack<InitDestination> initDestinationStack = new Stack<>();
    /**
     * Values (and their types) that have been constructed but not yet picked
     * up by their parents in a post-order traversal.
     */
    protected Vector<ValueAndType> postOrderValuesAndTypes = new Vector<>();
    /**
     * Types that have been constructed but not yet picked up by their
     * parents in a post-order traversal.
     */
    protected Vector<SrcType> postOrderSrcTypes = new Vector<>();
    /**
     * Names that have been stored but not yet picked up by their parents in a
     * post-order traversal.
     */
    protected Vector<String> postOrderNames = new Vector<>();

    /**
     * Construct and push an element to {@link #declarationAttributesStack}.
     * This is called at a {@link Procedure}, {@link VariableDeclaration},
     * {@link SizeofExpression}'s type operand, {@link OffsetofExpression}'s
     * type operand, {@link Typecast}'s type operand,
     * {@link VaArgExpression}'s type operand, or type operand of any NVL-C
     * built-in function call node.
     * 
     * @param parent
     *          the parent node of the declaration
     * @param isProcedureDefinition
     *          whether the declaration is a {@link #Procedure} and thus has a
     *          procedure definition
     * @param specifiers
     *          the specifiers from the declaration
     * @param specifiersAreVarArg
     *          whether the specifiers are associated with an ellipsis at the
     *          end of a parameter list
     * @param topDeclarator
     *          the topmost {@link Declarator} node of the declarator tree
     *          that is about to be traversed
     */
    protected void pushDeclarationAttributes(Traversable parent,
                                             boolean isProcedureDefinition,
                                             List<Specifier> specifiers,
                                             boolean specifiersAreVarArg,
                                             Declarator topDeclarator)
    {
      // Gather specifiers to the left of the declarator list.
      boolean hasStatic = false;
      boolean hasExtern = false;
      boolean isTypedef = false;
      boolean hasInline = false;
      boolean hasThread = false;
      final List<Specifier> typeSpecifiersAndQualifiers
        = new ArrayList<Specifier>();
      for (Specifier s : specifiers) {
        // Storage class specifiers.
        if (s == Specifier.AUTO)          /*redundant*/;
        else if (s == Specifier.REGISTER) /*TODO: useful?*/;
        else if (s == Specifier.STATIC)   hasStatic = true;
        else if (s == Specifier.EXTERN)   hasExtern = true;
        else if (s == Specifier.THREAD)   hasThread = true;
        else if (s == Specifier.TYPEDEF)  isTypedef = true;
        // Function specifier.
        else if (s == Specifier.INLINE)   hasInline = true;
        // Type specifiers and qualifiers.
        else typeSpecifiersAndQualifiers.add(s);
      }
      final SrcType srcType;
      if (specifiersAreVarArg) {
        assert(typeSpecifiersAndQualifiers.isEmpty());
        srcType = null;
      }
      else
        srcType = srcSymbolTable.typeSpecifiersAndQualifiersToSrcType(
          typeSpecifiersAndQualifiers, parent, llvmModule, warningsAsErrors);
      declarationAttributesStack.push(
        new DeclarationAttributes(parent, hasExtern, hasStatic, hasInline,
                                  hasThread, isTypedef,
                                  isProcedureDefinition, topDeclarator,
                                  srcType));
    }

    /**
     * Set the type at the top of {@link #declarationAttributesStack} using
     * the type already there as the target type of the type specified in a
     * list of specifiers from a {@link Declarator}.
     * 
     * @param specifiers
     *          the list of specifiers
     */
    protected void mergeSpecifiersToSrcType(List<Specifier> specifiers) {
      SrcType type = declarationAttributesStack.peek().srcType;
      for (Specifier specifier : specifiers) {
        // TODO: Handle other kinds of specifiers. So far, we've not found a
        // case where there are any others from C source.
        if (specifier instanceof PointerSpecifier) {
          type = SrcPointerType.get(type);
          for (Specifier specifierOnPtr
               : ((PointerSpecifier)specifier).getQualifiers())
            type = SrcQualifiedType.get(type, specifierOnPtr);
        }
        else
          throw new IllegalStateException("unexpected specifier: "+specifier);
      }
      declarationAttributesStack.peek().srcType = type;
    }
    /**
     * Wrapper for {@link mergeSpecifiersToSrcType} but without warnings when
     * passing raw type {@link List}.
     */
    @SuppressWarnings("unchecked")
    protected void mergeSpecifiersToSrcTypeUnchecked(@SuppressWarnings("rawtypes") List specifiers) {
      mergeSpecifiersToSrcType(specifiers);
    }
    /**
     * Set the type at the top of {@link #declarationAttributesStack} using
     * the type already there as the element type of the type specified in a
     * list of array specifiers from a {@link Declarator}. Also, call
     * {@link SrcSymbolTable#addParentFixup} for any dimension expressions.
     * 
     * @param arraySpecifiers
     *          the list of array specifiers
     * @param declarator
     *          the declarator
     */
    protected void mergeArraySpecifiersToSrcType(
      @SuppressWarnings("rawtypes") List arraySpecifiers,
      Declarator declarator)
    {
      if (arraySpecifiers == null || arraySpecifiers.size() == 0)
        return;
      SrcType srcType = declarationAttributesStack.peek().srcType;
      assert(arraySpecifiers.size() == 1);
      if (arraySpecifiers.get(0) instanceof BitfieldSpecifier)
        throw new IllegalStateException(
          "bit-field unexpected in array specifiers");
      assert(arraySpecifiers.get(0) instanceof ArraySpecifier);
      ArraySpecifier arraySpecifier = (ArraySpecifier)arraySpecifiers.get(0);
      for (int dimIdx = arraySpecifier.getNumDimensions() - 1;
           dimIdx >= 0; --dimIdx)
      {
        final Expression dimNode = arraySpecifier.getDimension(dimIdx);
        // TODO: Handle other kinds of expressions, including variable-length
        // arrays.
        if (dimNode == null)
          srcType = SrcArrayType.get(srcType);
        else {
          srcSymbolTable.addParentFixup(dimNode, declarator);
          visitTree(dimNode);
          final ValueAndType dim
            = postOrderValuesAndTypes.remove(postOrderValuesAndTypes.size()-1);
          final SrcIntegerType dimIntType
            = dim.getSrcType().toIso(SrcIntegerType.class);
          if (dimIntType == null
              || !(dim.getLLVMValue() instanceof LLVMConstantInteger))
            throw new UnsupportedOperationException(
              "non-constant array dimensions are not yet supported");
          final LLVMConstantInteger dimLLVMValue
            = (LLVMConstantInteger)dim.getLLVMValue();
          final BigInteger dimValue
            = dimIntType.isSigned()
              ? BigInteger.valueOf(dimLLVMValue.getSExtValue())
              : dimLLVMValue.getZExtValue();
          if (dimValue.signum() == -1)
            throw new SrcRuntimeException("array dimension is negative");
          srcType = SrcArrayType.get(srcType, dimValue.longValue());
        }
      }
      declarationAttributesStack.peek().srcType = srcType;
    }

    /**
     * Set the type at the top of {@link #declarationAttributesStack} using
     * the type already there as the return type of a function type with
     * parameter types from {@link #postOrderSrcTypes}.
     * 
     * @param nParams
     *          the number of parameters, whose types are at the top of
     *          {@link #postOrderSrcTypes}, where a null element represents an
     *          ellipsis
     * @param forNestedDeclarator
     *          whether this is being called for a {@link NestedDeclarator}
     *          (as opposed to a {@link ProcedureDeclarator}) and thus does
     *          not specify the parameter list of any {@link Procedure}
     */
    protected void mergeParamsToSrcType(int nParams,
                                        boolean forNestedDeclarator)
    {
      boolean isVarArg = false;
      ArrayList<SrcType> paramTypes = new ArrayList<>(nParams);
      final int firstParamIdx = postOrderSrcTypes.size() - nParams;
      if (nParams == 0) {
        // In procedure declarator without procedure definition, empty
        // parameter list leaves parameters unspecified.  Clang handles this by
        // specifying "..." as the entire parameter list.  With a definition,
        // an empty parameter list specifies zero parameters.  If the parameter
        // list is for a function specified in the return type
        // (forNestedDeclarator must be true) or in the parameter list
        // (isProcedureDefinition must be false because a VariableDeclaration
        // is the ancestor declaration), then that function does not have a
        // definition here, so "..." is the parameter list.
        isVarArg = !declarationAttributesStack.peek().isProcedureDefinition
                   || forNestedDeclarator;
      }
      else if (nParams == 1
               && postOrderSrcTypes.get(firstParamIdx).eqv(SrcVoidType))
      {
        // Void parameter list specifies zero parameters.
      }
      else {
        for (int paramIdx = firstParamIdx;
             paramIdx < postOrderSrcTypes.size(); ++paramIdx)
        {
          if (postOrderSrcTypes.get(paramIdx) == null)
            isVarArg = true;
          else {
            SrcType paramType = postOrderSrcTypes.get(paramIdx);
            final SrcArrayType paramArrayType
              = paramType.toIso(SrcArrayType.class);
            if (paramArrayType != null)
              paramType = SrcPointerType.get(paramArrayType.getElementType());
            else if (paramType.iso(SrcFunctionType.class))
              paramType = SrcPointerType.get(paramType);
            paramTypes.add(paramType);
          }
        }
      }
      postOrderNames.setSize(postOrderNames.size() - nParams);
      postOrderSrcTypes.setSize(firstParamIdx);
      SrcType[] paramTypesArray = new SrcType[paramTypes.size()];
      paramTypes.toArray(paramTypesArray);

      // Build function type.
      declarationAttributesStack.peek().srcType
        = SrcFunctionType.get(declarationAttributesStack.peek().srcType,
                              isVarArg, paramTypesArray);
    }

    /**
     * Compute the {@link SrcType} from the type {@link List} attached to a
     * {@link SizeofExpression}, {@link OffsetofExpression},
     * {@link Typecast}, {@link VaArgExpression}, or any NVL-C built-in
     * function call node. Also, call {@link SrcSymbolTable#addParentFixup}
     * for the type's declarator node, if any, and any descendant array
     * dimension expressions. (Cetus moves struct/union definitions and thus
     * any bit-field width expressions to a node outside of any operand, so
     * {@link SrcSymbolTable#addParentFixup} won't be called for that here.)
     * 
     * @param parent
     *          the {@link SizeofExpression}, {@link OffsetofExpression},
     *          {@link Typecast}, {@link VaArgExpression}, or NVL-C built-in
     *          function call node.
     * @param types
     *          the type list from {@code parent}
     * @return the {@link SrcType} specified by {@code types}
     */
    protected SrcType computeSrcTypeFromList(
      Expression parent, @SuppressWarnings("rawtypes") List types)
    {
      // The type list Cetus creates on these nodes takes one of three forms
      // (so far):
      // 
      //   1. VariableDeclaration-like Specifier series
      //   2. VariableDeclaration-like Specifier series,
      //      PointerSpecifier series
      //   3. VariableDeclaration-like Specifier series,
      //      one VariableDeclaration-like Declarator tree
      // 
      // For comparison, in a VariableDeclaration, the third form is always
      // the case except the Declarator tree is a child. Any PointerSpecifier
      // series from the second form would appear in the Declarator tree.
      final List<Specifier> specs = new ArrayList<>();
      final List<Specifier> ptrSpecs = new ArrayList<>();
      Declarator declarator = null;
      {
        int i = 0;
        for (i = 0;
             i < types.size()
             && types.get(i) instanceof Specifier
             && !(types.get(i) instanceof PointerSpecifier);
             ++i)
          specs.add((Specifier)types.get(i));
        for (;i < types.size() && types.get(i) instanceof PointerSpecifier;
             ++i)
          ptrSpecs.add((PointerSpecifier)types.get(i));
        if (i < types.size() && types.get(i) instanceof Declarator)
          declarator = (Declarator)types.get(i++);
        assert(i == types.size());
      }
      pushDeclarationAttributes(parent, false, specs, false, declarator);
      final SrcType srcType;
      if (declarator == null) {
        if (!ptrSpecs.isEmpty())
          mergeSpecifiersToSrcType(ptrSpecs);
        srcType = declarationAttributesStack.peek().srcType;
      }
      else {
        assert(ptrSpecs.isEmpty());
        srcSymbolTable.addParentFixup(declarator, parent);
        visitTree(declarator);
        postOrderNames.remove(postOrderNames.size() - 1);
        srcType = postOrderSrcTypes.remove(postOrderSrcTypes.size() - 1);
      }
      declarationAttributesStack.pop();
      return srcType;
    }

    /**
     * Declare an LLVM function upon a {@link ProcedureDeclarator} or upon
     * a {@link VariableDeclarator} whose type is a typedef name for a
     * procedure.
     * 
     * @param declarator
     *          the {@link Declarator}
     * @param declaratorFnType
     *          the function type
     */
    protected void declareFunction(Declarator declarator,
                                   SrcFunctionType declaratorFnType)
    {
      // If this is a block-scope declaration, ISO C99 says that the only
      // storage class specifier permitted is extern, that having no storage
      // class specifier is the same as extern, that linkage in that case is
      // dictated by the prior visible declaration of this identifier, and
      // that the linkage is external if there is no prior visible declaration
      // or if the prior visible declaration has no linkage. Thus, by those
      // rules, the only way a block-scope declaration of a function has
      // internal linkage is if there's a visible file-scope declaration with
      // internal linkage. However, ISO C99 also says that behavior is
      // undefined if a single translation unit contains the same identifier
      // with both external and internal linkage, and that can happen when the
      // prior visible declaration is at block scope and has no linkage but
      // the file-scope declaration has internal linkage. In that case, if the
      // file-scope declaration precedes the block-scope declaration, gcc
      // 4.2.1 chooses to resolve this undefined behavior by treating them as
      // separate identifiers, but clang (clang-600.0.54) chooses to treat
      // them as the same identifier with internal linkage (and clang
      // complains when they don't have the same type because, perhaps, the
      // file-scope declaration is not a function). If, instead, the
      // block-scope declaration precedes the file-scope declaration, both gcc
      // and clang complain that a static declaration follows a non-static
      // declaration. The clang approach is easy, so we go with that. That is,
      // in all cases, we assume the block-scope declaration of a function
      // refers to the file-scope declaration and thus has the same linkage, or
      // the block-scope declaration has external linkage if there is no
      // file-scope declaration.

      // Look up the previous file-scope and block-scope versions of the
      // function.
      final String name = declarator.getID().getName();
      final ValueAndType oldGlobalFn
        = srcSymbolTable.getFunction(llvmModuleIndex, name);
      final ValueAndType oldLocalFn;
      if (!declarationAttributesStack.peek().isGlobal) {
        final SymbolTable enclosingScope
          = IRTools.getAncestorOfType(declarator, SymbolTable.class);
        final Symbol oldSym
          = SymbolTools.getSymbolOfName(name, enclosingScope.getParent());
        final ValueAndType oldValueAndType = srcSymbolTable.getLocal(oldSym);
        oldLocalFn
          = oldValueAndType == null
            || !oldValueAndType.getSrcType().iso(SrcFunctionType.class)
            ? null : oldValueAndType;
      }
      else
        oldLocalFn = null;

      // Compute and record the new global function type.
      final SrcFunctionType newGlobalFnType;
      final SrcFunctionType localFnType;
      final LLVMFunction oldLLVMFn;
      if (oldGlobalFn == null) {
        // if we had previously created a local, we would have created a global
        assert(oldLocalFn == null);
        newGlobalFnType = localFnType = declaratorFnType;
        oldLLVMFn = null;
      }
      else {
        oldLLVMFn = (LLVMFunction)oldGlobalFn.getLLVMValue();
        // Compute the composite type of two function declarations as required
        // by ISO C99 sec. 6.2.7p4-5.
        final SrcFunctionType compositeFnType;
        final ValueAndType oldVisibleFn = oldLocalFn != null ? oldLocalFn
                                                             : oldGlobalFn;
        compositeFnType
          = oldVisibleFn.getSrcType().toEqv(SrcFunctionType.class)
            .buildCompositeBald(
              declaratorFnType,
              "incompatible declarations of function \"" + name + "\"");
        if (!declarationAttributesStack.peek().isGlobal)
          newGlobalFnType
            = oldGlobalFn.getSrcType().toIso(SrcFunctionType.class);
        else
          newGlobalFnType = compositeFnType;
        localFnType = compositeFnType;
      }

      // Build the function and rewrite its definition for the composite type
      // if necessary.
      final LLVMFunction newLLVMFn;
      if (oldGlobalFn != null && newGlobalFnType.eqv(oldGlobalFn.getSrcType()))
        newLLVMFn = oldLLVMFn;
      else {
        newLLVMFn = new LLVMFunction(llvmModule, name,
                                     newGlobalFnType.getLLVMType(llvmContext));
        srcSymbolTable.addFunction(llvmModuleIndex, name,
                                   new ValueAndType(newLLVMFn,
                                                    newGlobalFnType, true));
        if (oldLLVMFn != null && !oldLLVMFn.isDeclaration()) {
          // We've already seen the function definition, so keep it. However,
          // the code we generated for that definition's body assumes the
          // composite function type as of that definition, but the composite
          // function type might be different now due to the current
          // declaration. For example, the return type or a parameter type
          // might be a pointer-to-array type, but the size of that array type
          // might have been unspecified by the time of the function
          // definition while specified in the new composite function type.
          // Our approach is to rewrite the generated LLVM function with the
          // new composite function type and insert into the definition
          // bitcasts from/to each new parameter/return type to/from the old
          // one. Thus, like clang (3.5.1) and gcc (4.2.1), we make sure
          // future references to the function assume the composite function
          // type.
          //
          // However, our approach differs from clang's and gcc's in other
          // ways. TODO: First and most importantly, when generating the body
          // of the function, clang and gcc seem to always observe the exact
          // parameter types specified at the function definition (rather than
          // the composite parameter types computed by that point) for any
          // references to the parameters, but they, like us, observe the
          // composite parameter and return types for any recursive calls to
          // the function. For example:
          // 
          //   int (*fn(int (*p)[5]))[5];
          //   int (*fn(int (*p)[]))[] {
          //     sizeof *p; // clang/gcc complain incomplete type
          //     int (*p1)[3] = p; // we complain incompatible type
          //     sizeof *fn(0); // agree size is complete
          //     int (*p2)[3] = fn(0); // agree incompatible type
          //     fn(p1); // agree incompatible type
          //   }
          // 
          // Both clang and gcc complain that the first sizeof is being
          // applied to an incomplete type, but we assume its size is
          // specified as it is within the composite parameter type. However,
          // clang and gcc permit the assignment of p to p1 without warning
          // because p's array size isn't specified, but we complain their
          // types are incompatible because the array sizes are different in
          // the composite type. And yet, for the sizeof and assignment
          // involving the function calls, we all observe the composite type,
          // and so we all agree that the sizeof is OK and the assignment and
          // call are not. Unfortunately, it's not clear to me what ISO C99
          // specifies as the correct behavior here. 6.9.1p7 actually seems to
          // imply that we should observe the exact parameter types at the
          // definition from definition on, but that would contradict
          // 6.2.7p4-5's requirement that we compute composite types.
          // 
          // Second, where there is no complaint, clang (not sure about gcc)
          // generates the function definition's prototype (as opposed to its
          // body) with the composite function type computed as of the function
          // definition (as do we), so it adds bitcasts (we don't need to for
          // this case) to adjust to/from the exact parameter and return types
          // declared at the function definition. Third, clang (not sure about
          // gcc) does not adjust the generated function definition to reflect
          // the composite function type computed at declarations after the
          // definition. Instead, it bitcasts any future references to the
          // function to the new composite function type. These second and
          // third differences from our approach appear to be only
          // implementation details but might prove important as we try to
          // adjust the first difference listed above.
          final LLVMInstructionBuilder builder
            = new LLVMInstructionBuilder(llvmContext);
          // We need to add a dummy basic block so we have an insertion
          // position. Otherwise, LLVM fails an assertion on the
          // BasicBlock.moveAfter call below.
          {
            final LLVMBasicBlock dummyBB
              = basicBlockSetStack.createDummyBasicBlock("", newLLVMFn);
            for (LLVMBasicBlock bb = oldLLVMFn.getFirstBasicBlock();
                 bb.getInstance() != null; bb = bb.getNextBasicBlock())
            {
              bb.moveAfter(newLLVMFn.getLastBasicBlock());
              if (!newGlobalFnType.getReturnType().iso(SrcVoidType)) {
                final LLVMType newLLVMRetType
                  = newGlobalFnType.getReturnType().getLLVMType(llvmContext);
                for (LLVMInstruction insn = bb.getFirstInstruction();
                     insn.getInstance() != null;
                     insn = insn.getNextInstruction())
                {
                  if (insn instanceof LLVMReturnInstruction) {
                    builder.positionBuilderBefore(insn);
                    insn.setOperand(
                      0,
                      LLVMBitCast.create(builder, ".castReturn",
                                         insn.getOperand(0), newLLVMRetType));
                  }
                }
              }
            }
            dummyBB.delete();
          }
          builder.positionBuilderBefore(newLLVMFn.getFirstBasicBlock()
                                        .getFirstInstruction());
          for (LLVMArgument oldArg = oldLLVMFn.getFirstParameter(),
                            newArg = newLLVMFn.getFirstParameter();
               oldArg.getInstance() != null;
               oldArg = oldArg.getNextParameter(),
               newArg = newArg.getNextParameter())
          {
            newArg.setValueName(oldArg.getValueName());
            if (newArg.typeOf() != oldArg.typeOf())
              oldArg.replaceAllUsesWith(
                LLVMBitCast.create(builder,
                                   oldArg.getValueName() + ".castParam",
                                   newArg, oldArg.typeOf()));
            else
              oldArg.replaceAllUsesWith(newArg);
          }
        }
      }

      // Record any local declaration.
      if (!declarationAttributesStack.peek().isGlobal) {
        final LLVMValue cast
          = LLVMConstantExpression.bitCast(
              newLLVMFn,
              LLVMPointerType.get(localFnType.getLLVMType(llvmContext), 0));
        srcSymbolTable.addLocalFnDecl(
          declarationAttributesStack.peek().topDeclarator,
          localFnType, cast, jumpScopeTable);
      }

      // Compute the function's linkage.
      if (declarationAttributesStack.peek().hasStatic) {
        if (!declarationAttributesStack.peek().isGlobal)
          throw new SrcRuntimeException(
            "block-scope function declaration \"" + name
            + "\" has static specifier");
        if (oldLLVMFn != null
            && oldLLVMFn.getLinkage() == LLVMExternalLinkage)
          throw new SrcRuntimeException("linkage for function \"" + name
                                        + "\" redefined");
      }
      if (declarationAttributesStack.peek().hasStatic
          || oldLLVMFn != null
             && oldLLVMFn.getLinkage() == LLVMInternalLinkage)
        newLLVMFn.setLinkage(LLVMInternalLinkage);
      else if (declarationAttributesStack.peek().isGlobal) {
        // This function has external linkage, and so it might have an inline
        // definition. Moreover, this is a file-scope declaration/definition,
        // which can affect whether there's an inline definition.
        final InlineDefState oldInlineDefState;
        {
          final InlineDefState o
            = srcSymbolTable.getFunctionInlineDefState(llvmModuleIndex, name);
          oldInlineDefState = o==null ? InlineDefState.INLINE_C99_SO_FAR : o;
        }
        final InlineDefState inlineDefState;
        if (oldInlineDefState == InlineDefState.INLINE_GNU
            || oldInlineDefState == InlineDefState.INLINE_GNU_NOT)
          inlineDefState = oldInlineDefState;
        else if (declarationAttributesStack.peek().isProcedureDefinition
                 && procedure.getAttributeSpecifier() != null
                 && procedure.getAttributeSpecifier()
                    .contains("__gnu_inline__"))
        {
          // GNU inline semantics
          if (declarationAttributesStack.peek().hasInline
              && declarationAttributesStack.peek().hasExtern)
            inlineDefState = InlineDefState.INLINE_GNU;
          else
            inlineDefState = InlineDefState.INLINE_GNU_NOT;
        }
        else {
          // C99 inline semantics
          if (oldInlineDefState == InlineDefState.INLINE_C99_SO_FAR
              && declarationAttributesStack.peek().hasInline
              && !declarationAttributesStack.peek().hasExtern)
            // So far, all file-scope declarations specify inline without
            // extern, so we have an inline definition so far.
            inlineDefState = InlineDefState.INLINE_C99_SO_FAR;
          else
            // We found a file-scope declaration without inline or with extern,
            // so record that there is no inline definition.
            inlineDefState = InlineDefState.INLINE_C99_NOT;
        }
        srcSymbolTable.setFunctionInlineDefState(llvmModuleIndex, name,
                                                 inlineDefState);
        if (inlineDefState == InlineDefState.INLINE_C99_SO_FAR
            || inlineDefState == InlineDefState.INLINE_GNU)
        {
          // LLVM permits the available_externally linkage on definitions
          // but not declarations. There might be a definition here that
          // hasn't yet been added, or the definition might come from a
          // previous declaration.
          if (declarationAttributesStack.peek().isProcedureDefinition
              || !newLLVMFn.isDeclaration())
            newLLVMFn.setLinkage(LLVMAvailableExternallyLinkage);
        }
        else {
          // If we previously saw a definition and all declarations up to then
          // met the conditions for an inline definition, we set the linkage
          // accordingly, so correct it now.
          newLLVMFn.setLinkage(LLVMExternalLinkage);
        }
      }
      else if (oldLLVMFn != null)
        newLLVMFn.setLinkage(oldLLVMFn.getLinkage());

      // If any declaration/definition specifies inline, add an inline hint.
      // ISO C99 leaves the implementation of inlining pretty open, and it's
      // not clear exactly when the inline specifier can or should be ignored.
      // For example, clang (3.5.1) seems to ignore any inline specifier on a
      // function declaration appearing after the function's definition. We
      // never the ignore the inline specifier, even on a block-scope
      // declaration of a function.
      if (declarationAttributesStack.peek().isProcedureDefinition
          && procedure.getAttributeSpecifier() != null
          && procedure.getAttributeSpecifier().contains("__always_inline__")
          || oldLLVMFn != null
             && oldLLVMFn.hasAttribute(LLVMAlwaysInlineAttribute))
        newLLVMFn.addAttribute(LLVMAlwaysInlineAttribute);
      else if (declarationAttributesStack.peek().hasInline
               || oldLLVMFn != null
                  && oldLLVMFn.hasAttribute(LLVMInlineHintAttribute))
      {
        newLLVMFn.addAttribute(LLVMInlineHintAttribute);
        // ISO C99 says any externally linked inline function shall have a
        // definition in this translation unit, but clang (3.5.1) and gcc
        // (4.2.1) only warn about this situation and only sometimes. We
        // always warn about it.
        //
        // So far, findSymbol always finds the procedure definition if one
        // exists even if other declarations for this procedure exist. ISO C99
        // does not permit block-scope definitions of functions, so we only
        // need to search at file scope.
        if (!(tu.findSymbol(declarator.getID()) instanceof Procedure)
            && (newLLVMFn.getLinkage() == LLVMExternalLinkage
                || newLLVMFn.getLinkage() == LLVMAvailableExternallyLinkage))
          warn(warningsAsErrors,
               "externally linked inline function \"" + name
               + "\" has no definition in this translation unit");
      }

      // LLVM doesn't allow multiple file-scope declarations/definitions for
      // the same identifier. Replace any old LLVM declaration with an updated
      // declaration or definition.
      if (oldLLVMFn != null && oldLLVMFn != newLLVMFn) {
        oldLLVMFn.replaceAllUsesWith(
          LLVMConstantExpression.bitCast(newLLVMFn, oldLLVMFn.typeOf()));
        oldLLVMFn.delete();
        // LLVM appended a suffix because the original name was taken, so
        // reset the name now that the original name is not taken.
        newLLVMFn.setValueName(name);
      }

      if (declarationAttributesStack.peek().isProcedureDefinition) {
        srcFunctionType = newGlobalFnType;
        llvmFunction = newLLVMFn;
      }
    }

    /**
     * Push a character constant's value and type to
     * {@link #postOrderValuesAndTypes}.
     * 
     * @param wide
     *          whether it's a wide character constant
     * @param inVal
     *          the unsigned value of the character constant. For example, for
     *          a {@code '\xff'} from the source, {@code inVal} should be 255
     *          regardless of whether the type char is signed or unsigned. If
     *          the type char is signed and 8 bits (see {@link #SrcCharType}),
     *          the value pushed to {@link #postOrderValuesAndTypes} would
     *          then be an int with value -1.
     */
    protected void handleCharacterConstant(boolean wide, long inVal) {
      if (wide) {
        // TODO: Are we handling multibyte characters properly? See ISO C99
        // 6.4.4.4p11's mention of mbtowc.
        final SrcPrimitiveIntegerType type = SRC_WCHAR_TYPE;
        postOrderValuesAndTypes.add(new ValueAndType(
          LLVMConstantInteger.get(type.getLLVMType(llvmContext), inVal,
                                  type.isSigned()),
          type, false));
        return;
      }
      // ISO C99 sec. 6.4.4.4p2 defines an integer character constant as a
      // single-quoted character sequence without a leading "L". ISO C99 sec.
      // 6.4.4.3p10 says an integer character constant has type int not char.
      // The reason appears to be the possibility of multi-character integer
      // character constants, such as 'xy', whose handling is
      // implementation-defined:
      // http://stackoverflow.com/questions/20764538/type-of-character-constant
      // Cetus throws an error for a multi-character integer character
      // constant, clang and gcc warn about them by default, and so we don't
      // handle them. Nevertheless, the type is still int.
      final SrcPrimitiveIntegerType type = SRC_CHAR_CONST_TYPE;
      // ISO C99 sec. 6.4.4.4p13 makes it clear that we have to interpret the
      // value of an integer character constant as a char first before
      // sign-extending to an int. The last sentence of p10 also says this.
      final LLVMConstantInteger valChar
        = LLVMConstantInteger.get(SrcCharType.getLLVMType(llvmContext), inVal,
                                  SrcCharType.isSigned());
      LLVMConstant valInt
        = SrcCharType.isSigned()
          ? valChar.signExtend(type.getLLVMType(llvmContext))
          : valChar.zeroExtend(type.getLLVMType(llvmContext));
      postOrderValuesAndTypes.add(new ValueAndType(valInt, type, false));
    }

    /** Call the like-named method for each of {@link #pragmaTranslators}. */
    protected void translateStandalonePragmas(AnnotationStatement node) {
      for (PragmaTranslator p : pragmaTranslators)
        p.translateStandalonePragmas(node);
    }

    /** Call the like-named method for each of {@link #pragmaTranslators}. */
    protected void startStatementPragmas(Statement node) {
      for (PragmaTranslator p : pragmaTranslators)
        p.startStatementPragmas(node);
    }

    /** Call the like-named method for each of {@link #pragmaTranslators}. */
    protected void endStatementPragmas(Statement node) {
      for (PragmaTranslator p : pragmaTranslators)
        p.endStatementPragmas(node);
    }

    /** Call the like-named method for each of {@link #pragmaTranslators}. */
    protected void checkLabeledStatementPragmas(Statement node) {
      for (PragmaTranslator p : pragmaTranslators)
        p.checkLabeledStatementPragmas(node);
    }

    /**
     * This class implements an idiom for evaluating an expression whose
     * result will be examined and then discarded.
     * 
     * <p>
     * TODO: Evaluation of initializers or array dimensions at file scope
     * could use this. That is, we could create a throw-away function before
     * evaluating them so that we no longer have to check for file scope when
     * evaluating various expressions that cannot be evaluated without an
     * enclosing function. To find code that can then be eliminated, grep all
     * code for comparisons of llvmFunction, llvmBuilder, procedure,
     * initDestination.stackAddr, etc. [!=]= null.
     * </p>
     */
    protected class FakeFunctionForEval {
      private final SrcFunctionType oldSrcFunctionType;
      private final LLVMFunction oldLLVMFunction;
      private final LLVMInstructionBuilder oldLLVMBuilder;
      /**
       * Back up current function, if any, and set the current function to a
       * new temporary function that can be discarded after the evaluation.
       * The temporay function can receive all LLVM instructions generated
       * during the evaluation.
       * 
       * @param forWhat
       *          description of the purpose of the evaluation. This is used
       *          as part of the temporary function's name for the sake of
       *          debugging.
       */
      public FakeFunctionForEval(String forWhat) {
        oldSrcFunctionType = srcFunctionType;
        oldLLVMFunction = llvmFunction;
        oldLLVMBuilder = llvmBuilder;
        srcFunctionType = SrcFunctionType.get(SrcVoidType, false);
        llvmFunction = new LLVMFunction(
          llvmModule, "."+forWhat+".fakeFnForEval",
          srcFunctionType.getLLVMType(llvmContext));
        llvmBuilder = new LLVMInstructionBuilder(llvmContext);
        llvmBuilder.positionBuilderAtEnd(
          basicBlockSetStack.registerBasicBlock(".entry", llvmFunction));
      }
      /**
       * Delete the temporary function and restore the original function, if
       * any.
       */
      public void restore() {
        llvmFunction.delete();
        srcFunctionType = oldSrcFunctionType;
        llvmFunction = oldLLVMFunction;
        llvmBuilder = oldLLVMBuilder;
      }
    }

    public void visit(@SuppressWarnings("deprecation") AccessLevel node) {
      throw new SrcRuntimeException("non-C construct not supported");
    }
    public void visit(AnnotationDeclaration node) {}
    /** Pruned traversal so can perform actions pre and post. */
    public void visit(ClassDeclaration node) {
      // Must be a struct or a union.
      final boolean isStruct = node.getKey() == ClassDeclaration.STRUCT;
      if (!isStruct && node.getKey() != ClassDeclaration.UNION)
        throw new SrcRuntimeException(node.getKey() + " not supported");
      // Get tag. So far, Cetus always adds a tag to a tag-less struct or
      // union, so we don't need to handle that case specially.
      final String tag = node.getName().getName();
      // Resolve the type's scope and construct an opaque LLVM struct for it
      // if it doesn't already exist.
      //
      // A ClassDeclaration is a SymbolTable, but it does not represent a
      // scope in C, so we start the lookup at the ClassDeclaration's parent.
      //
      // A ClassDeclaration represents an explicit declaration or an explicit
      // definition of a struct or union, so the last argument here is true.
      SrcStructOrUnionType srcStructOrUnionType
        = srcSymbolTable.getSrcStructOrUnionType(
            isStruct, tag, node.getParent(), true, llvmContext);

      // Now that the struct or union has been constructed as an opaque LLVM
      // struct, iterate children and gather their types, which might include
      // a reference to this type.
      final int postOrderNamesOldSize = postOrderNames.size();
      if (node.getChildren() != null) { // null if there's no member list
        FlatIterator<Traversable> itr = new FlatIterator<>(node);
        while (itr.hasNext())
          visitTree(itr.next());
      }

      // node.getSymbols() does not return unnamed members (bit-fields or
      // C11 anonymous structures/unions), and node.getChildren() is not
      // always the same size as the number of fields because declarators
      // are sometimes grouped within declarations.
      final int nMembers = postOrderNames.size() - postOrderNamesOldSize;
      if (nMembers > 0) {
        final int nameIdxBegin = postOrderNames.size() - nMembers;
        final int typeIdxBegin = postOrderSrcTypes.size() - nMembers;
        final String[] memberNames = new String[nMembers];
        final SrcType[] memberTypes = new SrcType[nMembers];
        for (int nameIdx = nameIdxBegin, typeIdx = typeIdxBegin, destIdx = 0;
             destIdx < nMembers;
             ++nameIdx, ++typeIdx, ++destIdx)
        {
          memberNames[destIdx] = postOrderNames.get(nameIdx);
          SrcType memberType = postOrderSrcTypes.get(typeIdx);
          final SrcArrayType memberArrayType
            = memberType.toIso(SrcArrayType.class);
          if (memberArrayType != null) {
            // For flexible array member.
            if (!memberArrayType.numElementsIsSpecified()
                && isStruct && destIdx == nMembers-1)
              memberType = SrcArrayType.get(memberArrayType.getElementType(),
                                            0);
          }
          else {
            final SrcBitFieldType memberBitFieldType
              = memberType.toIso(SrcBitFieldType.class);
            if (memberBitFieldType != null
                && memberBitFieldType.getWidth() == 0
                && !memberNames[destIdx].isEmpty())
              throw new SrcRuntimeException("zero-width bit-field is named");
          }
          memberTypes[destIdx] = memberType;
        }
        srcStructOrUnionType.setBody(memberNames, memberTypes, llvmTargetData,
                                     llvmContext);
        postOrderNames.setSize(nameIdxBegin);
        postOrderSrcTypes.setSize(typeIdxBegin);
      }
    }
    /**
     * Pre-order traversal.
     * 
     * So far, every child is a {@link VariableDeclarator}, whose visit method
     * checks if its parent is an {@link Enumeration}, so we don't need to do
     * anything special here to let descendants know they're in an
     * enumeration, but we do have to initialize the implicit initializer
     * that is incremented at each enumerator.
     */
    public void visit(Enumeration node) {
      srcSymbolTable.addEnum(node, new SrcEnumType(node.getName().getName(),
                                                   node.getChildren().size()));
      enumeratorLLVMInit
        = LLVMConstantInteger.get(SRC_ENUM_CONST_TYPE.getLLVMType(llvmContext),
                                  0, false);
    }
    public void visit(LinkageSpecification node) {
      throw new SrcRuntimeException("non-C construct not supported");
    }
    public void visit(Namespace node) {
      throw new SrcRuntimeException("non-C construct not supported");
    }
    public void visit(NamespaceAlias node) {
      throw new SrcRuntimeException("non-C construct not supported");
    }
    public void visit(PreAnnotation node) {}
    /**
     * Function definitions only. Function prototype is visited as a
     * {@link ProcedureDeclarator} within a {@link VariableDeclaration}.
     *
     * Pruned traversal so can perform actions pre and post. Also, Cetus does
     * not include the {@link Procedure}'s {@link ProcedureDeclarator} in the
     * traversal, but we need to visit it.
     */
    public void visit(Procedure node) {
      jumpScopeTable.startScope(node);

      assert(node.getParent() == tu);
      procedure = node;
      pushDeclarationAttributes(node.getParent(), true, node.getSpecifiers(),
                                false, node.getDeclarator());
      srcSymbolTable.addParentFixup(node.getDeclarator(), node);

      // Build function prototype.
      // This sets srcFunctionType and llvmFunction.
      visitTree(node.getDeclarator());

      // Create entry block.
      final LLVMBasicBlock entryBB
        = basicBlockSetStack.registerBasicBlock(".entry", llvmFunction);
      llvmBuilder.positionBuilderAtEnd(entryBB);

      // For each parameter, set name and create stack allocation.
      //
      // We could probably add each store instruction next to the stack
      // allocation instruction, but it's more consistent with local
      // variables if all stack allocation instructions are together
      // up front.
      final int nparams = (int)llvmFunction.countParameters();
      final LLVMUser[] stackedParams = new LLVMUser[nparams];
      for (int i = 0; i < nparams; ++i) {
        final Declaration paramDeclaration = node.getParameter(i);
        assert(paramDeclaration.getChildren().size() == 1);
        final Declarator paramDeclarator
          = (Declarator)paramDeclaration.getChildren().get(0);
        final LLVMArgument llvmParam = llvmFunction.getParameter(i);
        llvmParam.setValueName(paramDeclarator.getID().getName());
        final ValueAndType stackedParam = srcSymbolTable.addNonStaticLocalVar(
          llvmModule, llvmBuilder,
          "function definition parameter "+(i+1)+" type",
          paramDeclarator, llvmParam.getValueName() + ".stackedParam",
          srcFunctionType.getParamTypes()[i], true, true, jumpScopeTable,
          null);
        stackedParams[i] = (LLVMStackAllocation)stackedParam.getLLVMValue();
      }
      for (int i = 0; i < nparams; ++i)
        srcFunctionType.getParamTypes()[i].store(
          true, stackedParams[i], llvmFunction.getParameter(i), llvmModule,
          llvmBuilder);

      if (!srcFunctionType.getReturnType().iso(SrcVoidType))
        srcFunctionType.getReturnType()
        .checkStaticOrAutoObject("function definition return type", true);

      // Record each contained label before traversing the body so that
      // forward references to labels are possible.
      {
        final DepthFirstIterator<Traversable> labelItr
          = new DepthFirstIterator<>(node.getBody());
        while (labelItr.hasNext()) {
          final Traversable n = labelItr.next();
          if (!(n instanceof Label))
            continue;
          final Label label = (Label)n;
          srcSymbolTable.addLabel(
            label,
            basicBlockSetStack.createBasicBlockEarly(
              label.getName().getName(), llvmFunction));
        }
      }

      // Build function body and add final return instruction.
      visitTree(node.getBody());
      {
        LLVMType retType = llvmFunction.getFunctionType().getReturnType();
        if (retType.getTypeKind() != LLVMVoidTypeKind)
          new LLVMReturnInstruction(llvmBuilder,
                                    LLVMConstant.constNull(retType));
        else
          new LLVMReturnInstruction(llvmBuilder, null);
      }

      llvmFunction = null;
      srcFunctionType = null;
      declarationAttributesStack.pop();
      procedure = null;

      jumpScopeTable.endScope(node);
    }
    public void visit(TemplateDeclaration node) {
      throw new SrcRuntimeException("non-C construct not supported");
    }
    public void visit(UsingDeclaration node) {
      throw new SrcRuntimeException("non-C construct not supported");
    }
    public void visit(UsingDirective node) {
      throw new SrcRuntimeException("non-C construct not supported");
    }
    /** Pruned traversal so can perform actions pre and post. */
    public void visit(VariableDeclaration node) {
      pushDeclarationAttributes(
        node.getParent(), false, node.getSpecifiers(),
        node.getNumDeclarators() == 1
        && node.getDeclarator(0) instanceof VariableDeclarator
        && ((VariableDeclarator)node.getDeclarator(0)).getSymbolName()
           .equals("..."),
        null);
      SrcType srcType = declarationAttributesStack.peek().srcType;
      FlatIterator<Traversable> itr = new FlatIterator<>(node);
      while (itr.hasNext()) {
        Declarator declarator = (Declarator)itr.next();
        declarationAttributesStack.peek().topDeclarator = declarator;
        visitTree(declarator);
        declarationAttributesStack.peek().srcType = srcType;
      }
      declarationAttributesStack.pop();
    }
    /** Pruned traversal so can perform actions pre and post. */
    public void visit(NestedDeclarator node) {
      mergeSpecifiersToSrcType(node.getSpecifiers());
      mergeArraySpecifiersToSrcType(node.getArraySpecifiers(), node);
      if (node.getParameters() != null) {
        FlatIterator<Traversable> itr = new FlatIterator<>(node);
        while (itr.hasNext()) {
          Traversable child = itr.next();
          // If there's an initializer, Cetus attaches it as a child of the
          // topmost declarator (necessarily a NestedDeclarator if there's
          // more than one level) not the bottom-most. However, we don't
          // traverse it until we reach the bottom-most.
          if (!(child instanceof Declarator)
              && !(child instanceof Initializer))
            visitTree(child);
        }
        mergeParamsToSrcType(node.getParameters().size(), true);
      }
      visitTree(node.getDeclarator());
    }
    public void visit(ProcedureDeclarator node) {
      mergeSpecifiersToSrcType(node.getSpecifiers());
      assert(node.getParameters() != null);
      mergeParamsToSrcType(node.getParameters().size(), false);
      assert(node.getArraySpecifiers() == null);
      final SrcFunctionType funcType
        = declarationAttributesStack.peek().srcType
          .toIso(SrcFunctionType.class);
      if (declarationAttributesStack.peek().isTypedef) {
        // A typedef declaration cannot have a procedure definition, so the
        // ancestor Declaration must be returned here (null would be
        // returned here if Procedure were the logical ancestor
        // Declaration).
        assert(node.getDeclaration() != null);
        srcSymbolTable.addTypedef(
          declarationAttributesStack.peek().topDeclarator, funcType);
        return;
      }
      if (declarationAttributesStack.peek().getNameAndType) {
        postOrderNames.add(node.getID().getName());
        postOrderSrcTypes.add(funcType);
        return;
      }
      declareFunction(node, funcType);
    }
    /** Pruned traversal so can perform actions pre and post. */
    public void visit(VariableDeclarator node) {
      // If it's an enumerator, store its value and increment the value for the
      // next enumerator.
      if (node.getParent() instanceof Enumeration) {
        if (node.getInitializer() != null) {
          // ISO C99 only allows integer constant expressions to initialize
          // enumerators, but we don't bother to disallow expressions we can
          // convert to integer constants, such as floating point constants.
          initDestinationStack.push(new InitDestination(SRC_ENUM_CONST_TYPE));
          visitTree(node.getInitializer());
          initDestinationStack.pop();
          enumeratorLLVMInit
            = (LLVMConstant)postOrderValuesAndTypes.remove(
                              postOrderValuesAndTypes.size()-1)
                            .getLLVMValue();
        }
        srcSymbolTable.getEnum((Enumeration)node.getParent())
          .addMember(node.getID().getName(), enumeratorLLVMInit);
        enumeratorLLVMInit
          = LLVMConstantExpression.add(
              enumeratorLLVMInit,
              LLVMConstantInteger.get(
                SRC_ENUM_CONST_TYPE.getLLVMType(llvmContext), 1, false));
        return;
      }

      // Fail if this is a function definition. Cetus lets you get away with
      // this if the topmost declarator is a NestedDeclarator, so we catch it
      // here. For example: int (*fn)[] {}
      if (declarationAttributesStack.peek().isProcedureDefinition)
        throw new SrcRuntimeException("function definition does not have a"
                                      + " valid function prototype");

      // The initializer is attached to the topmost declarator, and Cetus
      // stores the symbol by its topmost declarator, so find it.
      final Declarator topDeclarator
        = declarationAttributesStack.peek().topDeclarator;

      // Build the type.
      mergeSpecifiersToSrcTypeUnchecked(node.getSpecifiers());
      {
        // For some reason, Cetus places any bit-field specifier with the
        // array specifiers, so we need to extract it.
        final List<ArraySpecifier> arraySpecs = new ArrayList<>();
        @SuppressWarnings("rawtypes")
        final Iterator itr = node.getArraySpecifiers().iterator();
        BitfieldSpecifier bitFieldSpec = null;
        while (itr.hasNext()) {
          Object spec = itr.next();
          if (spec instanceof ArraySpecifier)
            arraySpecs.add((ArraySpecifier)spec);
          else if (spec instanceof BitfieldSpecifier) {
            assert(!itr.hasNext());
            bitFieldSpec = (BitfieldSpecifier)spec;
          }
          else
            throw new IllegalStateException(
              "unexpected kind of specifier in VariableDeclarator's array"
               + " specifiers: " + spec);
        }
        mergeArraySpecifiersToSrcType(arraySpecs, node);
        if (bitFieldSpec != null) {
          final SrcType bitFieldDeclaredType
            = declarationAttributesStack.peek().srcType;
          // ISO C99 sec. 6.7.2.1p4 says _Bool, int, and unsigned int or
          // other implementation-defined types. Clang (3.5.1) seems to
          // support any integer type, and running SPEC CPU 2006 found uses of
          // at least unsigned short.
          final SrcIntegerType bitFieldDeclaredIntType
            = bitFieldDeclaredType.toIso(SrcIntegerType.class);
          if (bitFieldDeclaredIntType == null)
            throw new SrcRuntimeException("bit-field has non-integer type: "
                                          + bitFieldDeclaredType);
          srcSymbolTable.addParentFixup(bitFieldSpec.getWidthExpression(),
                                        node);
          visitTree(bitFieldSpec.getWidthExpression());
          final ValueAndType widthExpr
            = postOrderValuesAndTypes.remove(postOrderValuesAndTypes.size()-1);
          final SrcIntegerType widthExprIntType
            = widthExpr.getSrcType().toIso(SrcIntegerType.class);
          if (widthExprIntType == null
              || !(widthExpr.getLLVMValue() instanceof LLVMConstantInteger))
            throw new SrcRuntimeException(
              "bit-field width expression is not a constant integer"
              + " expression");
          final LLVMConstantInteger widthValue
            = (LLVMConstantInteger)widthExpr.getLLVMValue();
          final long width
            = widthExprIntType.isSigned()
              ? widthValue.getSExtValue()
              : widthValue.getZExtValue().longValue();
          if (width > bitFieldDeclaredIntType.getLLVMWidth())
            throw new SrcRuntimeException(
              "bit-field width is too large for type");
          declarationAttributesStack.peek().srcType
            = new SrcBitFieldType(bitFieldDeclaredIntType, width);
        }
      }
      final SrcType declaratorType = declarationAttributesStack.peek()
                                     .srcType;

      // If it's a typedef, param, member declaration, or just a type
      // specification, store its type.
      if (declarationAttributesStack.peek().isTypedef) {
        srcSymbolTable.addTypedef(topDeclarator, declaratorType);
        return;
      }
      if (declarationAttributesStack.peek().getNameAndType) {
        postOrderNames.add(node.getID().getName());
        postOrderSrcTypes.add(declaratorType);
        return;
      }
      // struct/union members should already be handled.
      assert(!declaratorType.iso(SrcBitFieldType.class));

      // If it's a function declaration (whose type was specified using a
      // typedef name or else this would be a ProcedureDeclarator instead of a
      // VariableDeclarator), generate the LLVM function declaration.
      {
        final SrcFunctionType declaratorFunctionType
          = declaratorType.toIso(SrcFunctionType.class);
        if (declaratorFunctionType != null) {
          declareFunction(node, declaratorFunctionType);
          return;
        }
      }

      // Compute the name that will be used for the variable in LLVM.
      final String srcName = node.getID().getName();
      final String llvmName;
      if (!declarationAttributesStack.peek().isGlobal
          && declarationAttributesStack.peek().hasStatic)
        llvmName = procedure.getSymbolName() + "." + srcName;
      else
        llvmName = srcName;

      // We have a file-scope declaration or block-scope declaration of a
      // variable.

      // Handle block-scope declaration without extern.
      if (!declarationAttributesStack.peek().isGlobal
          && !declarationAttributesStack.peek().hasExtern)
      {
        final ValueAndType lvalueAndType;
        if (topDeclarator.getInitializer() == null) {
          {
            final SrcArrayType srcArrayType
              = declaratorType.toIso(SrcArrayType.class);
            if (srcArrayType != null) {
              // This is what gcc (4.2.1) and clang (clang-600.0.56) do. Unlike
              // for tentative definitions for variables of array type, there
              // seems to be no provision for assuming an array size of one for
              // block-scope definitions of variables, even when declared
              // static.
              if (!srcArrayType.numElementsIsSpecified())
                throw new SrcRuntimeException(
                  "variable has array type of unspecified size");
            }
          }
          if (declarationAttributesStack.peek().hasStatic)
            lvalueAndType = srcSymbolTable.addStaticLocalVar(
              llvmModule, topDeclarator, llvmName,
              declaratorType, true, false,
              declarationAttributesStack.peek().hasThread, jumpScopeTable,
              null);
          else
            lvalueAndType = srcSymbolTable.addNonStaticLocalVar(
              llvmModule, getLLVMAllocaBuilder(), null, topDeclarator,
              llvmName, declaratorType, true, false, jumpScopeTable, null);
        }
        else {
          final ValueAndType declaratorLvalueAndType;
          if (declarationAttributesStack.peek().hasStatic) {
            declaratorLvalueAndType = srcSymbolTable.addStaticLocalVar(
              llvmModule, topDeclarator, llvmName,
              declaratorType, false, true,
              declarationAttributesStack.peek().hasThread, jumpScopeTable,
              null);
            initDestinationStack.push(new InitDestination(declaratorType));
          }
          else {
            declaratorLvalueAndType = srcSymbolTable.addNonStaticLocalVar(
              llvmModule, getLLVMAllocaBuilder(), null, topDeclarator,
              llvmName, declaratorType, false, true, jumpScopeTable, null);
            // This must be inserted after any instructions that build its
            // value, so it doesn't necessarily belong in the entry block.
            initDestinationStack.push(new InitDestination(declaratorType,
                                                          llvmName));
          }
          visitTree(topDeclarator.getInitializer());
          initDestinationStack.pop();
          final ValueAndType initResult
            = postOrderValuesAndTypes.remove(postOrderValuesAndTypes.size()-1);
          if (declarationAttributesStack.peek().hasStatic) {
            final LLVMGlobalVariable oldVar
              = (LLVMGlobalVariable)declaratorLvalueAndType.getLLVMValue();
            lvalueAndType = srcSymbolTable.addStaticLocalVar(
              llvmModule, topDeclarator, llvmName,
              initResult.getSrcType(), true, true,
              declarationAttributesStack.peek().hasThread, jumpScopeTable,
              // Just in case it's an array whose size is specified by the init.
              declaratorType.eqv(initResult.getSrcType()) ? oldVar : null);
            final LLVMGlobalVariable var
              = (LLVMGlobalVariable)lvalueAndType.getLLVMValue();
            if (var != oldVar)
              srcSymbolTable.replaceLLVMGlobalVariable(oldVar, var, llvmName);
            var.setInitializer((LLVMConstant)initResult.getLLVMValue());
          }
          else {
            final LLVMStackAllocation oldAlloc
              = (LLVMStackAllocation)declaratorLvalueAndType.getLLVMValue();
            final LLVMStackAllocation alloc
              = (LLVMStackAllocation)initResult.getLLVMValue();
            oldAlloc.replaceAllUsesWith(
              LLVMBitCast.create(getLLVMAllocaBuilder(), llvmName+".inInit",
                                 alloc, oldAlloc.typeOf()));
            oldAlloc.eraseFromParent();
            alloc.setValueName(llvmName);
            lvalueAndType = srcSymbolTable.addNonStaticLocalVar(
              llvmModule, getLLVMAllocaBuilder(), null, topDeclarator,
              llvmName, initResult.getSrcType(), true, true, jumpScopeTable,
              alloc);
          }
        }
        return;
      }

      // We have a file-scope declaration or block-scope extern declaration of
      // a variable.

      // Look up the previous file-scope and block-scope versions of the
      // variable.
      final ValueAndType oldGlobalVar
        = srcSymbolTable.getGlobalVar(llvmModuleIndex, llvmName);
      final ValueAndType oldLocalVar;
      if (!declarationAttributesStack.peek().isGlobal) {
        final SymbolTable enclosingScope
          = IRTools.getAncestorOfType(node, SymbolTable.class);
        final Symbol oldSym
          = SymbolTools.getSymbolOfName(srcName, enclosingScope.getParent());
        final ValueAndType oldValueAndType = srcSymbolTable.getLocal(oldSym);
        LLVMValue val = null;
        if (oldValueAndType != null) {
          val = oldValueAndType.getLLVMValue();
          while (val instanceof LLVMConstantExpression)
            val = ((LLVMConstantExpression)val).getOperand(0);
        }
        oldLocalVar = val instanceof LLVMGlobalVariable ? oldValueAndType
                                                        : null;
      }
      else
        oldLocalVar = null;

      // Compute the new global variable type without considering the
      // initializer.
      final SrcType newGlobalVarTypeWithoutInit;
      final SrcType localVarType;
      final LLVMGlobalVariable oldLLVMVar;
      if (oldGlobalVar == null) {
        // if we had previously created a local, we would have created a global
        assert(oldLocalVar == null);
        newGlobalVarTypeWithoutInit = localVarType = declaratorType;
        oldLLVMVar = null;
      }
      else {
        oldLLVMVar = (LLVMGlobalVariable)oldGlobalVar.getLLVMValue();
        // Compute the composite type of the two global variable declarations
        // as required by ISO C99 sec. 6.2.7p4-5.
        final SrcType compositeVarType;
        final ValueAndType oldVisibleVar = oldLocalVar != null
                                           ? oldLocalVar : oldGlobalVar;
        compositeVarType = oldVisibleVar.getSrcType().buildComposite(
          declaratorType,
          "incompatible declarations of file-scope variable \""+srcName
          +"\"");
        if (!declarationAttributesStack.peek().isGlobal)
          newGlobalVarTypeWithoutInit = oldGlobalVar.getSrcType();
        else
          newGlobalVarTypeWithoutInit = compositeVarType;
        localVarType = compositeVarType;
      }

      // Compute any initializer and any type change (to the global
      // definition) that results.
      final LLVMConstant init;
      final SrcType newGlobalVarType;
      final LLVMGlobalVariable inInitLLVMVar;
      if (declarationAttributesStack.peek().hasExtern
          && topDeclarator.getInitializer() != null)
        throw new SrcRuntimeException(
          "extern declaration of variable \"" + srcName
          + "\" has initializer");
      else if (oldLLVMVar != null && !oldLLVMVar.isDeclaration()) {
        if (topDeclarator.getInitializer() != null)
          throw new SrcRuntimeException(
            "file-scope variable \"" + srcName + "\" redefined");
        newGlobalVarType = newGlobalVarTypeWithoutInit;
        // The bitcast is required in case the composite type is different
        // than the original type. Interestingly, clang (3.5.1) doesn't appear
        // to ever alter the type of a previously generated LLVM global
        // variable declaration that had an initializer, but it does assume
        // the composite type for future references to the variable.
        init = LLVMConstantExpression.bitCast(
                 oldLLVMVar.getInitializer(),
                 newGlobalVarType.getLLVMType(llvmContext));
        inInitLLVMVar = null;
      }
      else if (topDeclarator.getInitializer() != null) {
        // The initializer here is computed in terms of the composite type. If
        // it were the other way around, the following declarations would be
        // incompatible because they would have different sizes:
        //
        //   int a[3];
        //   int a[] = {1, 2}; // 3 elements, last is zero
        initDestinationStack.push(
          new InitDestination(newGlobalVarTypeWithoutInit));
        final ValueAndType inInitVar = srcSymbolTable.addGlobalVar(
          llvmModule, llvmModuleIndex, srcName, llvmName,
          llvmName+".tmpInInit", newGlobalVarTypeWithoutInit, false,
          declarationAttributesStack.peek().hasThread, null);
        inInitLLVMVar = (LLVMGlobalVariable)inInitVar.getLLVMValue();
        visitTree(topDeclarator.getInitializer());
        initDestinationStack.pop();
        final ValueAndType initValueAndType
          = postOrderValuesAndTypes.remove(postOrderValuesAndTypes.size()-1);
        // Just in case it's an array whose size is specified by the init.
        newGlobalVarType = initValueAndType.getSrcType();
        init = (LLVMConstant)initValueAndType.getLLVMValue();
      }
      else {
        newGlobalVarType = newGlobalVarTypeWithoutInit;
        init = null;
        inInitLLVMVar = null;
      }

      // Record the tentative definition status. If it has an initializer in
      // any declaration, then it has an external definition. Else, if it has
      // no extern specifier in at least one declaration, then it has a
      // tentative definition. Else, it's an extern declaration.
      if (init != null)
        srcSymbolTable.recordGlobalVarWithoutTentativeDef(llvmModuleIndex,
                                                          llvmName);
      else if (!declarationAttributesStack.peek().hasExtern)
        srcSymbolTable.recordGlobalVarWithTentativeDef(llvmModuleIndex,
                                                       llvmName);

      // Build the global variable and set its initializer.
      final ValueAndType newVar = srcSymbolTable.addGlobalVar(
        llvmModule, llvmModuleIndex, srcName, llvmName, llvmName,
        newGlobalVarType, init != null,
        declarationAttributesStack.peek().hasThread,
        oldGlobalVar != null
        && newGlobalVarType.eqv(oldGlobalVar.getSrcType())
        ? oldLLVMVar : null);
      final LLVMGlobalVariable newLLVMVar
        = (LLVMGlobalVariable)newVar.getLLVMValue();
      if (init != null)
        newLLVMVar.setInitializer(init);

      // Record any local declaration.
      if (!declarationAttributesStack.peek().isGlobal) {
        final LLVMValue cast
          = LLVMConstantExpression.bitCast(
              newLLVMVar,
              LLVMPointerType.get(localVarType.getLLVMType(llvmContext), 0));
        srcSymbolTable.addLocalExternVar(
          declarationAttributesStack.peek().topDeclarator, localVarType, cast,
          jumpScopeTable);
      }

      // Compute the variable's linkage.
      if (declarationAttributesStack.peek().hasStatic) {
        if (declarationAttributesStack.peek().hasExtern)
          throw new SrcRuntimeException(
            "extern declaration of variable \"" + srcName
            + "\" has static specifier");
        if (oldLLVMVar != null
            && oldLLVMVar.getLinkage() == LLVMExternalLinkage)
          throw new SrcRuntimeException(
            "linkage for file-scope variable \"" + srcName + "\" redefined");
      }
      if (declarationAttributesStack.peek().hasStatic
          || oldLLVMVar != null
             && oldLLVMVar.getLinkage() == LLVMInternalLinkage)
        newLLVMVar.setLinkage(LLVMInternalLinkage);

      // LLVM doesn't allow multiple file-scope declarations/definitions for
      // the same identifier. Replace any old LLVM declaration with an updated
      // declaration or definition.
      srcSymbolTable.replaceLLVMGlobalVariable(oldLLVMVar, newLLVMVar,
                                               llvmName);
      srcSymbolTable.replaceLLVMGlobalVariable(inInitLLVMVar, newLLVMVar,
                                               llvmName);
    }
    public void visit(AlignofExpression node) {
      // https://gcc.gnu.org/onlinedocs/gcc-4.9.2/gcc/Alignment.html#Alignment
      final SrcType srcType;
      if (node.getExpression() != null) {
        // Get type from expression operand.
        final FakeFunctionForEval fakeFunctionForEval
          = new FakeFunctionForEval("alignof");
        visitTree(node.getExpression());
        srcType = postOrderValuesAndTypes
                  .remove(postOrderValuesAndTypes.size()-1).getSrcType();
        fakeFunctionForEval.restore();
      }
      else
        srcType = computeSrcTypeFromList(node, node.getTypes());
      if (srcType.iso(SrcFunctionType.class))
        throw new SrcRuntimeException("alignof operand is of function type");
      if (srcType.isIncompleteType())
        throw new SrcRuntimeException("alignof operand is of incomplete type");
      if (srcType.iso(SrcBitFieldType.class))
        throw new SrcRuntimeException("alignof operand is a bit-field");
      // TODO: Handle alignof for variable length?
      postOrderValuesAndTypes.add(new ValueAndType(
        LLVMConstantInteger.get(
          SRC_SIZE_T_TYPE.getLLVMType(llvmContext),
          llvmTargetData.abiAlignmentOfType(srcType.getLLVMType(llvmContext)),
          true),
        SRC_SIZE_T_TYPE, false));
    }
    public void visit(ArrayAccess node) {
      assert(node.getChildren().size() == 1+node.getNumIndices());
      final int first = postOrderValuesAndTypes.size() - node.getNumIndices();
      ValueAndType op1 = postOrderValuesAndTypes.get(first-1);
      for (int i = 0; i < node.getNumIndices(); ++i) {
        final ValueAndType op2 = postOrderValuesAndTypes.get(first + i);
        op1 = ValueAndType.arraySubscript(op1, op2, llvmModule, llvmBuilder);
      }
      postOrderValuesAndTypes.setSize(first-1);
      postOrderValuesAndTypes.add(op1);
    }
    /** Pruned traversal so can perform actions pre and post. */
    public void visit(BinaryExpression node) {
      final BinaryOperator oper = node.getOperator();

      // Handle && and ||.
      // 
      // We might need to inject code around the evaluation of the second
      // operand in order to short-circuit its evaluation depending on the
      // result of the first operand, so a post-order traversal isn't possible.
      // The visitor for ConditionalExpression does something similar.
      if (oper == BinaryOperator.LOGICAL_AND
          || oper == BinaryOperator.LOGICAL_OR)
      {
        final String operName
          = node.getOperator() == BinaryOperator.LOGICAL_AND
            ? ".logicalAnd" : ".logicalOr";
        final SrcType resultType = SrcIntType;
        // Evaluate the first operand as an i1 value.
        visitTree(node.getLHS());
        final LLVMValue op1AsI1
          = postOrderValuesAndTypes.remove(postOrderValuesAndTypes.size()-1)
            .evalAsCond("first operand to binary \"" + oper.toString()
                        + "\"",
                        llvmModule, llvmBuilder);

        // If not at file scope, create a new basic block to hold the second
        // operand's evaluation.
        final LLVMBasicBlock op1LastBB, op2FirstBB;
        if (llvmFunction == null) {
          // We're at file scope, so we cannot create basic blocks, so
          // proceed as if the second operand is a constant expression and
          // thus has no side effect and thus does not require
          // short-circuiting behavior. If it isn't actually a constant
          // expression, we'll report an error when we try to use the result
          // as an initializer of a global variable (or when we try to
          // evaluate some construct for which we cannot generate LLVM IR
          // outside a function).
          op1LastBB = null;
          op2FirstBB = null;
        }
        else {
          // We're not at file scope, so we can safely create a basic block
          // to hold instructions evaluating the second operand. However,
          // we'll delete that basic block later if it turns out the second
          // operand is a constant expression and thus the basic block is
          // empty. If it's not a constant expression but it should have been
          // because we're creating the initializer for a static local, we'll
          // report an error when we try to use the result as the static
          // local's initializer.
          op1LastBB = llvmBuilder.getInsertBlock();
          op2FirstBB = basicBlockSetStack.createBasicBlockEarly(
            operName+".op2FirstBB", llvmFunction);
          llvmBuilder.positionBuilderAtEnd(op2FirstBB);
        }

        // Evaluate the second operand as an i1 value.
        visitTree(node.getRHS());
        final LLVMValue op2AsI1
          = postOrderValuesAndTypes.remove(postOrderValuesAndTypes.size()-1)
            .evalAsCond("second operand to binary \"" + oper.toString()
                         + "\"",
                         llvmModule, llvmBuilder);
        final LLVMBasicBlock op2LastBB
          = llvmFunction == null ? null : llvmBuilder.getInsertBlock();

        // Perform the && or || and obtain an i1 result value.
        //
        // Insert short-circuiting instructions only when both of the
        // following are true: we're inside a function (we cannot otherwise),
        // and the second operand is a non-constant expression. The latter
        // restriction is required to handle static local variable
        // initializers.
        final LLVMUser resultAsI1;
        if (llvmFunction == null || op2AsI1 instanceof LLVMConstant) {
          if (op2FirstBB != null) {
            // If we allocated a basic block for the second operand (because
            // we're in a function), delete it now because we're not using
            // it and because we never added a terminator to the original
            // basic block.
            llvmBuilder.positionBuilderAtEnd(op1LastBB);
            assert(op2FirstBB.getFirstInstruction().getInstance() == null);
            op2FirstBB.delete();
          }
          // For valid C code, the value name shouldn't matter here as we
          // should be dealing with constant expressions. For invalid C
          // code, it might help with debugging.
          if (oper == BinaryOperator.LOGICAL_AND)
            resultAsI1 = LLVMAndInstruction.create(
              llvmBuilder, operName+".i1", op1AsI1, op2AsI1);
          else {
            assert(oper == BinaryOperator.LOGICAL_OR);
            resultAsI1 = LLVMOrInstruction.create(
              llvmBuilder, operName+".i1", op1AsI1, op2AsI1);
          }
        }
        else {
          basicBlockSetStack.registerBasicBlock(op2FirstBB);
          final LLVMBasicBlock resultBB
            = basicBlockSetStack.registerBasicBlock(operName+".resultBB",
                                                      llvmFunction);
          new LLVMBranchInstruction(llvmBuilder, resultBB);
          llvmBuilder.positionBuilderAtEnd(op1LastBB);
          new LLVMBranchInstruction(
            llvmBuilder, op1AsI1,
            oper == BinaryOperator.LOGICAL_AND ? op2FirstBB : resultBB,
            oper == BinaryOperator.LOGICAL_AND ? resultBB : op2FirstBB);
          llvmBuilder.positionBuilderAtEnd(resultBB);
          final LLVMPhiNode resultPHI
            = new LLVMPhiNode(llvmBuilder, operName+".i1",
                              op1AsI1.typeOf()/*i1*/);
          resultPHI.addIncoming(new LLVMValue[]{op1AsI1, op2AsI1},
                                new LLVMBasicBlock[]{op1LastBB, op2LastBB});
          resultAsI1 = resultPHI;
        }
        assert(resultAsI1.typeOf() == LLVMIntegerType.get(llvmContext, 1));

        // Extend the result from an i1 to an int and store it.
        final LLVMValue resultValue
          = LLVMExtendCast.create(llvmBuilder, operName+".int", resultAsI1,
                                  SrcIntType.getLLVMType(llvmContext),
                                  ExtendType.ZERO);
        postOrderValuesAndTypes.add(new ValueAndType(resultValue,
                                                     resultType, false));
        return;
      }

      // For the remaining operators we can and must generate code for both
      // operands before performing any of the operation.
      visitTree(node.getLHS());
      visitTree(node.getRHS());

      // Handle remaining binary operators.
      final int idx = postOrderValuesAndTypes.size()-2;
      final ValueAndType op1 = postOrderValuesAndTypes.get(idx);
      final ValueAndType op2 = postOrderValuesAndTypes.get(idx + 1);
      postOrderValuesAndTypes.setSize(idx);
      final ValueAndType result;
      if (oper == BinaryOperator.MULTIPLY)
        result = ValueAndType.multiply(op1, op2, llvmModule, llvmBuilder);
      else if (oper == BinaryOperator.DIVIDE)
        result = ValueAndType.divide(op1, op2, llvmModule, llvmBuilder);
      else if (oper == BinaryOperator.MODULUS)
        result = ValueAndType.remainder(op1, op2, llvmModule, llvmBuilder);
      else if (oper == BinaryOperator.ADD)
        result = ValueAndType.add("binary \"+\"", op1, op2, llvmModule,
                                  llvmBuilder);
      else if (oper == BinaryOperator.SUBTRACT)
        result = ValueAndType.subtract(
          "binary \"-\"", op1, op2, llvmTargetData, llvmModule, llvmBuilder);
      else if (oper == BinaryOperator.SHIFT_LEFT)
        result = ValueAndType.shift(op1, op2, false,
                                    llvmModule, llvmBuilder);
      else if (oper == BinaryOperator.SHIFT_RIGHT)
        result = ValueAndType.shift(op1, op2, true,
                                    llvmModule, llvmBuilder);
      else if (oper == BinaryOperator.COMPARE_LT)
        result = ValueAndType.relational(op1, op2, false, false,
                                        llvmModule, llvmBuilder);
      else if (oper == BinaryOperator.COMPARE_GT)
        result = ValueAndType.relational(op1, op2, true, false,
                                         llvmModule, llvmBuilder);
      else if (oper == BinaryOperator.COMPARE_LE)
        result = ValueAndType.relational(op1, op2, false, true,
                                         llvmModule, llvmBuilder);
      else if (oper == BinaryOperator.COMPARE_GE)
        result = ValueAndType.relational(op1, op2, true, true,
                                         llvmModule, llvmBuilder);
      else if (oper == BinaryOperator.COMPARE_EQ)
        result = ValueAndType.equality(op1, op2, true,
                                       llvmModule, llvmBuilder);
      else if (oper == BinaryOperator.COMPARE_NE)
        result = ValueAndType.equality(op1, op2, false,
                                       llvmModule, llvmBuilder);
      else if (oper == BinaryOperator.BITWISE_AND)
        result = ValueAndType.andXorOr(op1, op2, AndXorOr.AND,
                                       llvmModule, llvmBuilder);
      else if (oper == BinaryOperator.BITWISE_EXCLUSIVE_OR)
        result = ValueAndType.andXorOr(op1, op2, AndXorOr.XOR,
                                       llvmModule, llvmBuilder);
      else if (oper == BinaryOperator.BITWISE_INCLUSIVE_OR)
        result = ValueAndType.andXorOr(op1, op2, AndXorOr.OR,
                                       llvmModule, llvmBuilder);
      else
        throw new IllegalStateException("oper: " + oper);
      postOrderValuesAndTypes.add(result);
    }
    /**
     * Pruned traversal because {@link AccessExpression} is a subtype of
     * {@link BinaryExpression}, which has pruned traversal.
     */
    public void visit(AccessExpression node) {
      {
        FlatIterator<Traversable> itr = new FlatIterator<>(node);
        while (itr.hasNext())
          visitTree(itr.next());
      }
      // Get operator and member name.
      final AccessOperator op = (AccessOperator)node.getOperator();
      if (op != AccessOperator.MEMBER_ACCESS       // .
          && op != AccessOperator.POINTER_ACCESS)  // ->
        throw new SrcRuntimeException("invalid member access operator: "
                                      + op);
      final boolean isArrow = op == AccessOperator.POINTER_ACCESS;
      final String member = ((Identifier)node.getRHS()).getName();

      // Get left-hand side value and its type.
      final ValueAndType exprNoPrep
        = postOrderValuesAndTypes.remove(postOrderValuesAndTypes.size() - 1);
      final ValueAndType expr
        = isArrow ? exprNoPrep.prepareForOp(llvmModule, llvmBuilder)
                  : exprNoPrep;
      final SrcType exprType = expr.getSrcType();

      // Get struct/union type.
      final SrcStructOrUnionType structOrUnionType;
      final EnumSet<SrcTypeQualifier> typeQualifiers;
      if (isArrow) {
        final SrcPointerType ptrType = exprType.toIso(SrcPointerType.class);
        if (ptrType == null)
          throw new SrcRuntimeException(
            "left-hand side of -> is not a pointer");
        structOrUnionType = ptrType.getTargetType()
                            .toIso(SrcStructOrUnionType.class);
        typeQualifiers = ptrType.getTargetType().expandSrcTypeQualifiers();
        if (structOrUnionType == null)
          throw new SrcRuntimeException(
            "left-hand side of -> is not a pointer to a struct or union");
      }
      else {
        typeQualifiers = exprType.expandSrcTypeQualifiers();
        structOrUnionType = exprType.toIso(SrcStructOrUnionType.class);
        if (structOrUnionType == null)
          throw new SrcRuntimeException(
            "left-hand side of . is not a struct or union");
      }

      // Get member and store result.
      //
      // We cannot create an alloca builder at file scope (where
      // llvmFunction=null) because an alloca builder requires a function.
      // According to StructOrUnionType.accessMember's preconditions, we can
      // provide a null instead of an alloca builder if the member access
      // operator is operating on an LLVM pointer, which here corresponds to
      // either a pointer rvalue or a struct or union lvalue. The only other
      // possible expression we can be operating on here is a struct or union
      // rvalue, so we need to be sure we'll never have a struct or union
      // rvalue at file scope in C.
      //
      // It's my understanding that a struct or union rvalue is never a
      // constant expression in C (it's either a function call result, or
      // it's an lvalue converted to an rvalue, which requires a load
      // instruction), so valid C code should never encounter this situation.
      // For invalid C code, we normally complain about non-constant
      // expressions at file scope when we set the initializer for a
      // file-scope variable or when we set the size of an array, but that's
      // too late to avoid crashing our compiler here. There are only three
      // C constructs I know of that can evaluate to a struct or union
      // rvalue: function call, conditional operator, and comma operator. A
      // function call at file scope is generally not allowed in C, so
      // FunctionCall's visitor method complains if one is evaluated at file
      // scope. The conditional operator and comma operator are allowed at
      // file scope, so we catch those cases and any others that might arise
      // here as a generic complaint about struct or union rvalues at file
      // scope.
      //
      // TODO: See FakeFunctionForEval for how we might clean up some of this.
      if (llvmFunction == null) {
        assert(expr.getLLVMValue() instanceof LLVMConstant
               == isArrow || expr.isLvalue());
        if (!(expr.getLLVMValue() instanceof LLVMConstant)) {
          throw new SrcRuntimeException(
            "struct or union rvalue at file scope is not a constant"
            + " expression");
        }
      }
      postOrderValuesAndTypes.add(
        structOrUnionType.accessMember(typeQualifiers, expr.getLLVMValue(),
                                       member, isArrow || expr.isLvalue(),
                                       llvmModule, llvmBuilder,
                                       llvmFunction == null
                                       ? null : getLLVMAllocaBuilder()));
    }
    /**
     * Pruned traversal because {@link AssignmentExpression} is a subtype of
     * {@link BinaryExpression}, which has pruned traversal.
     */
    public void visit(AssignmentExpression node) {
      visitTree(node.getLHS());
      visitTree(node.getRHS());
      final int idx = postOrderValuesAndTypes.size()-2;
      final ValueAndType lhs = postOrderValuesAndTypes.get(idx);
      final ValueAndType rhs = postOrderValuesAndTypes.get(idx+1);
      postOrderValuesAndTypes.setSize(idx);
      final ValueAndType val;
      final String operator;
      if (node.getOperator() == AssignmentOperator.NORMAL) {
        operator = "simple assignment operator";
        val = rhs;
      }
      else if (node.getOperator() == AssignmentOperator.MULTIPLY) {
        operator = "\"*=\"";
        val = ValueAndType.multiply(lhs, rhs, llvmModule, llvmBuilder);
      }
      else if (node.getOperator() == AssignmentOperator.DIVIDE) {
        operator = "\"/=\"";
        val = ValueAndType.divide(lhs, rhs, llvmModule, llvmBuilder);
      }
      else if (node.getOperator() == AssignmentOperator.MODULUS) {
        operator = "\"%=\"";
        val = ValueAndType.remainder(lhs, rhs, llvmModule, llvmBuilder);
      }
      else if (node.getOperator() == AssignmentOperator.ADD) {
        operator = "\"+=\"";
        val = ValueAndType.add(operator, lhs, rhs, llvmModule, llvmBuilder);
      }
      else if (node.getOperator() == AssignmentOperator.SUBTRACT) {
        operator = "\"-=\"";
        val = ValueAndType.subtract(operator, lhs, rhs, llvmTargetData,
                                    llvmModule, llvmBuilder);
      }
      else if (node.getOperator() == AssignmentOperator.SHIFT_LEFT) {
        operator = "\"<<=\"";
        val = ValueAndType.shift(lhs, rhs, false, llvmModule, llvmBuilder);
      }
      else if (node.getOperator() == AssignmentOperator.SHIFT_RIGHT) {
        operator = "\">>=\"";
        val = ValueAndType.shift(lhs, rhs, true, llvmModule, llvmBuilder);
      }
      else if (node.getOperator() == AssignmentOperator.BITWISE_AND) {
        operator = "\"&=\"";
        val = ValueAndType.andXorOr(lhs, rhs, AndXorOr.AND, llvmModule,
                                    llvmBuilder);
      }
      else if (node.getOperator() == AssignmentOperator.BITWISE_EXCLUSIVE_OR) {
        operator = "\"^=\"";
        val = ValueAndType.andXorOr(lhs, rhs, AndXorOr.XOR, llvmModule,
                                    llvmBuilder);
      }
      else if (node.getOperator() == AssignmentOperator.BITWISE_INCLUSIVE_OR) {
        operator = "\"|=\"";
        val = ValueAndType.andXorOr(lhs, rhs, AndXorOr.OR, llvmModule,
                                    llvmBuilder);
      }
      else
        throw new IllegalStateException();
      postOrderValuesAndTypes.add(
        ValueAndType.simpleAssign(operator, true, lhs, val, llvmModule,
                                  llvmBuilder, warningsAsErrors));
    }
    public void visit(CommaExpression node) {
      // Cetus joins successive comma operators into one CommaExpression.
      final int nexprs = node.getChildren().size();
      final int end = postOrderValuesAndTypes.size();
      final ValueAndType result = postOrderValuesAndTypes.get(end-1);
      postOrderValuesAndTypes.setSize(end-nexprs);
      postOrderValuesAndTypes.add(result);
    }
    public void visit(CompoundLiteral node) {
      throw new UnsupportedOperationException(
        "compound literals are not yet supported");
    }
    /** Pruned traversal so can perform actions pre and post. */
    public void visit(ConditionalExpression node) {
      final String operator = "conditional operator";
      final String op1Desc = "first operand to " + operator;
      final String op2Desc = "second operand to " + operator;
      final String op3Desc = "third operand to " + operator;
      final String operName = ".condOp";

      // We might need to inject code around the evaluations of the second and
      // third operands in order to skip one of their evaluations depending on
      // the result of the first operand, so a post-order traversal isn't
      // possible. However, for constant expressions (which are required for
      // initializers of global variables and static local variables), the
      // operands cannot have any side effects, and constant expressions
      // cannot have control flow instructions anyway. The visitor for
      // BinaryExpression does something similar where && and || are handled.

      // Evaluate the operands, starting a new basic block before each of the
      // second and third operands if at file scope.
      final LLVMBasicBlock op2FirstBB, op3FirstBB;
      final LLVMBasicBlock op1LastBB, op2LastBB, op3LastBB;
      final SrcType op2PrepType, op3PrepType;
      final LLVMValue op1AsI1;
      final ValueAndType op2Prep, op3Prep;
      {
        // First operand.
        visitTree(node.getCondition());
        op1AsI1
          = postOrderValuesAndTypes.remove(postOrderValuesAndTypes.size()-1)
            .evalAsCond(op1Desc, llvmModule, llvmBuilder);
        op1LastBB = llvmFunction == null ? null
                                         : llvmBuilder.getInsertBlock();

        // Second operand.
        if (llvmFunction == null)
          op2FirstBB = null;
        else {
          op2FirstBB = basicBlockSetStack.createBasicBlockEarly(
            operName + ".op2FirstBB", llvmFunction);
          llvmBuilder.positionBuilderAtEnd(op2FirstBB);
        }
        visitTree(node.getTrueExpression());
        op2Prep
          = postOrderValuesAndTypes.remove(postOrderValuesAndTypes.size()-1)
            .prepareForOp(llvmModule, llvmBuilder);
        op2PrepType = op2Prep.getSrcType();
        op2LastBB = llvmFunction == null ? null
                                         : llvmBuilder.getInsertBlock();

        // Third operand.
        if (llvmFunction == null)
          op3FirstBB = null;
        else {
          op3FirstBB = basicBlockSetStack.createBasicBlockEarly(
            operName + ".op3FirstBB", llvmFunction);
          llvmBuilder.positionBuilderAtEnd(op3FirstBB);
        }
        visitTree(node.getFalseExpression());
        op3Prep
          = postOrderValuesAndTypes.remove(postOrderValuesAndTypes.size()-1)
            .prepareForOp(llvmModule, llvmBuilder);
        op3PrepType = op3Prep.getSrcType();
        op3LastBB = llvmFunction == null ? null
                                         : llvmBuilder.getInsertBlock();
      }

      // Validate second and third operand types, and compute result type.
      final SrcType resultType;
      final SrcPointerType op2PtrType = op2PrepType
                                        .toIso(SrcPointerType.class);
      final SrcPointerType op3PtrType = op3PrepType
                                        .toIso(SrcPointerType.class);
      if (op2PrepType.iso(SrcArithmeticType.class)
          && op3PrepType.iso(SrcArithmeticType.class))
      {
        resultType
          = SrcArithmeticType.getTypeFromUsualArithmeticConversionsNoPrep(
              operator, op2PrepType, op3PrepType);
      }
      else if (op2PrepType.iso(SrcStructOrUnionType.class)
               && op2PrepType.iso(op3PrepType))
        resultType = op2PrepType;
      else if (op2PrepType.iso(SrcVoidType) && op2PrepType.iso(SrcVoidType))
        resultType = SrcVoidType;
      else if (op2PtrType != null && op3PtrType != null) {
        final SrcType op2TargetType = op2PtrType.getTargetType();
        final SrcType op3TargetType = op3PtrType.getTargetType();
        final EnumSet<SrcTypeQualifier> resultTargetTypeQuals
           = op2TargetType.expandSrcTypeQualifiers();
        resultTargetTypeQuals.addAll(op3TargetType.expandSrcTypeQualifiers());
        final boolean op2HasVoid = op2TargetType.iso(SrcVoidType);
        if (op2HasVoid || op3TargetType.iso(SrcVoidType)) {
          final SrcPointerType otherType
            = op2HasVoid ? op3PtrType : op2PtrType;
          final SrcType otherTargetType
            = op2HasVoid ? op3TargetType : op2TargetType;
          final ValueAndType voidOp
            = op2HasVoid ? op2Prep : op3Prep;
          final boolean isNullPtrConst = voidOp.isNullPointerConstant();
          if (!isNullPtrConst
              && otherType.getTargetType().iso(SrcFunctionType.class))
            throw new SrcRuntimeException(
              "second and third operands to " + operator + " are a function"
              + " pointer and a void pointer that is not a null pointer"
              + " constant");
          if (isNullPtrConst)
            resultType = otherType;
          else
            resultType = SrcPointerType.get(
                           SrcQualifiedType.get(otherTargetType,
                                                resultTargetTypeQuals));
        }
        else {
          final SrcType resultTargetTypeStripped
            = op2TargetType.toIso(SrcBaldType.class).buildComposite(
                op3TargetType.toIso(SrcBaldType.class),
                "second and third operands to "+operator+" have pointer"
                +" types with incompatible target types, neither of which"
                +" is void");
          resultType = SrcPointerType.get(
            SrcQualifiedType.get(resultTargetTypeStripped,
                                 resultTargetTypeQuals));
        }
      }
      else if ((op2PtrType != null) != (op3PtrType != null)) {
        final boolean op2IsPtr = op2PtrType != null;
        final SrcType ptrType = op2IsPtr ? op2PrepType : op3PrepType;
        final SrcType otherType = op2IsPtr ? op3PrepType : op2PrepType;
        if (!otherType.iso(SrcIntegerType.class))
          throw new SrcRuntimeException(
            "of the second and third operands to " + operator
            + ", only one is of pointer type");
        final ValueAndType otherOp = op2IsPtr ? op3Prep : op2Prep;
        if (!otherOp.isNullPointerConstant())
           throw new SrcRuntimeException(
             "second and third operands to " + operator + " are a pointer"
             + " and an integer expression that is not a null pointer"
             + " constant");
        resultType = ptrType;
      }
      else
        throw new SrcRuntimeException(
          "second and third operands to " + operator
          + " have types that cannot be combined: "
          + op2PrepType + " and " + op3PrepType);

      // Convert the second and third operands to the result type.
      final LLVMValue op2ConvValue, op3ConvValue;
      if (llvmFunction != null)
        llvmBuilder.positionBuilderAtEnd(op2LastBB);
      op2ConvValue
        = op2Prep.convertToNoPrep(resultType, op2Desc,
                                  llvmModule, llvmBuilder)
          .getLLVMValue();
      if (llvmFunction != null)
        llvmBuilder.positionBuilderAtEnd(op3LastBB);
      op3ConvValue
        = op3Prep.convertToNoPrep(resultType, op3Desc,
                                  llvmModule, llvmBuilder)
          .getLLVMValue();

      // Select and store the result.
      final LLVMValue resultValue;
      if (llvmFunction == null || (op2ConvValue instanceof LLVMConstant
                                   && op3ConvValue instanceof LLVMConstant))
      {
        // We skipped creating each basic block iff we're at file scope.
        assert((op2FirstBB == null) == (op3FirstBB == null));
        if (op2FirstBB != null) {
          llvmBuilder.positionBuilderAtEnd(op1LastBB);
          assert(op2FirstBB.getFirstInstruction().getInstance() == null);
          assert(op3FirstBB.getFirstInstruction().getInstance() == null);
          op2FirstBB.delete();
          op3FirstBB.delete();
        }
        if (resultType.iso(SrcVoidType))
          resultValue = null;
        else
          resultValue
            = LLVMSelectInstruction.create(llvmBuilder, operName+".result",
                                           op1AsI1, op2ConvValue,
                                           op3ConvValue);
      }
      else {
        basicBlockSetStack.registerBasicBlock(op2FirstBB);
        basicBlockSetStack.registerBasicBlock(op3FirstBB);
        final LLVMBasicBlock resultBB
          = basicBlockSetStack.registerBasicBlock(operName+".resultBB",
                                                    llvmFunction);
        llvmBuilder.positionBuilderAtEnd(op1LastBB);
        new LLVMBranchInstruction(llvmBuilder, op1AsI1, op2FirstBB,
                                  op3FirstBB);
        llvmBuilder.positionBuilderAtEnd(op2LastBB);
        new LLVMBranchInstruction(llvmBuilder, resultBB);
        llvmBuilder.positionBuilderAtEnd(op3LastBB);
        new LLVMBranchInstruction(llvmBuilder, resultBB);
        llvmBuilder.positionBuilderAtEnd(resultBB);
        if (resultType.iso(SrcVoidType))
          resultValue = null;
        else {
          final LLVMPhiNode resultPHI
            = new LLVMPhiNode(llvmBuilder, operName+".result",
                              resultType.getLLVMType(llvmContext));
          resultPHI.addIncoming(new LLVMValue[]{op2ConvValue, op3ConvValue},
                                new LLVMBasicBlock[]{op2LastBB, op3LastBB});
          resultValue = resultPHI;
        }
      }
      postOrderValuesAndTypes.add(new ValueAndType(resultValue, resultType,
                                                   false));
    }
    public void visit(DeleteExpression node) {
      throw new SrcRuntimeException("non-C construct not supported");
    }
    /** Pruned traversal so can perform actions pre and post. */
    public void visit(FunctionCall node) {
      final String name = node.getName() instanceof Identifier
                          ? node.getName().toString() : null;

      if (name != null && name.equals("__builtin_constant_p")) {
        final FakeFunctionForEval fakeFunctionForEval
          = new FakeFunctionForEval("__builtin_constant_p");
        if (node.getNumArguments() != 1)
          throw new SrcRuntimeException("wrong number of arguments to "+name);
        {
          final FlatIterator<Traversable> itr = new FlatIterator<>(node);
          assert(itr.hasNext());
          final Traversable nameNode = itr.next();
          assert(nameNode instanceof Identifier);
          assert(((Identifier)nameNode).getName().equals(name));
          assert(itr.hasNext());
          visitTree(itr.next());
          assert(!itr.hasNext());
        }
        final SrcIntegerType resType = SrcIntType;
        final ValueAndType arg
          = postOrderValuesAndTypes.remove(postOrderValuesAndTypes.size()-1)
            .prepareForOp(llvmModule, llvmBuilder);
        final LLVMValue res = LLVMConstantInteger.get(
          resType.getLLVMType(llvmContext),
          arg.getLLVMValue() instanceof LLVMConstant ? 1 : 0,
          false);
        postOrderValuesAndTypes.add(new ValueAndType(res, resType, false));
        fakeFunctionForEval.restore();
        return;
      }

      // Handle built-in functions.
      final SrcFunctionBuiltin builtin
        = srcSymbolTable.getFunctionBuiltin(name);
      if (builtin != null) {
        {
          final FlatIterator<Traversable> itr = new FlatIterator<>(node);
          assert(itr.hasNext());
          final Traversable nameNode = itr.next();
          assert(nameNode instanceof Identifier);
          assert(((Identifier)nameNode).getName().equals(name));
          while (itr.hasNext())
            visitTree(itr.next());
        }
        if (llvmFunction == null)
          throw new SrcRuntimeException("call to "+name+" at file scope");
        final ValueAndType args[] = new ValueAndType[node.getNumArguments()];
        final int firstArg
          = postOrderValuesAndTypes.size() - node.getNumArguments();
        for (int i = 0, j = firstArg; i < args.length; ++i, ++j)
          args[i] = postOrderValuesAndTypes.get(j);
        postOrderValuesAndTypes.setSize(firstArg);
        postOrderValuesAndTypes.add(
          builtin.call(node, args, srcSymbolTable, llvmModule,
                       llvmModuleIndex, llvmTargetData, llvmBuilder,
                       warningsAsErrors));
        return;
      }

      // Handle normal function calls.
      {
        final FlatIterator<Traversable> itr = new FlatIterator<>(node);
        while (itr.hasNext())
          visitTree(itr.next());
      }
      // We complain about most non-constant expressions at file scope when
      // we set the initializer for a global variable. However, function
      // calls can return struct or union rvalues, and accessing the member
      // of a union requires generating a stack allocation, so we must
      // complain before the AccessExpression is evaluated. See the
      // AccessExpression visitor's comments for further details.
      if (llvmFunction == null)
        throw new SrcRuntimeException("function call at file scope");
      final int firstArg
        = postOrderValuesAndTypes.size() - node.getNumArguments();
      final ValueAndType fnValueAndType
        = postOrderValuesAndTypes.get(firstArg - 1)
          .prepareForOp(llvmModule, llvmBuilder);
      final LLVMValue fn = fnValueAndType.getLLVMValue();
      final SrcPointerType fnPtrType
        = fnValueAndType.getSrcType().toIso(SrcPointerType.class);
      if (fnPtrType == null)
        throw new SrcRuntimeException(
          "call to something that is not a function or pointer to function");
      final SrcFunctionType fnType
        = fnPtrType.getTargetType().toIso(SrcFunctionType.class);
      if (fnType == null)
        throw new SrcRuntimeException(
          "call to something that is not a function or pointer to function");
      final ValueAndType args[] = new ValueAndType[node.getNumArguments()];
      for (int i = 0, j = firstArg; i < args.length; ++i, ++j)
        args[i] = postOrderValuesAndTypes.get(j);
      postOrderValuesAndTypes.setSize(firstArg - 1);
      postOrderValuesAndTypes.add(fnType.call(name, fn, args, llvmModule,
                                              llvmBuilder,
                                              warningsAsErrors));
    }
    public void visit(DestructorID node) {
      throw new SrcRuntimeException("non-C construct not supported");
    }
    public void visit(Identifier node) {
      final String name = node.getName();

      // If this node represents a function builtin being called,
      // FunctionCall's visitor method does not allow this node to be
      // traversed, but it might be traversed for some invalid usage.
      if (srcSymbolTable.getFunctionBuiltin(name) != null)
        throw new SrcRuntimeException("function builtin " + name
                                      + " used not as call");
      if (null
          != srcSymbolTable.getBuiltinTypeTable().lookup(name, llvmModule,
                                                         warningsAsErrors))
        throw new SrcRuntimeException("built-in type " + name
                                      + " used not as type");

      if (name.equals("__func__") || name.equals("__FUNCTION__")
          || name.equals("__PRETTY_FUNCTION__"))
      {
        if (procedure == null)
          throw new SrcRuntimeException(name+" used at file scope");
        final String fnName = procedure.getSymbolName();
        final String llvmVarName = "__func__." + fnName;
        final ValueAndType oldVar
          = srcSymbolTable.getGlobalVar(llvmModuleIndex, llvmVarName);
        if (oldVar != null) {
          postOrderValuesAndTypes.add(oldVar);
          return;
        }
        final SrcType srcType = SrcArrayType.get(SrcCharType,
                                                 fnName.length() + 1);
        final ValueAndType var = srcSymbolTable.addGlobalVar(
          llvmModule, llvmModuleIndex, name, llvmVarName, llvmVarName,
          srcType, true, false, null);
        final LLVMGlobalVariable llvmVar
          = (LLVMGlobalVariable)var.getLLVMValue();
        llvmVar.setInitializer(LLVMConstantArray.get(llvmContext, fnName,
                                                     true));
        // These three properties mimic clang (clang-600.0.56):
        // private unnamed_addr constant
        llvmVar.setLinkage(LLVMLinkage.LLVMPrivateLinkage);
        llvmVar.setConstant(true);
        // TODO: When we have an LLVM whose C API supports it, extend jllvm
        // accordingly, uncomment this, and add check for it in the test
        // suite. It's available by at least LLVM 3.5.0 as
        // LLVMSetUnnamedAddr.
        // See: http://lists.cs.uiuc.edu/pipermail/llvm-commits/Week-of-Mon-20140310/208071.html
        // See similar todo in SrcArrayType.prepareForOp.
        //var.setUnnamedAddr(true);
        postOrderValuesAndTypes.add(var);
        return;
      }

      // Handle member name after a . or -> or in an offsetof. Just return and
      // let the AccessExpression or OffsetofExpression visitor look it up via
      // the SrcStructOrUnionType.
      if (node.getParent() instanceof AccessExpression) {
        final AccessExpression parent = (AccessExpression)node.getParent();
        assert(parent.getChildren().size() == 2);
        if (parent.getChildren().get(1) == node)
          return;
      }
      if (node.getParent() instanceof OffsetofExpression)
        return;

      for (SymbolTable symTable = srcSymbolTable.lookupSymbolTable(node);
           symTable != null;
           symTable = srcSymbolTable.lookupSymbolTable(symTable.getParent()))
      {
        for (Symbol sym : symTable.getSymbols()) {
          if (!sym.getSymbolName().equals(name))
            continue;

          // Handle enumeration constant.
          final SrcEnumType enumType;
          if (sym instanceof VariableDeclarator
              && ((VariableDeclarator)sym).getParent() instanceof Enumeration
              && null !=
                 (enumType = srcSymbolTable.getEnum(
                    (Enumeration)(((VariableDeclarator)sym).getParent()))))
          {
            final SrcType type = SRC_ENUM_CONST_TYPE;
            postOrderValuesAndTypes.add(new ValueAndType(
              enumType.getMember(name), type, false));
            return;
          }

          // Handle local variable, local declaration of function, file-scope
          // variable, or function declared at file scope.
          if (sym instanceof VariableDeclarator
              || sym instanceof ProcedureDeclarator
              || sym instanceof NestedDeclarator
              || sym instanceof Procedure)
          {
            ValueAndType valueAndType = null;
            if (symTable instanceof TranslationUnit) {
              // Checking that we've actually reached file scope before
              // checking our file-scope tables is really important for the
              // following subtle example:
              //
              //   int i;
              //   void fn() {
              //     int i;
              //     {
              //       int j = i;
              //       int i;
              //     }
              //   }
              //
              // At the use of i, we'll climb Cetus's symbol tables to find
              // the i defined after that use. If we check
              // srcSymbolTable.getLocal, that i won't be there because it's
              // not yet defined. If we then check srcSymbolTable.getGlobalVar,
              // which is by name, the wrong i will be found. Instead, we must
              // climb further to find the i defined directly within fn. If
              // that i didn't exist, we would climb to the TranslationUnit and
              // then check srcSymbolTable.getGlobalVar.
              valueAndType = srcSymbolTable.getGlobalVar(llvmModuleIndex,
                                                         name);
              if (valueAndType == null)
                valueAndType = srcSymbolTable.getFunction(llvmModuleIndex,
                                                          name);
            }
            else
              valueAndType = srcSymbolTable.getLocal(sym);
            if (valueAndType != null) {
              postOrderValuesAndTypes.add(valueAndType);
              return;
            }
          }

          // Sym was not in any of our tables, so sym must be a symbol defined
          // later in this scope, so keep searching enclosing scopes.
        }
      }

      throw new SrcRuntimeException("unknown identifier: " + name);
    }
    public void visit(NameID node) {}
    public void visit(OperatorID node) {
      throw new SrcRuntimeException("non-C construct not supported");
    }
    public void visit(QualifiedID node) {
      throw new SrcRuntimeException("non-C construct not supported");
    }
    public void visit(TemplateID node) {
      throw new SrcRuntimeException("non-C construct not supported");
    }
    public void visit(InfExpression node) {
      throw new SrcRuntimeException("non-C construct not supported");
    }
    public void visit(BooleanLiteral node) {
      // ISO C99 sec. 7.16 says true and false are defined in stdbool.h, so
      // the parser cannot recognize them as keywords. Moreover, the
      // constructor for BooleanLiteral is never called in Cetus.
      throw new SrcRuntimeException("non-C construct not supported");
    }
    /**
     * Same as {@link #visit(EscapeLiteral)} but for character constants that
     * do not consist of escape sequences.
     */
    public void visit(CharLiteral node) {
      handleCharacterConstant(node.isWide(), node.getValue());
    }
    /**
     * Same as {@link #visit(CharLiteral)} but for character constants that
     * consist of escape sequences or that are wide character constants.
     */
    public void visit(EscapeLiteral node) {
      handleCharacterConstant(node.isWide(), node.getValue());
    }
    public void visit(FloatLiteral node) {
      // ISO C99 sec. 6.4.4.2p4 specifies how suffixes determine the type.
      SrcPrimitiveFloatingType srcType;
      final String suffix = node.getSuffix();
      if (suffix.equals(""))
        srcType = SrcDoubleType;
      else if (suffix.equalsIgnoreCase("f"))
        srcType = SrcFloatType;
      else if (suffix.equalsIgnoreCase("l"))
        srcType = SrcLongDoubleType;
      // TODO: Handle complex literals, which for gcc has a suffix like "if".
      else if (suffix.equalsIgnoreCase("if"))
        throw new UnsupportedOperationException(
          "complex literals are not yet supported");
      else
        throw new SrcRuntimeException("invalid floating suffix: " + suffix);
      // TODO: What if the value is too large for node.getValue's return or
      // LLVMConstantReal.get's parameter? Both are currently double.
      postOrderValuesAndTypes.add(new ValueAndType(
        LLVMConstantReal.get(srcType.getLLVMType(llvmContext),
                             node.getValue()),
        srcType, false));
    }
    public void visit(IntegerLiteral node) {
      // ISO C99 sec. 6.4.4.1p5 specifies how prefixes, suffixes, and values
      // determine the type.
      final String suffix = node.getSuffix();
      boolean unsigned;
      int longs;
      if (suffix.equals("")) {
        unsigned = false; longs = 0;
      }
      else if (suffix.equalsIgnoreCase("u")) {
        unsigned = true; longs = 0;
      }
      else if (suffix.equalsIgnoreCase("ul")
               || suffix.equalsIgnoreCase("lu"))
      {
        unsigned = true; longs = 1;
      }
      else if (suffix.equalsIgnoreCase("ull")
               || suffix.equalsIgnoreCase("llu"))
      {
        unsigned = true; longs = 2;
      }
      else if (suffix.equalsIgnoreCase("l")) {
        unsigned = false; longs = 1;
      }
      else if (suffix.equalsIgnoreCase("ll")) {
        unsigned = false; longs = 2;
      }
      else
        throw new SrcRuntimeException("invalid integer suffix: " + suffix);

      // TODO: What if the value is too large for Long, node.getValue's return,
      // or LLVMConstantInteger.get's parameter? All are currently long.
      // Actually, Cetus currently parses any integer literal bigger than 63
      // bits as a float. See test
      // BuildLLVMTest_PrimaryExpressions.integerLiteral for further discussion.

      // In C, integer constants are always positive, and any negative sign
      // is a separate unary minus operator. However, sometimes
      // Cetus/OpenARC gives us negative integer literals. So far, I've only
      // seen that for pragmas. For example:
      //
      //   #pragma openarc ftinject ftdata(k[-1:2])
      final boolean minus = node.getValue() < 0;
      final long absValue = minus ? -node.getValue() : node.getValue();

      // Given a positive value, the required bits to represent the value
      // are the highest-order bit that is set, plus all lower-order bits,
      // plus a sign bit if the type must be signed.
      final long bits = Long.SIZE - Long.numberOfLeadingZeros(absValue);
      final boolean dec = node.getBase() == IntegerLiteral.Base.DECIMAL;
      SrcPrimitiveIntegerType srcType = null;
      if (longs == 0 && !unsigned) {
        if (bits <= SrcIntType.getPosWidth())
          srcType = SrcIntType;
        else if (!dec && bits <= SrcUnsignedIntType.getPosWidth())
          srcType = SrcUnsignedIntType;
        else if (bits <= SrcLongType.getPosWidth())
          srcType = SrcLongType;
        else if (!dec && bits <= SrcUnsignedLongType.getPosWidth())
          srcType = SrcUnsignedLongType;
        else if (bits <= SrcLongLongType.getPosWidth())
          srcType = SrcLongLongType;
        else if (!dec && bits <= SrcUnsignedLongLongType.getPosWidth())
          srcType = SrcUnsignedLongLongType;
      }
      else if (longs == 0 && unsigned) {
        if (bits <= SrcUnsignedIntType.getPosWidth())
          srcType = SrcUnsignedIntType;
        else if (bits <= SrcUnsignedLongType.getPosWidth())
          srcType = SrcUnsignedLongType;
        else if (bits <= SrcUnsignedLongLongType.getPosWidth())
          srcType = SrcUnsignedLongLongType;
      }
      else if (longs == 1 && !unsigned) {
        if (bits <= SrcLongType.getPosWidth())
          srcType = SrcLongType;
        else if (!dec && bits <= SrcUnsignedLongType.getPosWidth())
          srcType = SrcUnsignedLongType;
        else if (bits <= SrcLongLongType.getPosWidth())
          srcType = SrcLongLongType;
        else if (!dec && bits <= SrcUnsignedLongLongType.getPosWidth())
          srcType = SrcUnsignedLongLongType;
      }
      else if (longs == 1 && unsigned) {
        if (bits <= SrcUnsignedLongType.getPosWidth())
          srcType = SrcUnsignedLongType;
        else if (bits <= SrcUnsignedLongLongType.getPosWidth())
          srcType = SrcUnsignedLongLongType;
      }
      else if (longs == 2 && !unsigned) {
        if (bits <= SrcLongLongType.getPosWidth())
          srcType = SrcLongLongType;
        else if (!dec && bits <= SrcUnsignedLongLongType.getPosWidth())
          srcType = SrcUnsignedLongLongType;
      }
      else if (longs == 2 && unsigned) {
        if (bits <= SrcUnsignedLongLongType.getPosWidth())
          srcType = SrcUnsignedLongLongType;
      }
      if (srcType == null)
        throw new UnsupportedOperationException(
          "integer constant is too large for supported types");
      ValueAndType res = new ValueAndType(
        LLVMConstantInteger.get(srcType.getLLVMType(llvmContext),
                                absValue, false),
        srcType, false);
      if (minus)
        res = res.unaryMinus(llvmModule, llvmBuilder);
      postOrderValuesAndTypes.add(res);
    }
    public void visit(StringLiteral node) {
      // TODO: Are we handling multibyte characters properly? See ISO C99
      // 6.4.5p5's mention of mbstowcs.
      final long[] valueArray = node.getValueArray();
      final SrcPrimitiveIntegerType elementType
        = node.isWide() ? SRC_WCHAR_TYPE : SrcCharType;
      final SrcType type = SrcArrayType.get(elementType, valueArray.length);
      final LLVMConstant[] elements = new LLVMConstant[valueArray.length];
      for (int i = 0; i < valueArray.length; ++i)
        elements[i]
          = LLVMConstantInteger.get(elementType.getLLVMType(llvmContext),
                                    valueArray[i],
                                    elementType.isSigned());
      postOrderValuesAndTypes.add(new ValueAndType(
        LLVMConstantArray.get(elementType.getLLVMType(llvmContext),
                              elements),
        type, false));
    }
    public void visit(MinMaxExpression node) {
      throw new SrcRuntimeException("non-C construct not supported");
    }
    public void visit(NewExpression node) {
      throw new SrcRuntimeException("non-C construct not supported");
    }
    public void visit(NVLGetRootExpression node) {
      final SrcFunctionBuiltin builtin
        = srcSymbolTable.getFunctionBuiltin("__builtin_nvl_get_root");
      if (llvmFunction == null)
        throw new SrcRuntimeException("call to "+builtin.getName()
                                      +" at file scope");
      final ValueAndType op1 = postOrderValuesAndTypes.remove(
                                 postOrderValuesAndTypes.size()-1);
      final SrcType op2 = computeSrcTypeFromList(node,
                                                 node.getSpecifiers());
      ValueAndType call = builtin.call(
        node, new ValueAndType[]{op1}, op2, srcSymbolTable, llvmModule,
        llvmModuleIndex, llvmTargetData, llvmBuilder, warningsAsErrors);
      postOrderValuesAndTypes.add(call);
    }
    public void visit(NVLAllocNVExpression node) {
      final SrcFunctionBuiltin builtin
        = srcSymbolTable.getFunctionBuiltin("__builtin_nvl_alloc_nv");
      if (llvmFunction == null)
        throw new SrcRuntimeException("call to "+builtin.getName()
                                      +" at file scope");
      final int argStart = postOrderValuesAndTypes.size() - 2;
      final ValueAndType op1 = postOrderValuesAndTypes.get(argStart);
      final ValueAndType op2 = postOrderValuesAndTypes.get(argStart + 1);
      postOrderValuesAndTypes.setSize(argStart);
      final SrcType op3 = computeSrcTypeFromList(node,
                                                 node.getSpecifiers());
      // We cannot compute the size of a function type or incomplete type,
      // so we cannot evaluate the call to the builtin. In the case of an
      // incomplete struct type, completing the struct type by the end of
      // the translation unit is not good enough for nvl_alloc_nv even
      // though it's good enough for some other NVL-C use cases. The
      // difference is that, as for sizeof, we must have the size in order
      // to generate the nvl_alloc_nv code here.
      if (op3.iso(SrcFunctionType.class))
        throw new SrcRuntimeException(
          "argument 3 to "+builtin.getName()+" is of function type");
      if (op3.isIncompleteType())
        throw new SrcRuntimeException(
          "argument 3 to "+builtin.getName()+" is of incomplete type");
      final ValueAndType op3Size = new ValueAndType(
        LLVMConstantInteger.get(
          SRC_SIZE_T_TYPE.getLLVMType(llvmContext),
          llvmTargetData.abiSizeOfType(op3.getLLVMType(llvmContext)),
          true),
        SRC_SIZE_T_TYPE, false);
      ValueAndType call = builtin.call(
        node, new ValueAndType[]{op1, op2, op3Size}, op3, srcSymbolTable,
        llvmModule, llvmModuleIndex, llvmTargetData, llvmBuilder,
        warningsAsErrors);
      postOrderValuesAndTypes.add(call);
    }
    public void visit(OffsetofExpression node) {
      // ISO C99 sec. 7.17p3.

      // Get type operand.
      final SrcType type = computeSrcTypeFromList(node, node.getSpecifiers());
      final SrcStructOrUnionType structOrUnion
        = type.toIso(SrcStructOrUnionType.class);
      if (structOrUnion == null)
        throw new SrcRuntimeException(
          "first operand to offsetof is not a struct or union type");

      // Get member designator operand.
      if (node.getExpression() instanceof AccessExpression)
        // clang (3.5.1) with command-line options "-std=c99 -pedantic"
        // reports this as an extension. For example: offsetof(struct S, f.f)
        throw new UnsupportedOperationException(
          "member of member in offsetof is not yet supported");
      assert(node.getExpression() instanceof Identifier);
      final String member = ((Identifier)node.getExpression()).getName();

      // Compute the offset.
      postOrderValuesAndTypes.add(structOrUnion.offsetof(member, llvmModule));
    }
    public void visit(RangeExpression node) {
      throw new SrcRuntimeException("non-C construct not supported");
    }
    /** Pruned traversal so can perform actions pre and post. */
    public void visit(SizeofExpression node) {
      final SrcType srcType;
      if (node.getExpression() != null) {
        // Get type from expression operand.
        final FakeFunctionForEval fakeFunctionForEval
          = new FakeFunctionForEval("sizeof");
        visitTree(node.getExpression());
        srcType = postOrderValuesAndTypes
                  .remove(postOrderValuesAndTypes.size()-1).getSrcType();
        fakeFunctionForEval.restore();
      }
      else
        srcType = computeSrcTypeFromList(node, node.getTypes());
      if (srcType.iso(SrcFunctionType.class))
        throw new SrcRuntimeException(
          "sizeof operand is of function type");
      if (srcType.isIncompleteType())
        throw new SrcRuntimeException(
          "sizeof operand is of incomplete type");
      if (srcType.iso(SrcBitFieldType.class))
        throw new SrcRuntimeException(
          "sizeof operand is a bit-field");
      // TODO: Handle sizeof for variable length.
      postOrderValuesAndTypes.add(new ValueAndType(
        LLVMConstantInteger.get(
          SRC_SIZE_T_TYPE.getLLVMType(llvmContext),
          llvmTargetData.abiSizeOfType(srcType.getLLVMType(llvmContext)),
          true),
        SRC_SIZE_T_TYPE, false));
    }
    private static final String ASM = "__asm__";
    public void visit(SomeExpression node) {
      // For now, our goal with inline assembly support is to handle just
      // enough to get SPEC CPU 2006 to pass, which is only the place we test
      // it. We'll add more support as the need arises.

      // Remove asm keyword.
      String expr = node.toString().trim();
      if (!expr.startsWith(ASM))
        throw new SrcRuntimeException("non-C construct not supported: "
                                      + node.toString());
      expr = expr.substring(ASM.length()).trim();

      // Remove outer parentheses.
      if (!expr.startsWith("(") || !expr.endsWith(")"))
        throw new SrcRuntimeException(
          "missing parentheses around asm expression: " + node.toString());
      expr = expr.substring(1, expr.length()-1).trim();

      // Retrieve assembly string, and replace "%" with "$".
      // TODO: Is that always the right thing to do? Can "%" be part of an
      // escape sequence that we should skip?
      // TODO: Handle any escape sequences (especially quotes).
      final Pattern quotesPattern
        = Pattern.compile("\\A\"(([^\"\\\\]|\\\\.)*)\"");
      final Matcher assemblyMatcher = quotesPattern.matcher(expr);
      if (!assemblyMatcher.find())
        throw new SrcRuntimeException(
          "could not find assembly string in asm expression: "
          + node.toString());
      final String assemblyIn = assemblyMatcher.group(1);
      expr = expr.substring(assemblyMatcher.end()).trim();
      final String assemblyOut = assemblyIn.replace('%', '$');

      // Remove colon.
      if (!expr.startsWith(":"))
        throw new SrcRuntimeException(
          "missing colon after assembly string in asm expression: "
          + node.toString());
      expr = expr.substring(1).trim();

      // Retrieve operand constraint string, and replace "+r" with "=r,0".
      // TODO: Handle other possibilities, and unescape any escape sequences
      // (especially quotes).
      final Matcher constraintMatcher = quotesPattern.matcher(expr);
      if (!constraintMatcher.find())
        throw new SrcRuntimeException(
          "could not find operand constraint string in asm expression: "
          + node.toString());
      final String constraintIn = constraintMatcher.group(1);
      expr = expr.substring(constraintMatcher.end()).trim();
      if (!constraintIn.equals("+r"))
        throw new UnsupportedOperationException(
          "asm expression's operand constraint string format is not yet"
          + " supported: " + node.toString());
      final String constraintOut = "=r,0";

      // Remove operand expression.
      // TODO: Are there other possible forms?
      final Pattern operandPattern = Pattern.compile("\\A\\([^()]*\\).*");
      final Matcher operandMatcher = operandPattern.matcher(expr);
      if (!operandMatcher.find())
        throw new SrcRuntimeException(
          "could not find operand expression in asm expression: "
          + node.toString());
      expr = expr.substring(operandMatcher.end()).trim();

      // Make sure there's nothing else.
      // TODO: Support other possibilities, such as additional operands.
      if (!expr.isEmpty())
        throw new UnsupportedOperationException(
          "asm expression format is not yet supported: "
          + node.toString());

      // Get the operand expression result. There should be only one based on
      // the above parsing, and it must be an lvalue because of the "+r".
      assert(node.getChildren().size() == 1);
      final ValueAndType operand
        = postOrderValuesAndTypes.remove(postOrderValuesAndTypes.size() - 1);
      if (!operand.isLvalue())
        throw new SrcRuntimeException(
          "asm expression operand is not an lvalue: " + node.toString());

      // Generate LLVM asm call.
      final LLVMType operandType
        = operand.getSrcType().getLLVMType(llvmContext);
      final LLVMFunctionType fnType
        = LLVMFunctionType.get(operandType, false, operandType);
      final LLVMConstantInlineASM asmLLVM
        = LLVMConstantInlineASM.get(fnType, assemblyOut, constraintOut, false,
                                    false);
      final LLVMCallInstruction call = new LLVMCallInstruction(
        llvmBuilder, ".asm", asmLLVM,
        operand.prepareForOp(llvmModule, llvmBuilder).getLLVMValue());
      operand.store(false, call, llvmModule, llvmBuilder);
      postOrderValuesAndTypes.add(new ValueAndType());
    }
    /** Pre-order traversal so we can validate first. */
    public void visit(StatementExpression node) {
      if (llvmFunction == null)
        throw new SrcRuntimeException("statement expression at file scope");
      // The child CompoundStatement ensures that a value is left as the
      // value of this StatementExpression.
    }
    public void visit(ThrowExpression node) {
      throw new SrcRuntimeException("non-C construct not supported");
    }
    public void visit(Typecast node) {
      final SrcType toType
        = computeSrcTypeFromList(node, node.getSpecifiers());
      postOrderValuesAndTypes.add(
        postOrderValuesAndTypes.remove(postOrderValuesAndTypes.size()-1)
        .explicitCast(toType, llvmModule, llvmBuilder));
    }
    public void visit(UnaryExpression node) {
      final ValueAndType operand
        = postOrderValuesAndTypes.remove(postOrderValuesAndTypes.size()-1);
      final ValueAndType result;
      if (node.getOperator() == UnaryOperator.ADDRESS_OF)
        result = operand.address(llvmModule, llvmBuilder);
      else if (node.getOperator() == UnaryOperator.DEREFERENCE)
        result = operand.indirection("operand to unary \"*\"", llvmModule,
                                     llvmBuilder);
      else if (node.getOperator() == UnaryOperator.PLUS)
        result = operand.unaryPlus(llvmModule, llvmBuilder);
      else if (node.getOperator() == UnaryOperator.MINUS)
        result = operand.unaryMinus(llvmModule, llvmBuilder);
      else if (node.getOperator() == UnaryOperator.BITWISE_COMPLEMENT)
        result = operand.unaryBitwiseComplement(llvmModule, llvmBuilder);
      else if (node.getOperator() == UnaryOperator.LOGICAL_NEGATION)
        result = operand.unaryNot(llvmModule, llvmBuilder);
      else if (node.getOperator() == UnaryOperator.POST_INCREMENT
               || node.getOperator() == UnaryOperator.POST_DECREMENT
               || node.getOperator() == UnaryOperator.PRE_INCREMENT
               || node.getOperator() == UnaryOperator.PRE_DECREMENT)
      {
        // We choose the smallest integer type for the 1 constant so we
        // don't accidentally promote the operand to a different type.
        final SrcIntegerType oneType = SrcBoolType; // smallest integer type
        final LLVMConstantInteger oneVal
          = LLVMConstantInteger.get(oneType.getLLVMType(llvmContext), 1,
                                    false);
        final ValueAndType one = new ValueAndType(oneVal, oneType, false);
        final ValueAndType old = operand.prepareForOp(llvmModule,
                                                      llvmBuilder);
        final ValueAndType new_;
        final String operator;
        if (node.getOperator() == UnaryOperator.POST_INCREMENT
            || node.getOperator() == UnaryOperator.PRE_INCREMENT)
        {
          operator = "\"++\"";
          new_ = ValueAndType.add(operator, old, one, llvmModule,
                                  llvmBuilder);
        }
        else {
          operator = "\"--\"";
          new_ = ValueAndType.subtract(operator, old, one, llvmTargetData,
                                       llvmModule, llvmBuilder);
        }
        ValueAndType.simpleAssign(operator, false, operand, new_, llvmModule,
                                  llvmBuilder, warningsAsErrors);
        if (node.getOperator() == UnaryOperator.POST_INCREMENT
            || node.getOperator() == UnaryOperator.POST_DECREMENT)
          result = old;
        else
          result = new_;
      }
      else
        throw new IllegalStateException();
      postOrderValuesAndTypes.add(result);
    }
    public void visit(VaArgExpression node) {
      final SrcFunctionBuiltin builtin
        = srcSymbolTable.getFunctionBuiltin("__builtin_va_arg");
      if (llvmFunction == null)
        throw new SrcRuntimeException("call to "+builtin.getName()
                                      +" at file scope");
      final ValueAndType op1 = postOrderValuesAndTypes.remove(
                                 postOrderValuesAndTypes.size()-1);
      final SrcType op2 = computeSrcTypeFromList(node,
                                                 node.getSpecifiers());
      ValueAndType call = builtin.call(
        node, new ValueAndType[]{op1}, op2, srcSymbolTable, llvmModule,
        llvmModuleIndex, llvmTargetData, llvmBuilder, warningsAsErrors);
      postOrderValuesAndTypes.add(call);
    }
    /** Pruned traversal so can perform actions pre and post. */
    public void visit(Initializer node) {
      // TODO: Handle the case that some initializers for child elements might
      // not be braced and thus might not be in child Initializer nodes.

      // TODO: Translate designated initializers. These will be very difficult
      // because of unions and variables of static storage duration (globals
      // and static locals). That is, currently a SrcUnionType and its
      // LLVMStructType both have the same first member so that it's easy to
      // generate a constant expression that initializes the first member of a
      // static-storage variable of that type. However, with designated
      // initializers, it will be possible to initialize a different member,
      // but you cannot bitcast aggregates in LLVM, so different
      // static-storage variables of the same SrcUnionType will have to be
      // generated with different declared LLVM types that accommodate their
      // initializers. Moreover, any aggregate types (struct, union, array)
      // that have members that are unions, recursively, will then have this
      // problem. Also, copying such aggregates requires llvm.memcpy. If all
      // that isn't bad enough, now imagine the LLVM type needed to intialize
      // an array of unions, where each element of the array has a different
      // union member initialized. clang (clang-600.0.56) output for these
      // cases shows some nasty solutions. See related todo in SrcUnionType's
      // setBody.

      final InitDestination initDestination = initDestinationStack.peek();
      final SrcStructOrUnionType initStructOrUnionType
        = initDestination.srcType.toIso(SrcStructOrUnionType.class);
      SrcArrayType initArrayType
        = initDestination.srcType.toIso(SrcArrayType.class);
      if (initDestination.srcType.isIncompleteType() && initArrayType == null)
        throw new SrcRuntimeException("initialization of incomplete type");

      // Compute the max number of initializer elements (that is, child
      // nodes). In the case of a string literal initializing an array, this
      // number does not constrain the number of characters in the string
      // literal.
      final long maxNumChildren;
      if (initDestination.srcType.iso(SrcStructType.class))
        maxNumChildren = initStructOrUnionType.getMemberTypes().length;
      else if (initArrayType != null)
        maxNumChildren = initArrayType.numElementsIsSpecified()
                         ? initArrayType.getNumElements()
                         : node.getChildren().size();
      else {
        // Union or scalar.
        maxNumChildren = 1;
      }
      if (node.getChildren().size() > maxNumChildren)
        throw new SrcRuntimeException("too many elements in initializer");

      // If there's only one child and it's not an Initializer node (and so we
      // don't need to push to initDestinationStack before visiting it), then
      // visit it now, and determine whether it initializes the destination
      // type or just an element of the destination type.
      final ValueAndType firstChildValueAndType;
      final boolean oneChildInitsDestinationType;
      if (node.getChildren().size() == 1
          && !(node.getChildren().get(0) instanceof Initializer))
      {
        visitTree(node.getChildren().get(0));
        firstChildValueAndType = postOrderValuesAndTypes.remove(
                                   postOrderValuesAndTypes.size()-1);
        if (initStructOrUnionType != null)
          // The child's type must be exactly the destination type or else it
          // might be initializing a destination element, which is perhaps a
          // struct/union.
          oneChildInitsDestinationType
            = firstChildValueAndType.getSrcType().iso(initDestination.srcType);
        else if (initArrayType != null)
          // The only way an array can initialize an array is when the
          // initializer is a string literal (which ValueAndType's
          // constructor's preconditions say means the LLVM value is then an
          // LLVMConstantArray). In that case, the destination element type
          // must be an integer type, or else the initializer might be for a
          // single destination element, which might be of type array or
          // pointer.
          oneChildInitsDestinationType
            = firstChildValueAndType.isStringLiteral()
              && initArrayType.getElementType().iso(SrcIntegerType.class);
        else
          oneChildInitsDestinationType = true;
      }
      else {
        firstChildValueAndType = null;
        oneChildInitsDestinationType = false;
      }

      // If array size is unspecified, update it.
      if (initArrayType != null && !initArrayType.numElementsIsSpecified()) {
        initDestination.updateArrayType(
          oneChildInitsDestinationType
          ? firstChildValueAndType.getSrcType().toIso(SrcArrayType.class)
            .getNumElements()
          : node.getChildren().size());
        initArrayType = initDestination.srcType.toIso(SrcArrayType.class);
      }

      // Generate a stack allocation if required.
      if (initDestination.stackAllocName != null)
        postOrderValuesAndTypes.add(
          initDestination.generateStackAlloc(llvmContext,
                                             getLLVMAllocaBuilder()));

      // Evaluate child initializers, and either build a list of constant
      // expressions (if constant expressions are required) or generate store
      // instructions.
      ArrayList<LLVMConstant> childConstants = new ArrayList<>();
      FlatIterator<Traversable> itr = new FlatIterator<>(node);
      int childIndex;
      for (childIndex = 0; itr.hasNext(); ++childIndex) {
        final Traversable child = itr.next();

        // Compute this child's lvalue and type.
        final InitDestination childInitDestination;
        if (oneChildInitsDestinationType)
          childInitDestination = initDestination;
        else if (initStructOrUnionType != null) {
          if (initDestination.stackAddr == null)
            childInitDestination
              = new InitDestination(initStructOrUnionType
                                    .getMemberTypes()[childIndex]);
          else
            childInitDestination = new InitDestination(
              initStructOrUnionType.accessMember(
                EnumSet.noneOf(SrcTypeQualifier.class),
                initDestination.stackAddr, childIndex, true,
                llvmModule, llvmBuilder, null));
        }
        else {
          assert(initArrayType != null);
          final LLVMIntegerType i32 = LLVMIntegerType.get(llvmContext, 32);
          if (initDestination.stackAddr == null)
            childInitDestination
              = new InitDestination(initArrayType.getElementType());
          else
            childInitDestination
              = new InitDestination(
                  initArrayType.getElementType(),
                  LLVMGetElementPointerInstruction.create(
                    llvmBuilder, ".arrayElement", initDestination.stackAddr,
                    LLVMConstant.constNull(i32),
                    LLVMConstantInteger.get(i32, childIndex, false)));
        }

        // Evaluate child initializer if we haven't yet. A value will be
        // retrieved only if we're building a constant expression or if the
        // child is not a compound initializer.
        final ValueAndType childValueAndType;
        if (childIndex == 0 && firstChildValueAndType != null)
          childValueAndType = firstChildValueAndType;
        else {
          initDestinationStack.push(childInitDestination);
          visitTree(child);
          initDestinationStack.pop();
          if (initDestination.stackAddr == null
              || !(child instanceof Initializer))
            childValueAndType = postOrderValuesAndTypes.remove(
                                 postOrderValuesAndTypes.size()-1);
          else
            childValueAndType = null;
        }

        // Convert any child initializer value to the correct type unless it's
        // a constant-expression compound initializer, which already has the
        // correct type.
        final LLVMValue childValueConverted;
        if (childValueAndType == null)
          childValueConverted = null;
        else if (child instanceof Initializer)
          childValueConverted = childValueAndType.getLLVMValue();
        else {
          // This would also be caught in the check for constant expressions
          // below, but we need to avoid generating a load instruction
          // while at file scope, which introduces a race that sometimes causes
          // LLVM assertions to fail ("Use still stuck around after Def is
          // destroyed") when we start cleaning up jllvm/LLVM memory.
          // TODO: This check could be eliminated if we were to use
          // FakeFunctionForEval when generating initializers at file scope.
          if (initDestination.stackAddr == null
              && childValueAndType.isLvalue()
              && !childValueAndType.getSrcType().iso(SrcArrayType.class))
            throw new SrcRuntimeException("initializer is not a constant");
          childValueConverted
            = childValueAndType.convertForAssign(childInitDestination.srcType,
                                                 AssignKind.INITIALIZATION,
                                                 llvmModule, llvmBuilder,
                                                 warningsAsErrors);
        }

        // Record constant expression, or generate child store instruction
        // unless the child is an Initializer node (in that case, store
        // instructions have already been generated).
        if (initDestination.stackAddr == null) {
          if (!(childValueConverted instanceof LLVMConstant))
            throw new SrcRuntimeException("initializer is not a constant");
          childConstants.add((LLVMConstant)childValueConverted);
        }
        else if (childValueConverted != null)
          childInitDestination.srcType.store(
            true, childInitDestination.stackAddr, childValueConverted,
            llvmModule, llvmBuilder);
      }

      // Check for uninitialized pointers to NVM.
      if (oneChildInitsDestinationType)
        ;
      else if (initStructOrUnionType != null) {
        final SrcType[] memberTypes = initStructOrUnionType.getMemberTypes();
        for (; childIndex < memberTypes.length; ++childIndex)
          memberTypes[childIndex]
          .checkAllocWithoutExplicitInit(initDestination.stackAddr == null);
      }
      else {
        assert(initArrayType != null);
        if (childIndex < maxNumChildren)
          initArrayType.getElementType()
          .checkAllocWithoutExplicitInit(initDestination.stackAddr == null);
      }

      // If generating store instructions, we're done.
      if (initDestination.stackAddr != null)
        return;

      // Combine child constant expressions.
      if (oneChildInitsDestinationType)
        postOrderValuesAndTypes.add(new ValueAndType(
          childConstants.get(0), initDestination.srcType, false));
      else if (initStructOrUnionType != null) {
        // llvmBuilder here should not end up being used because we have
        // constants
        postOrderValuesAndTypes.add(
          initStructOrUnionType.buildConstant(childConstants, llvmModule,
                                              llvmBuilder));
      }
      else {
        assert(initArrayType != null);
        final SrcType initElementType = initArrayType.getElementType();
        for (int i = childConstants.size();
             i < initArrayType.getNumElements();
             ++i)
          childConstants.add(
            LLVMConstant.constNull(initElementType.getLLVMType(llvmContext)));
        final LLVMConstant[] childConstantsArray
          = new LLVMConstant[childConstants.size()];
        childConstants.toArray(childConstantsArray);
        postOrderValuesAndTypes.add(new ValueAndType(
          LLVMConstantArray.get(initElementType.getLLVMType(llvmContext),
                                childConstantsArray),
          initArrayType, false));
      }
    }
    /** {@link Initializer} docs say this is not for C. */
    public void visit(ConstructorInitializer node) {
      throw new SrcRuntimeException("non-C construct not supported");
    }
    /** {@link Initializer} docs say this is not for C. */
    public void visit(ListInitializer node) {
      throw new SrcRuntimeException("non-C construct not supported");
    }
    /** {@link Initializer} docs say this is not for C. */
    public void visit(ValueInitializer node) {
      throw new SrcRuntimeException("non-C construct not supported");
    }
    /** Pruned traversal so can perform actions pre and post. */
    public void visit(Program node) {
      llvmModules = new LLVMModule[node.getChildren().size()];
      llvmModuleIdentifiers = new String[llvmModules.length];
      srcSymbolTable = new SrcSymbolTable(llvmContext, llvmModules.length,
                                          debugTypeChecksums);
      jumpScopeTable = new JumpScopeTable(srcSymbolTable);
      basicBlockSetStack = new BasicBlockSetStack(llvmContext);

      {
        final FlatIterator<Traversable> itr = new FlatIterator<>(node);
        while (itr.hasNext())
          visitTree(itr.next());
      }

      // Off by default. See LLVMModule#verify documentation for
      // justification.
      if (verifyModule) {
        for (int i = 0; i < llvmModules.length; ++i) {
          // Some failures (such as missing basic block terminators) cause an
          // abort no matter what action is specified here. Others (such as a
          // return operand type that does not match the function return type)
          // obey the action specified here.
          final String verifyError
            = llvmModules[i].verify(LLVMVerifierFailureAction
                                    .LLVMReturnStatusAction);
          if (verifyError != null)
            throw new IllegalStateException(
              "LLVM module verification failed for \""
              + llvmModuleIdentifiers[i] + "\":\n" + verifyError);
        }
      }
    }
    public void visit(AnnotationStatement node) {
      startStatementPragmas(node);
      translateStandalonePragmas(node);
      endStatementPragmas(node);
    }
    public void visit(BreakStatement node) {
      startStatementPragmas(node);
      assert(node.getChildren().isEmpty());
      if (breakTargetStack.isEmpty())
        throw new SrcRuntimeException(
          "break statement not inside a loop or switch body");
      new LLVMBranchInstruction(llvmBuilder, breakTargetStack.peek());
      // Any instructions placed in the following block are unreachable, but
      // the source code might contain unreachable code, or we will blindly
      // generate a terminator instruction here to start/finish a construct.
      llvmBuilder.positionBuilderAtEnd(
        basicBlockSetStack.registerBasicBlock(".break.dead", llvmFunction));
      endStatementPragmas(node);
    }
    /**
     * Pruned traversal because child expression is traversed at parent
     * {@link SwitchStatement}.
     */
    public void visit(Case node) {
      checkLabeledStatementPragmas(node);
      jumpScopeTable.setSwitchLabelScope(node);
      // SwitchStatement's visitor method populates caseMap before traversing
      // its body.
      final LLVMBasicBlock bb = srcSymbolTable.getCaseBasicBlock(node);
      basicBlockSetStack.registerBasicBlock(bb);
      new LLVMBranchInstruction(llvmBuilder, bb);
      llvmBuilder.positionBuilderAtEnd(bb);
    }
    /** Pruned traversal so can perform actions pre and post. */
    public void visit(CompoundStatement node) {
      startStatementPragmas(node);
      jumpScopeTable.startScope(node);
      {
        FlatIterator<Traversable> itr = new FlatIterator<>(node);
        while (itr.hasNext())
          visitTree(itr.next());
      }
      // If this is a statement expression and its last child is an expression
      // statement, that child will leave a value in postOrderValuesAndTypes.
      // Otherwise, we must leave a void expression.
      if (node.getParent() instanceof StatementExpression) {
        final List<Traversable> children = node.getChildren();
        if (children.isEmpty()
            || !(children.get(children.size()-1)
                 instanceof ExpressionStatement))
          postOrderValuesAndTypes.add(new ValueAndType());
      }
      jumpScopeTable.endScope(node);
      endStatementPragmas(node);
    }
    public void visit(ExceptionHandler node) {
      throw new SrcRuntimeException("non-C construct not supported");
    }
    public void visit(ContinueStatement node) {
      startStatementPragmas(node);
      assert(node.getChildren().isEmpty());
      if (continueTargetStack.isEmpty())
        throw new SrcRuntimeException(
          "continue statement not inside a loop body");
      new LLVMBranchInstruction(llvmBuilder, continueTargetStack.peek());
      // Any instructions placed in the following block are unreachable, but
      // the source code might contain unreachable code, or we will blindly
      // generate a terminator instruction here to start/finish a construct.
      llvmBuilder.positionBuilderAtEnd(
        basicBlockSetStack.registerBasicBlock(".cont.dead", llvmFunction));
      endStatementPragmas(node);
    }
    /** Pruned traversal so can perform actions pre and post. */
    public void visit(DeclarationStatement node) {
      startStatementPragmas(node);
      final FlatIterator<Traversable> itr = new FlatIterator<>(node);
      while (itr.hasNext())
        visitTree(itr.next());
      endStatementPragmas(node);
    }
    public void visit(Default node) {
      checkLabeledStatementPragmas(node);
      jumpScopeTable.setSwitchLabelScope(node);
      // SwitchStatement's visitor method populates defaultMap before
      // traversing its body.
      final LLVMBasicBlock bb = srcSymbolTable.getDefaultBasicBlock(node);
      basicBlockSetStack.registerBasicBlock(bb);
      new LLVMBranchInstruction(llvmBuilder, bb);
      llvmBuilder.positionBuilderAtEnd(bb);
    }
    /** Pruned traversal so can perform actions pre and post. */
    public void visit(DoLoop node) {
      startStatementPragmas(node);
      jumpScopeTable.startScope(node);
      final LLVMBasicBlock prevBB = llvmBuilder.getInsertBlock();

      // Record continue and break targets before evaluating body.
      final LLVMBasicBlock condFirstBB
        = basicBlockSetStack.registerBasicBlock(".do.condFirstBB",
                                                  llvmFunction);
      final LLVMBasicBlock nextBB
        = basicBlockSetStack.registerBasicBlock(".do.nextBB", llvmFunction);
      continueTargetStack.push(condFirstBB);
      breakTargetStack.push(nextBB);

      // Evaluate body.
      final LLVMBasicBlock bodyFirstBB
        = basicBlockSetStack.registerBasicBlock(".do.bodyFirstBB",
                                                  llvmFunction);
      llvmBuilder.positionBuilderAtEnd(bodyFirstBB);
      visitTree(node.getBody());
      final LLVMBasicBlock bodyLastBB = llvmBuilder.getInsertBlock();

      // Pop continue and break targets.
      continueTargetStack.pop();
      breakTargetStack.pop();

      // Evaluate controlling expression as an i1 value.
      llvmBuilder.positionBuilderAtEnd(condFirstBB);
      visitTree(node.getCondition());
      final LLVMValue cond
        = postOrderValuesAndTypes.remove(postOrderValuesAndTypes.size()-1)
          .evalAsCond("controlling expression of do statement",
                      llvmModule, llvmBuilder);
      final LLVMBasicBlock condLastBB = llvmBuilder.getInsertBlock();

      // Add control flow.
      llvmBuilder.positionBuilderAtEnd(prevBB);
      new LLVMBranchInstruction(llvmBuilder, bodyFirstBB);
      llvmBuilder.positionBuilderAtEnd(bodyLastBB);
      new LLVMBranchInstruction(llvmBuilder, condFirstBB);
      llvmBuilder.positionBuilderAtEnd(condLastBB);
      new LLVMBranchInstruction(llvmBuilder, cond, bodyFirstBB, nextBB);
      llvmBuilder.positionBuilderAtEnd(nextBB);

      jumpScopeTable.endScope(node);
      endStatementPragmas(node);
    }
    /** Pruned traversal so can perform actions pre and post. */
    public void visit(ExpressionStatement node) {
      startStatementPragmas(node);
      {
        final FlatIterator<Traversable> itr = new FlatIterator<>(node);
        while (itr.hasNext())
          visitTree(itr.next());
      }

      // If this is the last statement in a statement expression, then leave
      // its value as the value of the statement expression.
      final Traversable parent = node.getParent();
      if (parent instanceof CompoundStatement
          && parent.getParent() instanceof StatementExpression)
      {
        if (node == parent.getChildren().get(parent.getChildren().size()-1)) {
          endStatementPragmas(node);
          return;
        }
      }

      // ISO C99 sec. 6.8.3p2 says this is evaluated like a void expresison.
      // See SrcPrimitiveNonNumericType.convertFromNoPrep comments for how cast
      // to void is handled. In both cases, we call prepareForOp, which might
      // perform a load instruction. At least clang (clang-600.0.56) also does
      // so.
      postOrderValuesAndTypes.remove(postOrderValuesAndTypes.size()-1)
                             .prepareForOp(llvmModule, llvmBuilder);
      endStatementPragmas(node);
    }
    /** Pruned traversal so can perform actions pre and post. */
    public void visit(ForLoop node) {
      startStatementPragmas(node);
      jumpScopeTable.startScope(node);

      // Evaluate initial clause.
      visitTree(node.getInitialStatement());
      final LLVMBasicBlock initLastBB = llvmBuilder.getInsertBlock();

      // Evaluate controlling expression as an i1 value.
      final LLVMBasicBlock condFirstBB
        = basicBlockSetStack.registerBasicBlock(".for.condFirstBB",
                                                  llvmFunction);
      llvmBuilder.positionBuilderAtEnd(condFirstBB);
      final LLVMValue cond;
      if (node.getCondition() != null) {
        visitTree(node.getCondition());
        cond = postOrderValuesAndTypes.remove(postOrderValuesAndTypes.size()-1)
               .evalAsCond("controlling expression of for statement",
                           llvmModule, llvmBuilder);
      }
      else
        cond = LLVMConstantInteger.get(LLVMIntegerType.get(llvmContext, 1), 1,
                                       false);
      final LLVMBasicBlock condLastBB = llvmBuilder.getInsertBlock();

      // Record continue and break targets before evaluating body.
      final LLVMBasicBlock stepFirstBB
        = basicBlockSetStack.registerBasicBlock(".for.stepFirstBB",
                                                  llvmFunction);
      final LLVMBasicBlock nextBB
        = basicBlockSetStack.registerBasicBlock(".for.nextBB", llvmFunction);
      continueTargetStack.push(stepFirstBB);
      breakTargetStack.push(nextBB);

      // Evaluate body.
      final LLVMBasicBlock bodyFirstBB
        = basicBlockSetStack.registerBasicBlock(".for.bodyFirstBB",
                                                  llvmFunction);
      llvmBuilder.positionBuilderAtEnd(bodyFirstBB);
      visitTree(node.getBody());
      final LLVMBasicBlock bodyLastBB = llvmBuilder.getInsertBlock();

      // Pop continue and break targets.
      continueTargetStack.pop();
      breakTargetStack.pop();

      // Evaluate step clause.
      llvmBuilder.positionBuilderAtEnd(stepFirstBB);
      if (node.getStep() != null) {
        visitTree(node.getStep());
        postOrderValuesAndTypes.remove(postOrderValuesAndTypes.size()-1);
      }
      final LLVMBasicBlock stepLastBB = llvmBuilder.getInsertBlock();

      // Add control flow.
      llvmBuilder.positionBuilderAtEnd(initLastBB);
      new LLVMBranchInstruction(llvmBuilder, condFirstBB);
      llvmBuilder.positionBuilderAtEnd(condLastBB);
      new LLVMBranchInstruction(llvmBuilder, cond, bodyFirstBB, nextBB);
      llvmBuilder.positionBuilderAtEnd(bodyLastBB);
      new LLVMBranchInstruction(llvmBuilder, stepFirstBB);
      llvmBuilder.positionBuilderAtEnd(stepLastBB);
      new LLVMBranchInstruction(llvmBuilder, condFirstBB);
      llvmBuilder.positionBuilderAtEnd(nextBB);

      jumpScopeTable.endScope(node);
      endStatementPragmas(node);
    }
    public void visit(GotoStatement node) {
      startStatementPragmas(node);
      jumpScopeTable.setGotoScope(node);
      if (node.getTarget() == null)
        throw new SrcRuntimeException(
          "goto statement's target label is undefined: " + node.toString());
      new LLVMBranchInstruction(
        llvmBuilder,
        srcSymbolTable.getLabelBasicBlock(node.getTarget()));
      // Any instructions placed in the following block are unreachable, but
      // the source code might contain unreachable code, or we will blindly
      // generate a terminator instruction here to start/finish a construct.
      llvmBuilder.positionBuilderAtEnd(
        basicBlockSetStack.registerBasicBlock(".goto.dead", llvmFunction));
      endStatementPragmas(node);
    }
    /** Pruned traversal so can perform actions pre and post. */
    public void visit(IfStatement node) {
      startStatementPragmas(node);

      // Evaluate controlling expression as an i1 value.
      visitTree(node.getControlExpression());
      final LLVMValue cond
        = postOrderValuesAndTypes.remove(postOrderValuesAndTypes.size()-1)
          .evalAsCond("controlling expression of if statement",
                      llvmModule, llvmBuilder);
      final LLVMBasicBlock condLastBB = llvmBuilder.getInsertBlock();

      // Evaluate "then" statement.
      final LLVMBasicBlock thenFirstBB
        = basicBlockSetStack.registerBasicBlock(".if.thenFirstBB",
                                                  llvmFunction);
      llvmBuilder.positionBuilderAtEnd(thenFirstBB);
      visitTree(node.getThenStatement());
      final LLVMBasicBlock thenLastBB = llvmBuilder.getInsertBlock();

      // Evaluate "else" statement if it exists.
      final LLVMBasicBlock elseFirstBB, elseLastBB;
      if (node.getElseStatement() != null) {
        elseFirstBB = basicBlockSetStack.registerBasicBlock(
          ".if.elseFirstBB", llvmFunction);
        llvmBuilder.positionBuilderAtEnd(elseFirstBB);
        visitTree(node.getElseStatement());
        elseLastBB = llvmBuilder.getInsertBlock();
      }
      else
        elseFirstBB = elseLastBB = null;

      // Start next basic block and add control flow.
      final LLVMBasicBlock nextBB = basicBlockSetStack.registerBasicBlock(
        ".if.nextBB", llvmFunction);
      llvmBuilder.positionBuilderAtEnd(condLastBB);
      new LLVMBranchInstruction(llvmBuilder, cond, thenFirstBB,
                                elseFirstBB != null ? elseFirstBB : nextBB);
      llvmBuilder.positionBuilderAtEnd(thenLastBB);
      new LLVMBranchInstruction(llvmBuilder, nextBB);
      if (elseLastBB != null) {
        llvmBuilder.positionBuilderAtEnd(elseLastBB);
        new LLVMBranchInstruction(llvmBuilder, nextBB);
      }
      llvmBuilder.positionBuilderAtEnd(nextBB);

      endStatementPragmas(node);
    }
    public void visit(Label node) {
      checkLabeledStatementPragmas(node);
      jumpScopeTable.setGotoLabelScope(node);
      // Procedure's visitor method adds all its labels before traversing its
      // body so that forward references to labels are possible.
      final LLVMBasicBlock bb = srcSymbolTable.getLabelBasicBlock(node);
      basicBlockSetStack.registerBasicBlock(bb);
      new LLVMBranchInstruction(llvmBuilder, bb);
      llvmBuilder.positionBuilderAtEnd(bb);
    }
    public void visit(NullStatement node) {
      // ISO C99 sec. 6.8.3p3. Nothing to do.
      assert(node.getChildren().isEmpty());
    }
    /** Pruned traversal so can perform actions pre and post. */
    public void visit(ReturnStatement node) {
      startStatementPragmas(node);
      {
        FlatIterator<Traversable> itr = new FlatIterator<>(node);
        while (itr.hasNext())
          visitTree(itr.next());
      }
      final SrcType retType = srcFunctionType.getReturnType();
      if (node.getExpression() == null) {
        new LLVMReturnInstruction(llvmBuilder, null);
        if (!retType.iso(SrcVoidType))
          throw new SrcRuntimeException(
            "return statement without expression in function \""
            + procedure.getSymbolName()
            + "\", which has non-void return type");
      }
      else {
        if (retType.iso(SrcVoidType))
          throw new SrcRuntimeException(
            "return statement with expression in function \""
            + procedure.getSymbolName() + "\", which has void return type");
        new LLVMReturnInstruction(
          llvmBuilder,
          postOrderValuesAndTypes.remove(postOrderValuesAndTypes.size()-1)
          .convertForAssign(retType, AssignKind.RETURN_STATEMENT, llvmModule,
                            llvmBuilder, warningsAsErrors));
      }
      // Any instructions placed in the following block are unreachable, but
      // the source code might contain unreachable code, or we will blindly
      // generate a terminator instruction here to start/finish a construct.
      llvmBuilder.positionBuilderAtEnd(
        basicBlockSetStack.registerBasicBlock(".ret.dead", llvmFunction));
      endStatementPragmas(node);
    }
    /** Pruned traversal so can perform actions pre and post. */
    public void visit(SwitchStatement node) {
      startStatementPragmas(node);
      jumpScopeTable.startScope(node);

      // Evaluate controlling expression.
      visitTree(node.getExpression());
      final ValueAndType controlPrep
        = postOrderValuesAndTypes.remove(postOrderValuesAndTypes.size()-1)
          .prepareForOp(llvmModule, llvmBuilder);
      final SrcIntegerType controlPrepIntegerType
        = controlPrep.getSrcType().toIso(SrcIntegerType.class);
      if (controlPrepIntegerType == null)
        throw new SrcRuntimeException(
          "switch statement's controlling expression is not of integer type");
      final ValueAndType controlPromote
        = controlPrepIntegerType
          .integerPromoteNoPrep(controlPrep.getLLVMValue(), llvmModule,
                                llvmBuilder);

      // Evaluate and record each contained case statement and default
      // statement.
      final Map<LLVMConstant, LLVMBasicBlock> caseMapByValue
        = new IdentityHashMap<>();
      LLVMBasicBlock defaultBB = null;
      {
        final DepthFirstIterator<Traversable> caseDefaultItr
          = new DepthFirstIterator<>(node.getBody());
        // Must skip case and default from nested switch.
        caseDefaultItr.pruneOn(SwitchStatement.class);
        // To save time, skip some nodes that cannot contain case or default.
        caseDefaultItr.pruneOn(Declaration.class);
        caseDefaultItr.pruneOn(Expression.class);
        while (caseDefaultItr.hasNext()) {
          final Traversable descendant = caseDefaultItr.next();
          if (descendant instanceof Case) {
             final Case caseNode = (Case)descendant;
             visitTree(caseNode.getExpression());
             final ValueAndType exprPrep
               = postOrderValuesAndTypes.remove(
                   postOrderValuesAndTypes.size()-1)
                 .prepareForOp(llvmModule, llvmBuilder);
             if (!exprPrep.getSrcType().iso(SrcIntegerType.class))
               throw new SrcRuntimeException(
                 "case statement's expression is not of integer type");
             if (!(exprPrep.getLLVMValue() instanceof LLVMConstantInteger))
               throw new SrcRuntimeException(
                 "case statement's expression is not a constant expression");
             final LLVMConstantInteger exprConv
               = (LLVMConstantInteger)controlPromote.getSrcType()
                 .convertFromNoPrep(exprPrep, "case statement's expression",
                                    llvmModule, llvmBuilder);
             if (caseMapByValue.get(exprConv) != null)
               throw new SrcRuntimeException(
                 "case statement's value is not unique: "
                 + (controlPromote.getSrcType().toIso(SrcIntegerType.class)
                    .isSigned()
                    ? exprConv.getSExtValue() : exprConv.getZExtValue()));
             final LLVMBasicBlock bb
               = basicBlockSetStack.createBasicBlockEarly(".case",
                                                            llvmFunction);
             caseMapByValue.put(exprConv, bb);
             srcSymbolTable.addCase(caseNode, bb);
          }
          else if (descendant instanceof Default) {
             final Default defaultNode = (Default)descendant;
             if (defaultBB != null)
               throw new SrcRuntimeException(
                 "multiple default statements in switch statement");
             final LLVMBasicBlock bb
               = basicBlockSetStack.createBasicBlockEarly(".default",
                                                            llvmFunction);
             srcSymbolTable.addDefault(defaultNode, bb);
             defaultBB = bb;
          }
        }
      }

      // Generate switch instruction.
      final LLVMBasicBlock nextBB
        = basicBlockSetStack.registerBasicBlock(".switch.nextBB",
                                                  llvmFunction);
      final LLVMSwitchInstruction switchInsn
        = new LLVMSwitchInstruction(llvmBuilder, controlPromote.getLLVMValue(),
                                    defaultBB == null ? nextBB : defaultBB,
                                    caseMapByValue.size());
      for (Map.Entry<LLVMConstant,LLVMBasicBlock> e
           : caseMapByValue.entrySet())
        switchInsn.addCase(e.getKey(), e.getValue());

      // Any instructions placed in the following block are unreachable, but
      // the source code might contain unreachable code, or we will blindly
      // generate a terminator instruction here to start/finish a construct.
      llvmBuilder.positionBuilderAtEnd(
        basicBlockSetStack.registerBasicBlock(".switch.dead", llvmFunction));

      // Record break target before evaluating body.
      breakTargetStack.push(nextBB);

      // Evaluate body, which might place dead code into the above block, or
      // it might start immediately with a labeled statement. See ISO C99 sec.
      // 6.8.4.2p7 for an example of dead code.
      visitTree(node.getBody());
      new LLVMBranchInstruction(llvmBuilder, nextBB);
      llvmBuilder.positionBuilderAtEnd(nextBB);

      // Pop break target.
      breakTargetStack.pop();

      jumpScopeTable.endScope(node);
      endStatementPragmas(node);
    }
    /** Pruned traversal so can perform actions pre and post. */
    public void visit(WhileLoop node) {
      startStatementPragmas(node);
      jumpScopeTable.startScope(node);
      final LLVMBasicBlock prevBB = llvmBuilder.getInsertBlock();

      // Evaluate controlling expression as an i1 value.
      final LLVMBasicBlock condFirstBB
        = basicBlockSetStack.registerBasicBlock(".while.condFirstBB",
                                                  llvmFunction);
      llvmBuilder.positionBuilderAtEnd(condFirstBB);
      visitTree(node.getCondition());
      final LLVMValue cond
        = postOrderValuesAndTypes.remove(postOrderValuesAndTypes.size()-1)
          .evalAsCond("controlling expression of while statement",
                      llvmModule, llvmBuilder);
      final LLVMBasicBlock condLastBB = llvmBuilder.getInsertBlock();

      // Record continue and break targets before evaluating body.
      final LLVMBasicBlock nextBB
        = basicBlockSetStack.registerBasicBlock(".while.nextBB",
                                                  llvmFunction);
      continueTargetStack.push(condFirstBB);
      breakTargetStack.push(nextBB);

      // Evaluate body.
      final LLVMBasicBlock bodyFirstBB
        = basicBlockSetStack.registerBasicBlock(".while.bodyFirstBB",
                                                  llvmFunction);
      llvmBuilder.positionBuilderAtEnd(bodyFirstBB);
      visitTree(node.getBody());
      final LLVMBasicBlock bodyLastBB = llvmBuilder.getInsertBlock();

      // Pop continue and break targets.
      continueTargetStack.pop();
      breakTargetStack.pop();

      // Add control flow.
      llvmBuilder.positionBuilderAtEnd(prevBB);
      new LLVMBranchInstruction(llvmBuilder, condFirstBB);
      llvmBuilder.positionBuilderAtEnd(condLastBB);
      new LLVMBranchInstruction(llvmBuilder, cond, bodyFirstBB, nextBB);
      llvmBuilder.positionBuilderAtEnd(bodyLastBB);
      new LLVMBranchInstruction(llvmBuilder, condFirstBB);
      llvmBuilder.positionBuilderAtEnd(nextBB);

      jumpScopeTable.endScope(node);
      endStatementPragmas(node);
    }
    /** Pruned traversal so can perform actions pre and post. */
    public void visit(TranslationUnit node) {
      jumpScopeTable.startScope(node);
      tu = node;

      // Set up the module.
      ++llvmModuleIndex;
      llvmModule = llvmModules[llvmModuleIndex]
        = new LLVMModule(node.getInputFilename(), llvmContext);
      llvmModule.setTargetTriple(llvmTargetTriple);
      llvmModule.setDataLayout(llvmTargetData.stringRepresentation());
      llvmModuleIdentifiers[llvmModuleIndex] = node.getInputFilename();

      // Traverse the translation unit.
      {
        FlatIterator<Traversable> itr = new FlatIterator<>(node);
        while (itr.hasNext())
          visitTree(itr.next());
      }

      srcSymbolTable.initTentativeDefinitions(llvmModuleIndex, llvmModule);
      srcSymbolTable.pruneUnusedFunctions(llvmModuleIndex);
      srcSymbolTable.checkNVMStoredStructs();
      srcSymbolTable.initializeTypeChecksumVariables(llvmModule,
                                                     llvmModuleIndex);

      if (dumpLLVMModules)
        llvmModule.dump();

      jumpScopeTable.endScope(node);
    }
  }

  /**
   * Free LLVM memory associated with this object. See {@link #getLLVMModules}
   * for some important notes.
   */
  public void finalize() {
    // Disposing of the context disposes of the modules.
    llvmContext.dispose();
    llvmTargetData.dispose();
  }
}
