package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SRC_PTRDIFF_T_TYPE;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SRC_SIZE_T_TYPE;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcCharType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcIntType;
import static openacc.codegen.llvmBackend.SrcPrimitiveNonNumericType.SrcVoidType;

import java.util.Set;

import openacc.analysis.SubArray;

import org.jllvm.LLVMConstant;
import org.jllvm.LLVMConstantInteger;

import cetus.hir.Annotation;
import cetus.hir.AnnotationStatement;
import cetus.hir.Case;
import cetus.hir.Default;
import cetus.hir.Expression;
import cetus.hir.Label;
import cetus.hir.Statement;
import cetus.hir.Traversable;

/**
 * The LLVM backend's abstract base class for pragma translators.
 * 
 * <p>
 * Each concrete subclass encapsulates translation for a set of related
 * pragmas. An object of each concrete subclass is stored in an array in
 * {@link BuildLLVM.Visitor}, which loops through the array and calls this
 * interface upon encountering pragmas.
 * </p>
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public abstract class PragmaTranslator {
  protected final BuildLLVM.Visitor v;

  /**
   * Construct a new pragma translator.
   * 
   * @param v
   *          the {@link BuildLLVM.Visitor} that will call this pragma
   *          translator
   */
  public PragmaTranslator(BuildLLVM.Visitor v) {
    this.v = v;
  }

  /**
   * Start translating any known pragmas attached to the given statement.
   * 
   * <p>
   * Called before the statement itself is translated. Thus, basic blocks
   * scopes can be pushed here.
   * </p>
   * 
   * @param node
   *          the statement, which is never a {@link Label}, {@link Case}, or
   *          {@link Default}
   */
  public final void startStatementPragmas(Statement node) {
    assert(!(node instanceof Label) && !(node instanceof Case)
           && !(node instanceof Default));
    for (StatementPragmaTranslator pragma : getStatementPragmaTranslators()) {
      final Annotation annot
        = node.getAnnotation(pragma.getAnnotationClass(), pragma.getName());
      if (annot != null)
        pragma.start(annot);
    }
  }

  /**
   * Finish translating any known pragmas attached to the given statement.
   * 
   * <p>
   * Called after the statement itself is translated. Thus, basic blocks
   * scopes can be popped here.
   * </p>
   * 
   * @param node
   *          the statement, which is never a {@link Label}, {@link Case}, or
   *          {@link Default}
   */
  public final void endStatementPragmas(Statement node) {
    assert(!(node instanceof Label) && !(node instanceof Case)
        && !(node instanceof Default));
    StatementPragmaTranslator[] pragmas = getStatementPragmaTranslators();
    for (int i = pragmas.length-1; i >= 0; --i) {
      StatementPragmaTranslator pragma = pragmas[i];
      final Annotation annot
        = node.getAnnotation(pragma.getAnnotationClass(), pragma.getName());
      if (annot != null)
        pragma.end(annot);
    }
  }

  /**
   * Complain for any known pragmas attached to a labeled statement.
   * 
   * <p>
   * In general, we do not allow basic block scopes to be specified by
   * pragmas attached to labeled statements (including case and default)
   * because the semantics are confusing and often broken.
   * </p>
   *
   * <p>
   * OpenMP has the same restriction for some of its directives because a
   * labeled statement has more than one entry point: one before the label,
   * and one after the label. Thus, initialization code automatically
   * inserted before the label wouldn't always execute. It's up to the user
   * to move the pragma after the label if he wants initialization code to
   * execute there.
   * </p>
   * 
   * <p>
   * Even for pragmas that do not involve initialization code, the semantics
   * of a pragma on a labeled statement is confusing, so we generally do no
   * permit it. Does it apply to just the label? That interpretation doesn't
   * seem meaningful for any pragma, but the Cetus IR encourages that
   * interpretation because it treats a label as a {@link Statement} that is
   * separate from the statement following the label, so it's easiest to
   * implement this interpretation even if it doesn't make sense. Does it
   * apply to the statement following the label? That's probably the most
   * reasonable interpretation based on the C grammar, but it's probably not
   * what a C programmer would expect for a case or default. Does it apply
   * to the everything between a case/default and the next break? That
   * probably is what a C programmer would expect, but it doesn't generalize
   * well (there might not be a break, or the break might be guarded by an
   * if and there might be a following case that is subsumed when the if
   * condition is false).
   * </p>
   *
   * <p>
   * TODO: Our LLVM pass should complain if a resilience region has multiple
   * entry points. That would actually catch this case except when a goto
   * label has no corresponding gotos, but it's a confusing case anyway, and
   * it's easy to detect here, so we complain here. Multiple exit points is
   * not a problem for us because we can add deinit code at every exit, but
   * an LLVM pass could complain about that if it turns out there's a reason
   * to.
   * </p>
   * 
   * @param node
   *          a {@link Label}, {@link Case}, or {@link Default} node
   * @throws SrcRuntimeException
   *           for any offending pragma attached to {@code node}
   */
  public final void checkLabeledStatementPragmas(Statement node) {
    assert(node instanceof Label || node instanceof Case
           || node instanceof Default);
    for (StatementPragmaTranslator pragma : getStatementPragmaTranslators()) {
      final Annotation annot
        = node.getAnnotation(pragma.getAnnotationClass(), pragma.getName());
      if (annot != null)
        throw new SrcRuntimeException(
          pragma.getName()+" pragma cannot be applied to a labeled statement");
    }
  }

  /**
   * Translator for an individual pragma that can be attached to a statement.
   */
  protected static abstract class StatementPragmaTranslator {
    private final String name;
    private final Class<? extends Annotation> annotationClass;

    /**
     * Construct a new statement pragma translator.
     * 
     * @param annotationClass
     *   the {@link Annotation} subclass as which the pragma is attached to
     *   statements
     * @param name
     *   the name/key of the pragma as used in the source code
     */
    public StatementPragmaTranslator(
      Class<? extends Annotation> annotationClass, String name)
    {
      this.name = name;
      this.annotationClass = annotationClass;
    }

    public Class<? extends Annotation> getAnnotationClass() {
      return annotationClass;
    }
    public String getName() {
      return name;
    }

    /**
     * Same as {@link PragmaTranslator#startStatementPragmas} but for a single
     * pragma of this kind found to be attached to a statement.
     * 
     * @param annot
     *          the annotation extracted for the found pragma. It's always
     *          safe to cast {@code annot} to the class returned by
     *          {@link #getAnnotationClass}.
     */
    public abstract void start(Annotation annot);

    /**
     * Same as {@link PragmaTranslator#endStatementPragmas} but for a single
     * pragma of this kind found to be attached to a statement.
     * 
     * @param annot
     *          the annotation extracted for the found pragma. It's always
     *          safe to cast {@code annot} to the class returned by
     *          {@link #getAnnotationClass}.
     */
    public abstract void end(Annotation annot);
  }

  /**
   * Get translators for pragmas that can be attached to statements and that
   * are handled by this pragma translator.
   * 
   * <p>
   * If multiple translators apply to the same pragma, then translators
   * appearing earlier in the list enclose translators appearing later in the
   * list. That is, their {@link StatementPragmaTranslator#start} methods are
   * called in order, and their {@link StatementPragmaTranslator#end} methods
   * are called in reverse order.
   * </p>
   */
  protected abstract StatementPragmaTranslator[]
    getStatementPragmaTranslators();

  /** Translate any known standalone pragmas represented by the given node. */
  public abstract void translateStandalonePragmas(AnnotationStatement node);

  /**
   * Evaluate a pragma clause whose operand is expected to be an expression of
   * integer type.
   * 
   * @param annot
   *          the annotation specifying the pragma
   * @param pragmaName
   *          the pragma name, as specified in source code
   * @param clauseName
   *          the clause name, as specified in source code
   * @param defaultValue
   *          the default value, used when the clause is not present. It is
   *          assumed to be an rvalue of type {@code int}.
   * @return the resulting value, possibly an lvalue
   * @throws SrcRuntimeException
   *           if the clause's operand is not of integer type
   */
  protected final ValueAndType evalPragmaIntegerClause(
    Annotation annot, String pragmaName, String clauseName,
    long defaultValue)
  {
    final Expression expr = (Expression)annot.get(clauseName);
    if (expr == null)
      return new ValueAndType(
        LLVMConstantInteger.get(SrcIntType.getLLVMType(v.llvmContext),
                                defaultValue, SrcIntType.isSigned()),
        SrcIntType, false);
    v.srcSymbolTable.addParentFixup(expr, annot.getAnnotatable());
    v.visitTree(expr);
    final ValueAndType res
      = v.postOrderValuesAndTypes.remove(v.postOrderValuesAndTypes.size()-1);
    if (!res.getSrcType().prepareForOp().iso(SrcIntegerType.class))
      throw new SrcRuntimeException(
        pragmaName+" pragma's "+clauseName+" clause has non-integer type");
    return res;
  }

  /**
   * Evaluate a pragma clause whose operand is expected to be a modifiable
   * lvalue of integer type.
   * 
   * @param annot
   *          the annotation specifying the pragma
   * @param pragmaName
   *          the pragma name, as specified in source code
   * @param clauseName
   *          the clause name, as specified in source code
   * @return an rvalue holding the address of the given lvalue. If the clause
   *         is not specified, then the result is a null pointer constant of
   *         type {@code char*}.
   * @throws SrcRuntimeException
   *           if the clause's operand is not a modifiable lvalue of integer
   *           type
   */
  protected final ValueAndType evalPragmaIntegerLvalueClause(
    Annotation annot, String pragmaName, String clauseName)
  {
    final Expression expr = (Expression)annot.get(clauseName);
    if (expr == null) {
      final SrcPointerType srcType = SrcPointerType.get(SrcCharType);
      return new ValueAndType(
        LLVMConstant.constNull(srcType.getLLVMType(v.llvmContext)),
        srcType, false);
    }
    v.srcSymbolTable.addParentFixup(expr, annot.getAnnotatable());
    v.visitTree(expr);
    final ValueAndType res
      = v.postOrderValuesAndTypes.remove(v.postOrderValuesAndTypes.size()-1);
    if (!res.isModifiableLvalue()
        || !res.getSrcType().iso(SrcIntegerType.class))
      throw new SrcRuntimeException(
        pragmaName+" pragma's "+clauseName+" clause is not a modifiable"
        + " lvalue of integer type");
    return res.address(v.llvmModule, v.llvmBuilder);
  }

  /**
   * Evaluate a pragma data clause, such as the {@code ftdata} clause for
   * the {@code openarc ftinject} pragma.
   * 
   * @param annot
   *          the annotation specifying the pragma
   * @param pragmaName
   *          the pragma name, as specified in source code
   * @param clauseName
   *          the clause name, as specified in source code
   * @return a {@link ValueType} 2-dimensional array such that each row
   *         encodes one array specification from the data clause and has
   *         the format
   *         {@code [addr, startElement, numElements, elementSize]}. This
   *         array specification flattens multi-dimensional array
   *         specifications. Thus, {@code addr}'s type is a pointer to the
   *         innermost element type. For example, if the original type was
   *         {@code int[5][6]}, then {@code addr}'s type is {@code int*}.
   * @throws SrcRuntimeException
   *           if errors are detected in the clause
   */
  protected final ValueAndType[][] evalPragmaDataClause(
    Annotation annot, String pragmaName, String clauseName)
  {
    final Set<SubArray> data = annot.get(clauseName);
    if (data == null)
      return new ValueAndType[0][4];
    final ValueAndType[][] res = new ValueAndType[data.size()][4];
    final Traversable parent = annot.getAnnotatable();
    int i = 0;
    for (SubArray arr : data)
      res[i++] = evalPragmaDataClauseSubArray(parent, pragmaName,
                                              clauseName, arr);
    return res;
  }

  /**
   * Evaluate a {@link SubArray} from a pragma data clause, such as the
   * {@code ftdata} clause for the {@code openarc ftinject} pragma.
   * 
   * @param parent
   *          the context in which to evaluate. If the {@code arr} comes
   *          from an {@link Annotation} {@code annot}, then {@code parent}
   *          is normally {@code annot.getAnnotatable()}.
   * @param pragmaName
   *          the pragma name, as specified in source code
   * @param clauseName
   *          the clause name, as specified in source code
   * @param arr
   *          the {@link SubArray} to evaluate, or null
   * @return a {@link ValueType} array encoding {@code arr} in the format
   *         {@code [addr, startElement, numElements, elementSize]}. This
   *         array specification flattens multi-dimensional array
   *         specifications. Thus, {@code addr}'s type is a pointer to the
   *         innermost element type. For example, if the original type was
   *         {@code int[5][6]}, then {@code addr}'s type is {@code int*}. If
   *         {@code arr} is null, all elements in the result are set to null
   *         values.
   * @throws SrcRuntimeException
   *           if errors are detected in {@link SubArray}
   */
  protected final ValueAndType[] evalPragmaDataClauseSubArray(
    Traversable parent, String pragmaName, String clauseName, SubArray arr)
  {
    if (arr == null)
      return new ValueAndType[]{
        new ValueAndType(
          LLVMConstant.constNull(SrcPointerType.get(SrcVoidType)
                                 .getLLVMType(v.llvmContext)),
          SrcPointerType.get(SrcVoidType), false),
        new ValueAndType(
          LLVMConstant.constNull(SRC_PTRDIFF_T_TYPE
                                 .getLLVMType(v.llvmContext)),
          SRC_PTRDIFF_T_TYPE, false),
        new ValueAndType(
          LLVMConstant.constNull(SRC_SIZE_T_TYPE.getLLVMType(v.llvmContext)),
          SRC_SIZE_T_TYPE, false),
        new ValueAndType(
          LLVMConstant.constNull(SRC_SIZE_T_TYPE.getLLVMType(v.llvmContext)),
          SRC_SIZE_T_TYPE, false),
      };

    // This algorithm follows Seyong's explanation to me of the way OpenARC
    // handles the ftdata clause for pragmas like "openarc ftinject".
    //
    // If it's specified as a pointer, then treat it as an array whose first
    // element is pointed to by the pointer, and require numElements to be
    // specified.
    //
    // Else if it's specified as an array, then numElements defaults to the
    // number of elements in the array minus startElement.
    // 
    // Else if it's specified as an lvalue, then treat it as a one-element
    // array.
    //
    // Else, it's a non-pointer rvalue, so it's an error. Actually, this case
    // is probably currently unreachable because
    // ACCAnalysis.updateSymbolsInSubArray always seems to catch it and
    // complain.
    v.srcSymbolTable.addParentFixup(arr.getArrayName(), parent);
    v.visitTree(arr.getArrayName());
    final ValueAndType expr
      = v.postOrderValuesAndTypes.remove(v.postOrderValuesAndTypes.size()-1);
    SrcPointerType ptrType = expr.getSrcType().toIso(SrcPointerType.class);
    SrcArrayType arrayType = expr.getSrcType().toIso(SrcArrayType.class);
    if (arrayType != null && arrayType.isIncompleteType())
      throw new SrcRuntimeException(
        pragmaName+" pragma's "+clauseName+" clause has expression of"
        +" incomplete array type");
    ValueAndType addr;
    SrcType elementType;
    ValueAndType startElement = null; // init suppresses compiler error
    ValueAndType numElements = null;  // init suppresses compiler error
    if (ptrType != null || arrayType != null) {
      addr = expr;
      int dim = 0;
      // Loop through dimensions assuming contiguous memory (that is, all
      // dimensions beyond the first specify the full dimension), and convert
      // to the equivalent one-dimensional startElement and numElements.
      do {
        // TODO: For dim > 0: (1) complain if startElementSub is not 0, and
        // (2) complain if numElementsSub is not the size of the array (unless
        // it's a pointer).
        ValueAndType startElementSub;
        if (arr.getArrayDimension() < dim+1)
          startElementSub = new ValueAndType(
            LLVMConstant.constNull(SRC_PTRDIFF_T_TYPE
                                   .getLLVMType(v.llvmContext)),
            SRC_PTRDIFF_T_TYPE, false);
        else {
          Expression indexExpr = arr.getStartIndices().get(dim);
          v.srcSymbolTable.addParentFixup(indexExpr, parent);
          v.visitTree(indexExpr);
          startElementSub = v.postOrderValuesAndTypes.remove(
                              v.postOrderValuesAndTypes.size()-1);
        }
        ValueAndType numElementsSub;
        if (arr.getArrayDimension() < dim+1
            || arr.getLengths().get(dim) == null)
        {
          if (ptrType != null)
            throw new SrcRuntimeException(
              pragmaName+" pragma's "+clauseName+" clause dimension "+dim
              +" is of pointer type but has no specified length");
          assert(arrayType != null);
          final ValueAndType size = new ValueAndType(
            LLVMConstantInteger.get(
              SRC_SIZE_T_TYPE.getLLVMType(v.llvmContext),
              arrayType.getNumElements(), true),
            SRC_SIZE_T_TYPE, false);
          numElementsSub = ValueAndType.subtract(
            pragmaName+" pragma's "+clauseName+" clause dimension "+dim
            +" length computation",
            size, startElementSub, v.llvmTargetData, v.llvmModule,
            v.llvmBuilder);
        }
        else {
          final Expression lengthExpr = arr.getLengths().get(dim);
          v.srcSymbolTable.addParentFixup(lengthExpr, parent);
          v.visitTree(lengthExpr);
          numElementsSub = v.postOrderValuesAndTypes
                           .remove(v.postOrderValuesAndTypes.size()-1);
        }
        if (dim == 0) {
          startElement = startElementSub;
          numElements = numElementsSub;
        }
        else {
          startElement = ValueAndType.multiply(startElement, numElementsSub,
                                               v.llvmModule, v.llvmBuilder);
          startElement = ValueAndType.add(
            pragmaName+" pragma's "+clauseName+" clause startElement"
            +" computation",
            startElement, startElementSub, v.llvmModule, v.llvmBuilder);
          numElements = ValueAndType.multiply(numElements, numElementsSub,
                                              v.llvmModule, v.llvmBuilder);
        }
        elementType = ptrType != null ? ptrType.getTargetType()
                                      : arrayType.getElementType();
        ptrType = elementType.toIso(SrcPointerType.class);
        arrayType = elementType.toIso(SrcArrayType.class);
        if (arrayType != null)
          // Due to the indirection call, when this loop is done, we'll have
          // the next to the innermost type. If that's an array, the
          // prepareForOp call turns it into a pointer so that, in case the
          // original array was multi-dimensional, there's no potential
          // confusion over the meaning of the array length.
          addr = addr.indirection("argument to "+pragmaName+" pragma's "
                                  +clauseName+" clause",
                                  v.llvmModule, v.llvmBuilder)
                 .prepareForOp(v.llvmModule, v.llvmBuilder);
        ++dim;
      } while (arrayType != null);
      if (dim < arr.getArrayDimension())
        throw new SrcRuntimeException(
          pragmaName+" pragma's "+clauseName+" clause has too many"
          +" dimensions");
    }
    else if (expr.isLvalue()) {
      if (arr.getArrayDimension() > 0)
        throw new SrcRuntimeException(
          pragmaName+" pragma's "+clauseName+" clause has lvalue of"
          +" non-pointer type but also has dimensions");
      addr = expr.address(v.llvmModule, v.llvmBuilder);
      elementType = expr.getSrcType();
      startElement = new ValueAndType(
        LLVMConstant.constNull(SRC_PTRDIFF_T_TYPE.getLLVMType(v.llvmContext)),
        SRC_PTRDIFF_T_TYPE, false);
      numElements = new ValueAndType(
        LLVMConstantInteger.get(SRC_SIZE_T_TYPE.getLLVMType(v.llvmContext),
                                1, true),
        SRC_SIZE_T_TYPE, false);
    }
    else
      throw new SrcRuntimeException(
        pragmaName+" pragma's "+clauseName+" clause has rvalue of"
        +" non-pointer type");

    // Get the element size.
    if (elementType.isIncompleteType())
      throw new SrcRuntimeException(
        pragmaName+" pragma's "+clauseName+" clause has expression with"
        +" incomplete element type");
    final ValueAndType elementSize = new ValueAndType(
      LLVMConstantInteger.get(
        SRC_SIZE_T_TYPE.getLLVMType(v.llvmContext),
        v.llvmTargetData.abiSizeOfType(elementType.getLLVMType(v.llvmContext)),
        SRC_SIZE_T_TYPE.isSigned()),
      SRC_SIZE_T_TYPE, false);

    return new ValueAndType[]{addr, startElement, numElements, elementSize};
  }
}