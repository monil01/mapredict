package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SRC_WCHAR_TYPE;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcBoolType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcCharType;
import static openacc.codegen.llvmBackend.SrcPrimitiveNonNumericType.SrcVoidType;

import java.util.EnumSet;

import openacc.codegen.llvmBackend.SrcQualifiedType.SrcTypeQualifier;
import openacc.codegen.llvmBackend.SrcType.AndXorOr;

import org.jllvm.LLVMArrayType;
import org.jllvm.LLVMConstant;
import org.jllvm.LLVMConstantArray;
import org.jllvm.LLVMFunctionType;
import org.jllvm.LLVMInstructionBuilder;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMTargetData;
import org.jllvm.LLVMType;
import org.jllvm.LLVMValue;

/**
 * A triple: an {@link LLVMValue}, the {@link SrcType} that originally
 * specified its {@link LLVMType}, and a boolean indicating whether it
 * represents either (1) an lvalue or function designator, or (2) an rvalue.
 * 
 * <p>
 * Instances of this classes are immutable. See {@link #ValueAndType}
 * documentation for how to interpret the triple.
 * </p>
 */
public class ValueAndType {
  private final LLVMValue llvmValue;
  private final SrcType srcType;
  private final boolean lvalueOrFnDesignator;
  public LLVMValue getLLVMValue() { return llvmValue; }
  public SrcType getSrcType() { return srcType; }

  /** Is this an lvalue? (ISO C99 sec. 6.3.2.1p1.) */
  public boolean isLvalue() {
    return lvalueOrFnDesignator && !srcType.iso(SrcFunctionType.class);
  }

  /** Is this a modifiable lvalue? (ISO C99 sec. 6.3.2.1p1.) */
  public boolean isModifiableLvalue() {
    return isLvalue()
           && !srcType.iso(SrcArrayType.class)
           && !srcType.isIncompleteType()
           && !srcType.storageHasQualifier(SrcTypeQualifier.CONST);
  }

  /**
   * Construct a new value and its type.
   * 
   * <p>
   * A void expression (ISO C99 sec. 6.3.2.2) is always represented with
   * {@code llvmValue} as null, {@code srcType} as {@link #SrcVoidType}, and
   * {@code lvalueOrFnDesignator} as false.
   * </p>
   * 
   * <p>
   * If {@code srcType.isa(SrcBitFieldType.class)}, then this represents a
   * bit-field. If, additionally, {@code lvalueOrFnDesignator} is false, then
   * {@code llvmValue}'s type is the bit-field's full storage unit type, and
   * {@code llvmValue}'s full value, including any bits that would otherwise
   * store other bit-fields, exactly equals just the single bit-field's value.
   * See {@link SrcBitFieldType}'s documentation for further details.
   * </p>
   * 
   * <p>
   * ISO C99 sec. 6.7.3p3, 6.3.2.1p2, and 6.5.4f89 make it clear that type
   * qualifiers on rvalues should be ignored.
   * </p>
   * 
   * @param llvmValue
   *          the value. Null only for a void expression.
   * @param srcType
   *          the type of the value. {@link #SrcVoidType} only for a void
   *          expression.
   * @param lvalueOrFnDesignator
   *          true if this is an lvalue or, when {@code srcType} is a
   *          {@link SrcFunctionType}, a function designator. Otherwise, this
   *          is an rvalue. If {@code lvalueOrFnDesignator} is true, then
   *          {@code llvmValue}'s type must be a pointer to the type specified
   *          by {@code srcType}. If {@code lvalueOrFnDesignator} is false
   *          instead, then {@code llvmValue}'s type must be exactly the type
   *          specified by {@code srcType}, and it must not be a function (but
   *          a pointer to a function is fine), and it must not be an array
   *          (as hinted at by ISO C99 sec. 6.3.2.1p2 because it doesn't
   *          convert an array to an rvalue) unless it's a string literal or a
   *          constant-expression compound initializer (the value is an
   *          {@link LLVMConstantArray} in both cases; see
   *          {@link #isStringLiteral}).
   */
  public ValueAndType(LLVMValue llvmValue, SrcType srcType,
                      boolean lvalueOrFnDesignator)
  {
    if (llvmValue == null || srcType.iso(SrcVoidType)) {
      // void expression
      assert(llvmValue == null);
      assert(srcType.eqv(SrcVoidType));
      assert(lvalueOrFnDesignator == false);
    }
    else if (lvalueOrFnDesignator) {
      // lvalue or function designator
      assert(llvmValue.typeOf()
             == SrcPointerType.get(srcType).getLLVMType(llvmValue.typeOf()
                                                        .getContext()));
    }
    else {
      // rvalue
      assert(llvmValue.typeOf() == srcType.getLLVMType(llvmValue.typeOf()
                                                       .getContext()));
      assert(!(srcType.iso(SrcFunctionType.class))
             && !(llvmValue.typeOf() instanceof LLVMFunctionType));
      assert((!(srcType.iso(SrcArrayType.class))
              && !(llvmValue.typeOf() instanceof LLVMArrayType))
             || llvmValue instanceof LLVMConstant);
    }
    this.llvmValue = llvmValue;
    this.srcType = srcType;
    this.lvalueOrFnDesignator = lvalueOrFnDesignator;
  }

  /**
   * Create a void expression. See
   * {@link #ValueAndType(LLVMValue, SrcType, boolean)}.
   */
  public ValueAndType() {
    llvmValue = null;
    srcType = SrcVoidType;
    lvalueOrFnDesignator = false;
  }

  /**
   * Is this a null pointer constant? (ISO C99 sec. 6.3.2.3p3.)
   * 
   * <p>
   * It does not matter whether this is the result of a {@link #prepareForOp}
   * call, which does not affect or create a null pointer constant.
   * </p>
   * 
   * @return true iff this is a null pointer constant
   */
  public boolean isNullPointerConstant() {
    if (!(llvmValue instanceof LLVMConstant))
      return false;
    if (!((LLVMConstant)llvmValue).isNullValue())
      return false;
    if (srcType.iso(SrcIntegerType.class))
      return true;
    final SrcPointerType ptrType = srcType.toIso(SrcPointerType.class);
    if (ptrType == null)
      return false;
    return ptrType.getTargetType().eqv(SrcVoidType);
  }

  /**
   * Is this a string literal?
   * 
   * <p>
   * This must not be the result of a {@link #prepareForOp} call, which would
   * convert a string literal into a pointer.
   * </p>
   * 
   * @return true iff this is a string literal (or a constant-expression
   *         initializer that is effectively a string literal)
   */
  public boolean isStringLiteral() {
    // See documentation for ValueAndType constructor.
    if (!(llvmValue instanceof LLVMConstantArray))
      return false;
    final SrcType elementType
      = srcType.toIso(SrcArrayType.class).getElementType();
    return elementType.eqv(SrcCharType) || elementType.eqv(SRC_WCHAR_TYPE);
  }

  /**
   * Perform the usual conversions normally required when used as an operand.
   * 
   * <p>
   * That is, ISO C99 sec. 6.3.2.1: lvalue becomes rvalue, array becomes
   * pointer (so string literal is converted to an LLVM private unnamed_addr
   * constant), and function designator becomes pointer.
   * </p>
   * 
   * <p>
   * ISO C99 sec. 6.3.2.1p2 says that {@link #prepareForOp} must drop type
   * qualifiers when converting lvalues to rvalues. {@link #prepareForOp} is
   * otherwise concerned with array types and function types, which have no
   * type qualifiers, as documented at {@link SrcQualifiedType}. Type
   * qualifiers on rvalues are generally ignored in C, as mentioned in the
   * comments for {@link #ValueAndType(LLVMValue, SrcType, boolean)}. Thus,
   * the result's type is always a {@link SrcBaldType}. TODO: Dropping type
   * qualifiers and, perhaps as a result, expanding top-level typedefs, losing
   * information that might be helpful in diagnostics, so we might want to
   * reconsider this behavior. I'm not sure why it matters that we drop type
   * qualifiers when converting lvalues to rvalues given that type qualifiers
   * on rvalues, which could be produced by an explicit cast, are generally
   * ignored.
   * </p>
   * 
   * @param llvmModule
   *          the LLVM module into which any required declarations should be
   *          generated for the conversions
   * @param llvmBuilder
   *          the IR builder with which any LLVM instructions should be
   *          generated for the conversions
   * @return the new rvalue and type, which together might be the same as the
   *         contents of {@code this} if no LLVM operations were required to
   *         perform the conversions and no type changes were required.
   */
  public ValueAndType prepareForOp(LLVMModule llvmModule,
                                   LLVMInstructionBuilder llvmBuilder)
  {
    return srcType.prepareForOp(llvmValue, lvalueOrFnDesignator, llvmModule,
                                llvmBuilder);
  }

  /**
   * Perform the default argument promotions.
   * 
   * <p>
   * That is, ISO C99 sec. 6.5.2.2p6-7: integer promotions plus float
   * becomes double.
   * </p>
   * 
   * <p>
   * This should not be the result of a {@link #prepareForOp} call as that
   * will be called here if appropriate.
   * </p>
   * 
   * <p>
   * Warnings or errors (depending on the {@code warn} and
   * {@code warningAsErrors} arguments) about default argument promotion
   * failures (because the type involves NVM storage) might not be
   * detectable or reported until an incomplete type is completed. In that
   * case, the incomplete type is used as is in the default argument
   * promotion now.
   * </p>
   * 
   * @param warn
   *          whether default argument promotion failures should be reported
   *          as warnings instead of errors
   * @param warningsAsErrors
   *          whether to treat warnings as errors. This is irrelevant if
   *          {@code warn} is false. (Default argument promotion failures
   *          are errors if either {@code warn} is false or both
   *          {@code warn} and {@code warningsAsErrors} are true, but the
   *          message in the latter case points out that a warning is being
   *          treated as an error.)
   * @param msgPrefix
   *          message prefix for any error or warning reported
   * @param llvmModule
   *          the LLVM module into which any required declarations should be
   *          generated for the conversions
   * @param llvmBuilder
   *          the IR builder with which any LLVM instructions should be
   *          generated for the conversions
   * @return the new value, or null if default argument promotion failed and
   *         that case is being treated as a warning. In the first case, the
   *         new value might be the same as {@code this} if no LLVM
   *         operations were required to perform the conversions and no type
   *         changes were required.
   * @throws SrcRuntimeException
   *           if default argument promotion failed and that case is being
   *           treated as an error
   */
  public LLVMValue defaultArgumentPromote(
    boolean warn, boolean warningsAsErrors, String msgPrefix,
    LLVMModule llvmModule, LLVMInstructionBuilder llvmBuilder)
  {
    final ValueAndType prep = prepareForOp(llvmModule, llvmBuilder);
    prep.srcType.checkDefaultArgumentPromote(warn, warningsAsErrors,
                                             msgPrefix);
    return prep.srcType.defaultArgumentPromoteNoPrep(prep.llvmValue,
                                                     llvmModule, llvmBuilder);
  }

  /**
   * Same as
   * {@link #defaultArgumentPromote(boolean, boolean, String, LLVMModule, LLVMInstructionBuilder)}
   * except {@code warn} is always false.
   */
  public LLVMValue defaultArgumentPromote(
    String msgPrefix, LLVMModule llvmModule,
    LLVMInstructionBuilder llvmBuilder)
  {
    return defaultArgumentPromote(false, false, msgPrefix, llvmModule,
                                  llvmBuilder);
  }

  /**
   * Store a value into an lvalue.
   * 
   * <p>
   * This must be an lvalue and so should not be the result of a
   * {@link #prepareForOp} call.
   * </p>
   * 
   * @param forInit
   *          whether the store is for an initialization. Otherwise, the type
   *          of this must not have a const qualifier, and the caller should
   *          have reported an error if it does.
   * @param value
   *          the value to store. It must have already been converted to the
   *          same type as this. Thus, in the case of a
   *          {@link SrcBitFieldType}, the LLVM type of {@code value} is the
   *          storage unit's type, and the full {@code value} including all
   *          extra bits is assumed to be exactly the bit-field's value, which
   *          is truncated and written into only the bit-field's bits for this
   *          lvalue.
   * @param llvmModule
   *          the LLVM module into which any required declarations should be
   *          generated
   * @param llvmBuilder
   *          the IR builder with which any LLVM instructions should be
   *          generated
   */
  public void store(boolean forInit, LLVMValue value, LLVMModule llvmModule,
                    LLVMInstructionBuilder llvmBuilder)
  {
    srcType.store(forInit, llvmValue, value, llvmModule, llvmBuilder);
  }

  /**
   * Load a value from an lvalue.
   * 
   * <p>
   * This must be an lvalue and so should not be the result of a
   * {@link #prepareForOp} call.
   * </p>
   * 
   * @param llvmModule
   *          the LLVM module into which any required declarations should be
   *          generated
   * @param llvmBuilder
   *          the IR builder with which any LLVM instructions should be
   *          generated
   * @return the loaded rvalue, which has the same type as this except
   *         top-level type qualifiers have been removed. Thus, in the case of
   *         a {@link SrcBitFieldType}, the LLVM type of the result is the
   *         storage unit's type, and the full value of the result including
   *         all extra bits is exactly equal to the bit-field's value.
   */
  public ValueAndType load(LLVMModule llvmModule,
                           LLVMInstructionBuilder llvmBuilder)
  {
    return srcType.load(llvmValue, llvmModule, llvmBuilder);
  }

  public static class AssignKind {
    public static final AssignKind INITIALIZATION
      = new AssignKind("initialization");
    public static final AssignKind ASSIGNMENT_OPERATOR
      = new AssignKind("assignment operator");
    public static final AssignKind RETURN_STATEMENT
      = new AssignKind( "return statement");
    private final String description;
    private AssignKind(String description) {
      this.description = description;
    }
    public final String toString() {
      return description;
    }
  };
  public static class ArgAssignKind extends AssignKind {
    public ArgAssignKind(String fnName, int argIdx) {
      this(fnName, "argument "+(argIdx+1));
    }
    public ArgAssignKind(String fnName, String paramName) {
      super(paramName+" to "
            +(fnName == null ? "pointer" : "\""+fnName+"\""));
    }
  }

  /**
   * Convert to another type for the sake of some kind of assignment. The
   * possible kinds of assignment are enumerated in {@link AssignKind}.
   * 
   * <p>
   * This should not be the result of a {@link #prepareForOp} call as that
   * will be called here if appropriate.
   * </p>
   * 
   * @param toType
   *          the destination type
   * @param assignKind
   *          the kind of assignment
   * @param llvmModule
   *          the LLVM module into which any required declarations should be
   *          generated for the conversion
   * @param llvmBuilder
   *          the IR builder with which any LLVM instructions should be
   *          generated for the conversion
   * @param warningsAsErrors
   *          whether to report warnings as errors
   * @return the new value, which might be exactly the return of
   *         {@link #getLLVMValue} if no LLVM operations were required to
   *         perform the conversion
   * @throws SrcRuntimeException
   *           if the conversion is not valid for the specified kind of
   *           assignment in C, as (partially) specified in ISO C99 6.5.16.1
   */
  public LLVMValue convertForAssign(SrcType toType, AssignKind assignKind,
                                    LLVMModule llvmModule,
                                    LLVMInstructionBuilder llvmBuilder,
                                    boolean warningsAsErrors)
  {
    // ISO C99 sec. 6.3.2.1p3.
    final ValueAndType fromPrep
      = isStringLiteral() && toType.iso(SrcArrayType.class) 
        ? this : prepareForOp(llvmModule, llvmBuilder);

    // ISO C99 sec. 6.5.16.1 says that, when (implicitly) converting among
    // pointer types, either the target types must be compatible, or one must
    // be void and other must not be a function type.
    // (Explicit pointer casts appear to be unconstrained, according to ISO C99
    // sec. 6.5.4p3, so these checks belong here but not in
    // SrcType#convertFromNoPrep.)
    // Many compilers, such as gcc, treat failures of these checks as
    // warnings by default, so we do too. For NVL-C, some such conversions
    // become errors when convertFromNoPrep is called below.
    final SrcPointerType fromPtrType
      = fromPrep.srcType.toIso(SrcPointerType.class);
    final SrcPointerType toPtrType
      = toType.toIso(SrcPointerType.class);
    if (fromPtrType != null && toPtrType != null) {
      final SrcType fromTargetType = fromPtrType.getTargetType();
      final SrcType toTargetType = toPtrType.getTargetType();
      final boolean fromTargetTypeIsVoid = fromTargetType.iso(SrcVoidType);
      final boolean toTargetTypeIsVoid = toTargetType.iso(SrcVoidType);
      if (fromTargetTypeIsVoid != toTargetTypeIsVoid) {
        final SrcType nonVoidTargetType
          = fromTargetTypeIsVoid ? fromTargetType : toTargetType;
        if (nonVoidTargetType.iso(SrcFunctionType.class))
          BuildLLVM.warn(warningsAsErrors,
            assignKind + " requires conversion between void pointer type and"
            + " function pointer type without explicit cast");
      }
      else if (!fromTargetTypeIsVoid && ! toTargetTypeIsVoid)
        toTargetType.toIso(SrcBaldType.class).checkCompatibility(
          fromTargetType.toIso(SrcBaldType.class), true, warningsAsErrors,
          assignKind+" requires conversion between pointer types with"
          +" incompatible target types without explicit cast");
      final EnumSet<SrcTypeQualifier>
        toTargetQuals = toTargetType.expandSrcTypeQualifiers(),
        fromTargetQuals = fromTargetType.expandSrcTypeQualifiers();
      if (!toTargetQuals.containsAll(fromTargetQuals)) {
        final StringBuilder diff = new StringBuilder();
        for (SrcTypeQualifier qual : fromTargetQuals) {
          if (!toTargetQuals.contains(qual)) {
            if (diff.length() > 0)
              diff.append(", ");
            diff.append(qual.name().toLowerCase());
          }
        }
        BuildLLVM.warn(warningsAsErrors,
          assignKind + " discards type qualifiers from pointer target type: "
          + diff.toString());
      }
    }

    // ISO C99 sec. 6.5.16.1 does not permit implicit conversion between
    // pointers and arithmetic types except in two cases: from 0 to pointer,
    // and from pointer to _Bool.
    if ((fromPrep.srcType.iso(SrcArithmeticType.class)
         && toPtrType != null
         && !fromPrep.isNullPointerConstant())
        || (fromPtrType != null
            && toType.iso(SrcArithmeticType.class)
            && !toType.iso(SrcBoolType)))
      BuildLLVM.warn(warningsAsErrors,
        assignKind + " requires conversion between pointer type and"
        + " arithmetic type without explicit cast");

    // Assigning or initializing void never makes sense (but is permitted by
    // explicit casts and so is permitted by SrcType#convertFromNoPrep).
    assert(!toType.iso(SrcVoidType));

    final LLVMValue res = toType.convertFromNoPrep(
      fromPrep, assignKind.toString(), llvmModule, llvmBuilder);
    assert(res != null);
    return res;
  }

  /**
   * A convenient wrapper around {@link SrcType#convertFromNoPrep}. There
   * are two differences. First, this and the first parameter are reversed.
   * Second, the result is wrapped in a {@link ValueAndType}, which is always
   * an rvalue and, of course, always has the specified {@code toType}.
   */
  public ValueAndType convertToNoPrep(SrcType toType, String operation,
                                      LLVMModule llvmModule,
                                      LLVMInstructionBuilder llvmBuilder)
  {
    return new ValueAndType(toType.convertFromNoPrep(this, operation,
                                                     llvmModule, llvmBuilder),
                            toType, false);
  }

  /**
   * Evaluate this as a scalar condition for constructs like "{@code &&}", "
   * {@code ||}", "{@code if}", "{@code for}", etc.
   * 
   * <p>
   * This should not be the result of a {@link #prepareForOp} call as that
   * will be called here if appropriate.
   * </p>
   * 
   * @param operand
   *          identification of the operand and the construct for which the
   *          operand is being evaluated as a condition. For example, "
   *          {@code first operand to "&&"}".
   * @param llvmModule
   *          the LLVM module into which any required declarations should be
   *          generated
   * @param llvmBuilder
   *          the IR builder with which any LLVM instructions should be
   *          generated
   * @return an i1 whose value is {@code 0} if this has a zero/null-pointer
   *         value, or {@code 1} if it has a non-zero/non-null-pointer value
   * @throws SrcRuntimeException
   *           if this is not of scalar type
   */
  public LLVMValue evalAsCond(String operand, LLVMModule llvmModule,
                              LLVMInstructionBuilder llvmBuilder)
  {
    final ValueAndType operandPrep = prepareForOp(llvmModule, llvmBuilder);
    return operandPrep.getSrcType().evalAsCondNoPrep(
      operand, operandPrep.getLLVMValue(), llvmModule, llvmBuilder);
  }

  /**
   * Perform C's address operator ("{@code &}") on this.
   * 
   * <p>
   * This should not be the result of a {@link #prepareForOp} call as that
   * will be called here if appropriate.
   * </p>
   * 
   * <p>
   * This will not generate instructions. That way, this is always correctly
   * folded with "{@code *}" and "{@code []}" as specified in ISO C99 sec.
   * 6.5.3.2p3.
   * </p>
   * 
   * @param llvmModule
   *          the LLVM module into which any required declarations should be
   *          generated
   * @param llvmBuilder
   *          the IR builder with which any LLVM instructions should be
   *          generated
   * @return the new rvalue and type
   * @throws SrcRuntimeException
   *           if this is not an lvalue
   */
  public ValueAndType address(LLVMModule llvmModule,
                              LLVMInstructionBuilder llvmBuilder)
  {
    return srcType.address(llvmValue, lvalueOrFnDesignator, llvmModule,
                           llvmBuilder);
  }

  /**
   * Perform C's indirection operator ("{@code *}") on this.
   * 
   * <p>
   * This should not be the result of a {@link #prepareForOp} call as that
   * will be called here if appropriate.
   * </p>
   * 
   * <p>
   * This will not generate instructions other than what is required by
   * {@link #prepareForOp} (such as a load). That way, this is always
   * correctly folded with "{@code &}" as specified in ISO C99 sec. 6.5.3.2p3.
   * </p>
   * 
   * @param operand
   *          identification of the operator and operand for which the
   *          indirection is being performed. For example,
   *          "{@code operand to unary "*"}" or
   *          "{@code subscripted expression}".
   * @param llvmModule
   *          the LLVM module into which any required declarations should be
   *          generated
   * @param llvmBuilder
   *          the IR builder with which any LLVM instructions should be
   *          generated
   * @return the new lvalue and type
   * @throws SrcRuntimeException
   *           if this is not of pointer type
   */
  public ValueAndType indirection(String operand, LLVMModule llvmModule,
                                  LLVMInstructionBuilder llvmBuilder)
  {
    final ValueAndType operandPrep = prepareForOp(llvmModule, llvmBuilder);
    return operandPrep.getSrcType()
           .indirectionNoPrep(operand, operandPrep.getLLVMValue(), llvmModule,
                              llvmBuilder);
  }

  /**
   * Perform an explicit C cast to another type.
   * 
   * <p>
   * This should not be the result of a {@link #prepareForOp} call as that
   * will be called here if appropriate.
   * </p>
   * 
   * @param toType
   *          the destination type
   * @param llvmModule
   *          the LLVM module into which any required declarations should be
   *          generated for the conversion
   * @param llvmBuilder
   *          the IR builder with which any LLVM instructions should be
   *          generated for the conversion
   * @return the new rvalue and type
   * @throws SrcRuntimeException
   *           if the conversion is not valid in C
   */
  public ValueAndType explicitCast(SrcType toType, LLVMModule llvmModule,
                                   LLVMInstructionBuilder llvmBuilder)
  {
    final ValueAndType fromPrep = prepareForOp(llvmModule, llvmBuilder);
    if (!toType.iso(SrcVoidType)) {
      if (!toType.iso(SrcScalarType.class))
        throw new SrcRuntimeException(
          "explicit cast's destination type is not of scalar type");
      if (!fromPrep.srcType.iso(SrcScalarType.class))
        throw new SrcRuntimeException(
          "explicit cast's expression is not of scalar type");
    }
    // The above checks are from ISO C99 sec. 6.5.4p2, and so they do not
    // permit explicit casts to perform some conversions that would otherwise
    // be performed by the SrcType#convertFrom call below (because they are
    // permitted for assignment and initialization): assigning a struct/union
    // to a struct/union or assigning a string literal to an array.
    final LLVMValue res = toType.convertFromNoPrep(fromPrep, "explicit cast",
                                                   llvmModule, llvmBuilder);
    return new ValueAndType(res, toType, false);
  }

  /**
   * Perform C's unary arithmetic "{@code +}" operator on this.
   * 
   * <p>
   * This should not be the result of a {@link #prepareForOp} call as that
   * will be called here if appropriate.
   * </p>
   * 
   * @param llvmModule
   *          the LLVM module into which any required declarations should be
   *          generated for the conversion
   * @param llvmBuilder
   *          the IR builder with which any LLVM instructions should be
   *          generated
   * @return the new rvalue and type
   * @throws SrcRuntimeException
   *           if this is not of arithmetic type
   */
  public ValueAndType unaryPlus(LLVMModule llvmModule,
                                LLVMInstructionBuilder llvmBuilder)
  {
    final ValueAndType operandPrep = prepareForOp(llvmModule, llvmBuilder);
    return operandPrep.getSrcType().unaryPlusNoPrep(
      operandPrep.getLLVMValue(), llvmModule, llvmBuilder);
  }

  /**
   * Perform C's unary arithmetic "{@code -}" operator on this.
   * 
   * <p>
   * This should not be the result of a {@link #prepareForOp} call as that
   * will be called here if appropriate.
   * </p>
   * 
   * @param llvmModule
   *          the LLVM module into which any required declarations should be
   *          generated for the conversion
   * @param llvmBuilder
   *          the IR builder with which any LLVM instructions should be
   *          generated
   * @return the new rvalue and type
   * @throws SrcRuntimeException
   *           if this is not of arithmetic type
   */
  public ValueAndType unaryMinus(LLVMModule llvmModule,
                                 LLVMInstructionBuilder llvmBuilder)
  {
    final ValueAndType operandPrep = prepareForOp(llvmModule, llvmBuilder);
    return operandPrep.getSrcType().unaryMinusNoPrep(
      operandPrep.getLLVMValue(), llvmModule, llvmBuilder);
  }

  /**
   * Perform C's unary arithmetic "{@code ~}" operator on this.
   * 
   * <p>
   * This should not be the result of a {@link #prepareForOp} call as that
   * will be called here if appropriate.
   * </p>
   * 
   * @param llvmModule
   *          the LLVM module into which any required declarations should be
   *          generated for the conversion
   * @param llvmBuilder
   *          the IR builder with which any LLVM instructions should be
   *          generated
   * @return the new rvalue and type
   * @throws SrcRuntimeException
   *           if this is not of integer type
   */
  public ValueAndType unaryBitwiseComplement(LLVMModule llvmModule,
                                             LLVMInstructionBuilder llvmBuilder)
  {
    final ValueAndType operandPrep = prepareForOp(llvmModule, llvmBuilder);
    return operandPrep.getSrcType().unaryBitwiseComplementNoPrep(
      operandPrep.getLLVMValue(), llvmModule, llvmBuilder);
  }

  /**
   * Perform C's unary "{@code !}" operator on this.
   * 
   * <p>
   * This should not be the result of a {@link #prepareForOp} call as that
   * will be called here if appropriate.
   * </p>
   * 
   * @param llvmModule
   *          the LLVM module into which any required declarations should be
   *          generated for the conversion
   * @param llvmBuilder
   *          the IR builder with which any LLVM instructions should be
   *          generated
   * @return the new rvalue and type
   * @throws SrcRuntimeException
   *           if this is not of scalar type
   */
  public ValueAndType unaryNot(LLVMModule llvmModule,
                               LLVMInstructionBuilder llvmBuilder)
  {
    final ValueAndType operandPrep = prepareForOp(llvmModule, llvmBuilder);
    return operandPrep.getSrcType().unaryNotNoPrep(
      operandPrep.getLLVMValue(), llvmModule, llvmBuilder);
  }

  /**
   * Perform C's binary "{@code *}" operator.
   * 
   * @param op1
   *          the first operand, which should not be the result of a
   *          {@link #prepareForOp} call as that will be called here if
   *          appropriate
   * @param op2
   *          the second operand, which should not be the result of a
   *          {@link #prepareForOp} call as that will be called here if
   *          appropriate
   * @param llvmModule
   *          the LLVM module into which any required declarations should be
   *          generated
   * @param llvmBuilder
   *          the IR builder with which any LLVM instructions should be
   *          generated
   * @return the new rvalue and type
   * @throws SrcRuntimeException
   *           if {@code op1} and {@code op2} are not of valid types for the
   *           operation
   */
  public static ValueAndType multiply(
    ValueAndType op1, ValueAndType op2,
    LLVMModule llvmModule, LLVMInstructionBuilder llvmBuilder)
  {
    final ValueAndType op1Prep = op1.prepareForOp(llvmModule, llvmBuilder);
    final ValueAndType op2Prep = op2.prepareForOp(llvmModule, llvmBuilder);
    return SrcType.multiplyNoPrep(op1Prep, op2Prep, llvmModule, llvmBuilder);
  }

  /**
   * Same as {@link #multiply} except perform C's binary "{@code /}" operator.
   */
  public static ValueAndType divide(
    ValueAndType op1, ValueAndType op2,
    LLVMModule llvmModule, LLVMInstructionBuilder llvmBuilder)
  {
    final ValueAndType op1Prep = op1.prepareForOp(llvmModule, llvmBuilder);
    final ValueAndType op2Prep = op2.prepareForOp(llvmModule, llvmBuilder);
    return SrcType.divideNoPrep(op1Prep, op2Prep, llvmModule, llvmBuilder);
  }

  /**
   * Same as {@link #multiply} except perform C's binary "{@code %}" operator.
   */
  public static ValueAndType remainder(
    ValueAndType op1, ValueAndType op2,
    LLVMModule llvmModule, LLVMInstructionBuilder llvmBuilder)
  {
    final ValueAndType op1Prep = op1.prepareForOp(llvmModule, llvmBuilder);
    final ValueAndType op2Prep = op2.prepareForOp(llvmModule, llvmBuilder);
    return SrcType.remainderNoPrep(op1Prep, op2Prep, llvmModule, llvmBuilder);
  }

  /**
   * Same as {@link #multiply} except perform C's binary "{@code +}" operator.
   * {@code operator} identifies the operator for which the addition is being
   * performed. For example, "{@code binary "+"}" or
   * "{@code subscript operator}".
   */
  public static ValueAndType add(
    String operator, ValueAndType op1, ValueAndType op2,
    LLVMModule llvmModule, LLVMInstructionBuilder llvmBuilder)
  {
    final ValueAndType op1Prep = op1.prepareForOp(llvmModule, llvmBuilder);
    final ValueAndType op2Prep = op2.prepareForOp(llvmModule, llvmBuilder);
    return SrcType.addNoPrep(operator, op1Prep, op2Prep, llvmModule,
                             llvmBuilder);
  }

  /**
   * Same as {@link #multiply} except perform C's binary "{@code -}" operator.
   * {@code operator} identifies the operator for which the subtraction is
   * being performed. For example, "{@code binary "-"}" or "{@code "--"}".
   */
  public static ValueAndType subtract(
    String operator, ValueAndType op1, ValueAndType op2,
    LLVMTargetData llvmTargetData, LLVMModule llvmModule,
    LLVMInstructionBuilder llvmBuilder)
  {
    final ValueAndType op1Prep = op1.prepareForOp(llvmModule, llvmBuilder);
    final ValueAndType op2Prep = op2.prepareForOp(llvmModule, llvmBuilder);
    return SrcType.subtractNoPrep(operator, op1Prep, op2Prep, llvmTargetData,
                                  llvmModule, llvmBuilder);
  }

  /**
   * Same as {@link #multiply} except perform C's binary:
   * <ul>
   *   <li>"{@code <<}" operator if {@code right} is false</li>
   *   <li>"{@code >>}" operator if {@code right} is true</li>
   * </ul>
   */
  public static ValueAndType shift(
    ValueAndType op1, ValueAndType op2, boolean right,
    LLVMModule llvmModule, LLVMInstructionBuilder llvmBuilder)
  {
    final ValueAndType op1Prep = op1.prepareForOp(llvmModule, llvmBuilder);
    final ValueAndType op2Prep = op2.prepareForOp(llvmModule, llvmBuilder);
    return SrcType.shiftNoPrep(op1Prep, op2Prep, right, llvmModule,
                               llvmBuilder);
  }

  /**
   * Same as {@link #multiply} except perform C's binary:
   * <ul>
   *   <li>
   *     "{@code <}" operator if {@code greater} is false and {@code equals}
   *     is false
   *   </li>
   *   <li>
   *     "{@code >}" operator if {@code greater} is true and {@code equals}
   *     is false
   *   </li>
   *   <li>
   *     "{@code <=}" operator if {@code greater} is false and {@code equals}
   *     is true
   *   </li>
   *   <li>
   *     "{@code >=}" operator if {@code greater} is true and {@code equals}
   *     is true
   *   </li>
   * </ul>
   */
  public static ValueAndType relational(
    ValueAndType op1, ValueAndType op2, boolean greater, boolean equals,
    LLVMModule llvmModule, LLVMInstructionBuilder llvmBuilder)
  {
    final ValueAndType op1Prep = op1.prepareForOp(llvmModule, llvmBuilder);
    final ValueAndType op2Prep = op2.prepareForOp(llvmModule, llvmBuilder);
    return SrcType.relationalNoPrep(op1Prep, op2Prep, greater, equals,
                                    llvmModule, llvmBuilder);
  }

  /**
   * Same as {@link #multiply} except perform C's binary:
   * <ul>
   *   <li>
   *     "{@code ==}" operator if {@code equals} is true
   *   </li>
   *   <li>
   *     "{@code !=}" operator if {@code equals} is false
   *   </li>
   * </ul>
   */
  public static ValueAndType equality(
    ValueAndType op1, ValueAndType op2, boolean equals,
    LLVMModule llvmModule, LLVMInstructionBuilder llvmBuilder)
  {
    final ValueAndType op1Prep = op1.prepareForOp(llvmModule, llvmBuilder);
    final ValueAndType op2Prep = op2.prepareForOp(llvmModule, llvmBuilder);
    return SrcType.equalityNoPrep(op1Prep, op2Prep, equals,
                                  llvmModule, llvmBuilder);
  }

  /**
   * Same as {@link #multiply} except perform C's binary "{@code &}" operator.
   */
  public static ValueAndType andXorOr(
    ValueAndType op1, ValueAndType op2, AndXorOr kind,
    LLVMModule llvmModule, LLVMInstructionBuilder llvmBuilder)
  {
    final ValueAndType op1Prep = op1.prepareForOp(llvmModule, llvmBuilder);
    final ValueAndType op2Prep = op2.prepareForOp(llvmModule, llvmBuilder);
    return SrcType.andXorOrNoPrep(op1Prep, op2Prep, kind,
                                  llvmModule, llvmBuilder);
  }

  /**
   * Same as {@link #multiply} except perform C's array subscripting
   * ("{@code []}") operator.
   */
  public static ValueAndType arraySubscript(
    ValueAndType op1, ValueAndType op2,
    LLVMModule llvmModule, LLVMInstructionBuilder llvmBuilder)
  {
    final ValueAndType op1Prep = op1.prepareForOp(llvmModule, llvmBuilder);
    final ValueAndType op2Prep = op2.prepareForOp(llvmModule, llvmBuilder);
    return SrcType.arraySubscriptNoPrep(op1Prep, op2Prep, llvmModule,
                                        llvmBuilder);
  }

  /**
   * Same as {@link #multiply} except perform C's "{@code =}" operator.
   * {@code operator} identifies the operator for which the assignment is
   * being performed. For example, "{@code "--"}". {@code binary} identifies
   * whether that operator is a binary operator and thus whether {@code op2}
   * is a constant the user did not specify.
   */
  public static ValueAndType simpleAssign(
    String operator, boolean binary, ValueAndType op1, ValueAndType op2,
    LLVMModule llvmModule, LLVMInstructionBuilder llvmBuilder,
    boolean warningsAsErrors)
  {
    return SrcType.simpleAssign(operator, binary, op1, op2, llvmModule,
                                llvmBuilder, warningsAsErrors);
  }

  @Override
  public String toString() {
    final StringBuilder str = new StringBuilder();
    str.append("<");
    if (isLvalue())
      str.append("lval");
    else if (lvalueOrFnDesignator)
      str.append("fndsg");
    else
      str.append("rval");
    str.append(", ");
    str.append(srcType);
    str.append(", ");
    str.append(llvmValue);
    str.append(">");
    return str.toString();
  }
}