package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcPrimitiveNonNumericType.SrcVoidType;

import java.lang.ref.WeakReference;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Set;

import openacc.codegen.llvmBackend.SrcQualifiedType.SrcTypeQualifier;
import openacc.codegen.llvmBackend.ValueAndType.ArgAssignKind;

import org.jllvm.LLVMCallInstruction;
import org.jllvm.LLVMContext;
import org.jllvm.LLVMFunctionType;
import org.jllvm.LLVMInstructionBuilder;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMType;
import org.jllvm.LLVMValue;

/**
 * The LLVM backend's class for all function types from the C source.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class SrcFunctionType extends SrcBaldType {
  private static final HashMap<SrcFunctionType, WeakReference<SrcFunctionType>>
    cache = new HashMap<>();

  private final SrcType returnType;
  private final boolean isVarArg;
  private final SrcType[] paramTypes;

  private SrcFunctionType(SrcType returnType, boolean isVarArg,
                          SrcType... paramTypes)
  {
    this.returnType = returnType;
    this.isVarArg = isVarArg;
    this.paramTypes = paramTypes;
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + returnType.hashCode();
    result = prime * result + (isVarArg ? 1231 : 1237);
    result = prime * result + Arrays.hashCode(paramTypes);
    return result;
  }
  @Override
  public boolean equals(Object obj) {
    if (this == obj) return true;
    if (obj == null) return false;
    if (getClass() != obj.getClass()) return false;
    final SrcFunctionType other = (SrcFunctionType)obj;
    if (returnType != other.returnType) return false;
    if (isVarArg != other.isVarArg) return false;
    if (!Arrays.equals(paramTypes, other.paramTypes)) return false;
    return true;
  }

  /**
   * Get the specified function type.
   * 
   * <p>
   * After this method has been called once for a particular type of function,
   * it is guaranteed to return the same object for the same type of function
   * until that object can be garbage-collected because it is no longer
   * referenced outside this class's internal cache.
   * </p>
   * 
   * @param returnType
   *          the function type's return type
   * @param isVarArg
   *          whether the function type is variadic
   * @param paramTypes
   *          the function type's parameter types
   * @return the specified function type
   * @throws SrcRuntimeException
   *           if the resulting type does not satisfy function type
   *           constraints
   */
  public static SrcFunctionType get(SrcType returnType, boolean isVarArg,
                                    SrcType... paramTypes)
  {
    for (int i = 0; i < paramTypes.length; ++i)
      paramTypes[i].checkStaticOrAutoObject("function parameter "+(i+1)
                                            +" type", false);
    if (!returnType.iso(SrcVoidType))
      returnType.checkStaticOrAutoObject("function return type", false);

    final SrcFunctionType cacheKey
      = new SrcFunctionType(returnType, isVarArg, paramTypes);
    WeakReference<SrcFunctionType> ref;
    synchronized (cache) {
      ref = cache.get(cacheKey);
    }
    SrcFunctionType type;
    if (ref == null || (type = ref.get()) == null) {
      type = new SrcFunctionType(returnType, isVarArg, paramTypes);
      ref = new WeakReference<>(type);
      synchronized (cache) {
        cache.put(cacheKey, ref);
      }
    }
    return type;
  }
  @Override
  protected void finalize() {
    synchronized (cache) {
      if (cache.get(this).get() == null)
        cache.remove(this);
    }
  }

  /** Get the function type's return type. */
  public SrcType getReturnType() {
    return returnType;
  }
  /**
   * Is the function type variadic? See {@link #paramsAreUnspecified} for how
   * an unspecified parameter list is encoded.
   */
  public boolean isVarArg() {
    return isVarArg;
  }
  /**
   * Get the function type's parameter types. See
   * {@link #paramsAreUnspecified} for how an unspecified parameter list is
   * encoded.
   */
  public SrcType[] getParamTypes() {
    return paramTypes;
  }
  /**
   * Is the parameter list left unspecified?
   * 
   * <p>
   * This is a shorthand for checking that {@link #isVarArg} returns true
   * while {@link #getParamTypes} returns an empty array (this is modeled
   * after LLVM's encoding of an unspecified parameter list). The function
   * type for a function definition always specifies a (possibly empty)
   * parameter list, so this method always returns false in that case.
   * </p>
   */
  public boolean paramsAreUnspecified() {
    return isVarArg && paramTypes.length == 0;
  }

  /**
   * Generate a call to a function of this type.
   * 
   * @param fnName
   *          the source name of the function, or null if a function
   *          pointer
   * @param fn
   *          the function to call, which must be of this type
   * @param args
   *          the actual arguments to the call, none of which should be the
   *          result of a {@link #prepareForOp} call as that will be called
   *          here where appropriate
   * @param llvmModule
   *          the LLVM module into which any required declarations should be
   *          generated
   * @param llvmBuilder
   *          the IR builder with which any LLVM instructions should be
   *          generated
   * @param warningsAsErrors
   *          whether to report warnings as errors
   * @return the call's rvalue result (void expression if void return type)
   */
  public ValueAndType call(String fnName, LLVMValue fn, ValueAndType[] args,
                           LLVMModule llvmModule,
                           LLVMInstructionBuilder llvmBuilder,
                           boolean warningsAsErrors)
  {
    assert(fn.typeOf()
           == SrcPointerType.get(this).getLLVMType(llvmModule.getContext()));
    final LLVMValue[] argValues = new LLVMValue[args.length];
    if (args.length < paramTypes.length)
      throw new SrcRuntimeException("too few arguments in call to "+fnName);
    if (!isVarArg && paramTypes.length < args.length)
      throw new SrcRuntimeException("too many arguments in call to "
                                    +fnName);
    for (int i = 0; i < args.length; ++i) {
      final ArgAssignKind argAssignKind = new ArgAssignKind(fnName, i);
      if (i < paramTypes.length)
        argValues[i] = args[i].convertForAssign(
          paramTypes[i].toIso(SrcBaldType.class), argAssignKind, llvmModule,
          llvmBuilder, warningsAsErrors);
      else {
        argValues[i] = args[i].defaultArgumentPromote(
          "invalid "+argAssignKind.toString(), llvmModule,
          llvmBuilder);
      }
    }
    final boolean isVoid = returnType.iso(SrcVoidType);
    final LLVMValue resultValue
      = new LLVMCallInstruction(llvmBuilder, isVoid ? "" : ".ret", fn,
                                argValues);
    return new ValueAndType(isVoid ? null : resultValue, returnType, false);
  }

  @Override
  public boolean eqvBald(SrcBaldType other) {
    // Optimize for trivial eqv relation.
    if (this == other)
      return true;
    if (!(other instanceof SrcFunctionType))
      return false;
    final SrcFunctionType otherFunctionType = (SrcFunctionType)other;
    if (!returnType.eqv(otherFunctionType.returnType))
      return false;
    if (isVarArg != otherFunctionType.isVarArg)
      return false;
    if (paramTypes.length != otherFunctionType.paramTypes.length)
      return false;
    for (int i = 0; i < paramTypes.length; ++i)
      if (!paramTypes[i].eqv(otherFunctionType.paramTypes[i]))
        return false;
    return true;
  }

  @Override
  public boolean isIncompleteType() {
    // According to ISO C99 sec. 6.2.5p1, functions are not object types or
    // incomplete types.
    return false;
  }

  @Override
  public boolean hasEffectiveQualifier(SrcTypeQualifier... quals) {
    return false;
  }

  @Override
  public SrcFunctionType withoutEffectiveQualifier(SrcTypeQualifier... quals)
  {
    return this;
  }

  @Override
  public SrcTypeIterator componentIterator(boolean storageOnly,
                                           Set<SrcType> skipTypes)
  {
    if (storageOnly)
      throw new IllegalStateException(
        "storage-only iteration of function type");
    return new SrcTypeComponentIterator(storageOnly, skipTypes,
                                        new SrcType[]{returnType}, paramTypes);
  }

  @Override
  public SrcFunctionType buildCompositeBald(
    SrcBaldType other, boolean warn, boolean warningsAsErrors,
    String msgPrefix)
  {
    // ISO C99 6.7.5.3p15. TODO: We do not correctly handle the case in which
    // a function definition's parameter list is an identifier list with types
    // specified after the closing parenthesis. That is, we incorrectly assume
    // that case is no different than the case where the parameter list is
    // specified with types directly (Cetus appears to distinguish these cases
    // with Procedure.is_old_style_function). Fortunately, I rarely see that
    // old style, and it's obsolecent according to ISO C99 sec. 6.11.7.
    // (According to ISO C99 sec. 6.7.5.3p3, it's not permitted at all on
    // function prototypes without definitions.)
    if (this == other)
      return this;
    final SrcFunctionType otherFn = other.toEqv(SrcFunctionType.class);
    if (otherFn == null) {
      BuildLLVM.warnOrError(warn, warningsAsErrors,
        msgPrefix+": function type is incompatible with non-function type: "
        +other);
      return null;
    }
    final SrcType compositeReturnType = returnType.buildComposite(
      otherFn.returnType, warn, warningsAsErrors,
      msgPrefix+": function types are incompatible because their return"
      +" types are incompatible");
    if (compositeReturnType == null)
      return null;
    final SrcFunctionType res;
    if (!paramsAreUnspecified() && !otherFn.paramsAreUnspecified()) {
      if (paramTypes.length != otherFn.paramTypes.length) {
        BuildLLVM.warnOrError(warn, warningsAsErrors,
          msgPrefix+": function types are incompatible because they specify"
          +" a different number of parameters");
        return null;
      }
      if (isVarArg != otherFn.isVarArg) {
        BuildLLVM.warnOrError(warn, warningsAsErrors,
          msgPrefix+": function types are incompatible because only one is"
          +" variadic");
        return null;
      }
      final SrcType[] newParamTypes = new SrcType[paramTypes.length];
      for (int i = 0; i < paramTypes.length; ++i) {
        newParamTypes[i]
          = paramTypes[i].toIso(SrcBaldType.class).buildComposite(
              otherFn.paramTypes[i].toIso(SrcBaldType.class),
              warn, warningsAsErrors,
              msgPrefix+": function types are incompatible because"
              +" parameter "+(i+1)+" type is incompatible");
        if (newParamTypes[i] == null)
          return null;
      }
      res = SrcFunctionType.get(compositeReturnType, isVarArg,
                                newParamTypes);
    }
    else if (paramsAreUnspecified() != otherFn.paramsAreUnspecified()) {
      final SrcFunctionType fnTypeWithParams
        = !paramsAreUnspecified() ? this : otherFn;
      if (fnTypeWithParams.isVarArg()) {
        BuildLLVM.warnOrError(warn, warningsAsErrors,
          msgPrefix+": function types are incompatible because one has an"
          +" unspecified parameter list while the other is variadic");
        return null;
      }
      for (int i = 0; i < fnTypeWithParams.paramTypes.length; ++i) {
        final SrcBaldType baldParamType
          = fnTypeWithParams.paramTypes[i].toIso(SrcBaldType.class);
        final String msg
          = msgPrefix+": function types are incompatible because one has an"
            +" unspecified parameter list while the other's parameter "
            +(i+1)+" type is not compatible with itself after default"
            +" argument promotions";
        final SrcType promoted = baldParamType.defaultArgumentPromote(
          warn, warningsAsErrors, msg);
        if (promoted == null)
          return null;
        if (!promoted.checkCompatibility(baldParamType, warn,
                                         warningsAsErrors, msg))
          return null;
      }
      res = SrcFunctionType.get(compositeReturnType, fnTypeWithParams.isVarArg,
                                fnTypeWithParams.paramTypes);
    }
    else
      res = SrcFunctionType.get(compositeReturnType, isVarArg, paramTypes);
    return res.eqv(this) ? this : res.eqv(otherFn) ? otherFn : res;
  }

  /**
   * Same as {@link #buildCompositeBald(SrcType, String, boolean, boolean)}
   * except {@code warn} is always false.
   */
  public SrcFunctionType buildCompositeBald(SrcBaldType other,
                                            String msgPrefix)
  {
    return buildCompositeBald(other, false, false, msgPrefix);
  }

  @Override
  public LLVMFunctionType getLLVMType(LLVMContext context) {
    final LLVMType[] paramLLVMTypes = new LLVMType[paramTypes.length];
    for (int i = 0; i < paramTypes.length; ++i)
      paramLLVMTypes[i] = paramTypes[i].getLLVMType(context);
    return LLVMFunctionType.get(returnType.getLLVMType(context), isVarArg,
                                paramLLVMTypes);
  }
  @Override
  public LLVMFunctionType getLLVMTypeAsPointerTarget(LLVMContext context) {
    return getLLVMType(context);
  }

  @Override
  public SrcType prepareForOp(EnumSet<SrcTypeQualifier> srcTypeQualifiers) {
    // Type qualifiers on function type are not permitted.
    assert(srcTypeQualifiers.isEmpty());
    return SrcPointerType.get(this);
  }
  @Override
  public ValueAndType prepareForOp(
    EnumSet<SrcTypeQualifier> srcTypeQualifiers, LLVMValue value,
    boolean lvalueOrFnDesignator, LLVMModule module,
    LLVMInstructionBuilder builder)
  {
    // As mentioned in ValueAndType's constructor's preconditions,
    // lvalueOrFnDesignator=false is impossible when the srcType is a
    // SrcFunctionType.
    assert(lvalueOrFnDesignator);
    // ISO C99 sec. 6.3.2.1p4.
    return new ValueAndType(value, prepareForOp(), false);
  }

  @Override
  public SrcType defaultArgumentPromoteNoPrep() {
    // prepareForOp has been called and it never returns a function.
    throw new IllegalStateException();
  }
  @Override
  public LLVMValue defaultArgumentPromoteNoPrep(LLVMValue value,
                                                LLVMModule module,
                                                LLVMInstructionBuilder builder)
  {
    // prepareForOp has been called and it never returns a function.
    throw new IllegalStateException();
  }

  @Override
  public LLVMValue convertFromNoPrep(ValueAndType from, String operation,
                                     LLVMModule module,
                                     LLVMInstructionBuilder builder)
  {
    throw new SrcRuntimeException(operation
                                  + " requires conversion to function type");
  }

  @Override
  public String toString(String nestedDecl, Set<SrcType> skipTypes) {
    StringBuilder str = new StringBuilder(nestedDecl);
    str.append("(");
    for (int i = 0; i < paramTypes.length; ++i) {
      if (i > 0)
        str.append(", ");
      str.append(paramTypes[i].toString("", skipTypes));
    }
    if (paramTypes.length > 0 && isVarArg)
      str.append(", ...");
    else if (paramTypes.length == 0 && !isVarArg)
      str.append("void");
    str.append(")");
    return returnType.toString(str.toString(), skipTypes);
  }

  @Override
  public String toCompatibilityString(Set<SrcStructOrUnionType> structSet,
                                      LLVMContext ctxt)
  {
    throw new IllegalStateException(
      "toCompatibilityString called for function");
  }
}
