package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcParamType.SRC_PARAM_METADATA;
import static openacc.codegen.llvmBackend.SrcParamType.SRC_PARAM_TYPE_CHECKSUM_FROM_PTR_ARG;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcCharType;
import static openacc.codegen.llvmBackend.SrcPrimitiveNonNumericType.SrcVoidType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.ListIterator;

import org.jllvm.LLVMBitCast;
import org.jllvm.LLVMCallInstruction;
import org.jllvm.LLVMConstant;
import org.jllvm.LLVMContext;
import org.jllvm.LLVMFunction;
import org.jllvm.LLVMFunctionType;
import org.jllvm.LLVMInstructionBuilder;
import org.jllvm.LLVMIntegerType;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMPointerType;
import org.jllvm.LLVMStackAllocation;
import org.jllvm.LLVMTargetData;
import org.jllvm.LLVMType;
import org.jllvm.LLVMValue;
import org.jllvm.LLVMVariableArgumentInstruction;
import org.jllvm.LLVMVoidType;

import cetus.hir.Traversable;

/**
 * The LLVM backend's class for all function builtins from the C source.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class SrcFunctionBuiltin {
  public enum Instruction {ALLOCA, VA_ARG};
  private final String builtinName;
  /** null if function builtin translates to instruction or constant rvalue */
  private final String targetBaseName;
  /** false if {@link #targetBaseName} is null */
  private final boolean builtinReturnTypeOverload;
  /** null if {@link #targetBaseName} is null */
  private final boolean[] builtinParamTypeOverloads;
  /** null if function builtin translates to function or constant rvalue */
  private final Instruction targetInstruction;
  private final SrcReturnType builtinReturnType;
  private final boolean isVaArg;
  /** size 0 if function builtin translates to a constant rvalue */
  private final SrcParamType[] builtinParamTypes;
  /** null if function builtin translates to a constant rvalue */
  private final String[] builtinParamNames;
  /** null if function builtin translates to a function or instruction */
  private final ValueAndType constantResult;

  /**
   * Create a new function builtin that translates to another function call.
   * 
   * @param builtinName
   *          the function builtin's name in the source code after
   *          preprocessing
   * @param targetBaseName
   *          the name of the target function (possibly an LLVM intrinsic
   *          including the "{@code llvm.}" prefix) to which the builtin
   *          should be translated. Appending to this name is done in the
   *          following order: special return type (such as
   *          {@link #SRC_RET_NVM_PTR_TYPE_FROM_TYPE_ARG}), special parameter
   *          types (such as {@link #SRC_PARAM_MPI_GROUP}) in parameter
   *          order.
   * @param builtinReturnType
   *          the return type for the function builtin
   * @param isVaArg
   *          whether the function builtin and target function are variadic
   * @param builtinParamTypes
   *          the parameter types for the function builtin.
   */
  public SrcFunctionBuiltin(String builtinName, String targetBaseName,
                            SrcReturnType builtinReturnType,
                            boolean isVaArg,
                            SrcParamType... builtinParamTypes)
  {
    this(builtinName, targetBaseName, builtinReturnType, isVaArg,
         builtinParamTypes, false, null, null);
  }

  /**
   * Create a new function builtin that (1) translates to a call to an
   * overloaded LLVM intrinsic, or (2) needs its formal parameters named in
   * error messages (because the builtin represents a pragma), or (3) both.
   * 
   * @param builtinName
   *          the function builtin's name in the source code after
   *          preprocessing
   * @param targetBaseName
   *          the target LLVM intrinsic name without overloaded types
   *          appended. Appending to this name is done in the following
   *          order: special return type (such as
   *          {@link #SRC_RET_NVM_PTR_TYPE_FROM_TYPE_ARG}), special
   *          parameter types (such as {@link #SRC_PARAM_MPI_GROUP}) in
   *          parameter order, {@code builtinReturnTypeOverload}, and
   *          {@code builtinParamTypeOverloads} in parameter order.
   * @param builtinReturnType
   *          the return type for the function builtin
   * @param isVaArg
   *          whether the function builtin and target function are variadic
   * @param builtinParamTypes
   *          the parameter types for the function builtin
   * @param builtinReturnTypeOverload
   *          whether the return type should be appended to
   *          {@code targetBaseName}
   * @param builtinParamTypeOverloads
   *          which parameter types should be appended to
   *          {@code targetBaseName}, or null for none. For pointer types,
   *          the target types are appended instead. If you just want an
   *          intrinsic parameter that accepts any pointer type, instead
   *          make it {@code i8*} (that is, {@code llvm_ptr_ty} in the
   *          {@code .td} file), and specify a {@code void*} type or
   *          something like {@link #SRC_PARAM_MPI_GROUP} in
   *          {@code builtinParamTypes}.
   * @param builtinParamNames
   *          names to use for actual arguments in error messages, or null
   *          if they should be referred to as "argument 1", etc. This is
   *          useful for pragmas whose clauses are validated as an ordered
   *          list of formal parameters to imaginary function builtins even
   *          though those clauses can appear in a different order in the
   *          application source code, thus making "argument 1" meaningless.
   *          If builtinParamNames is not null, isVaArg must be false.
   */
  public SrcFunctionBuiltin(
    String builtinName, String targetBaseName,
    SrcReturnType builtinReturnType, boolean isVaArg,
    SrcParamType[] builtinParamTypes, boolean builtinReturnTypeOverload,
    boolean[] builtinParamTypeOverloads, String[] builtinParamNames)
  {
    this.builtinName = builtinName;
    this.targetBaseName = targetBaseName;
    this.builtinReturnTypeOverload = builtinReturnTypeOverload;
    assert(builtinParamTypeOverloads == null
           || builtinParamTypes.length == builtinParamTypeOverloads.length);
    this.builtinParamTypeOverloads = builtinParamTypeOverloads;
    this.targetInstruction = null;
    this.builtinReturnType = builtinReturnType;
    this.isVaArg = isVaArg;
    this.builtinParamTypes = builtinParamTypes;
    if (builtinParamNames != null) {
      assert(builtinParamTypes.length == builtinParamNames.length);
      this.builtinParamNames = builtinParamNames;
      assert(!isVaArg);
    }
    else {
      this.builtinParamNames = new String[builtinParamTypes.length];
      for (int i = 0; i < builtinParamTypes.length; ++i)
        this.builtinParamNames[i] = "argument "+(i+1);
    }
    this.constantResult = null;
  }

  /**
   * Same as {@link #SrcFunctionBuiltin(String, String, SrcReturnType, boolean, SrcParamType...)}
   * except the target is an LLVM instruction instead of a function, and
   * nothing is appended to the target instruction name regardless of any
   * special return type or parameter types.
   */
  public SrcFunctionBuiltin(
    String builtinName, Instruction targetInstruction,
    SrcReturnType builtinReturnType, boolean isVaArg,
    SrcParamType... builtinParamTypes)
  {
    this.builtinName = builtinName;
    this.targetBaseName = null;
    this.builtinReturnTypeOverload = false;
    this.builtinParamTypeOverloads = null;
    this.targetInstruction = targetInstruction;
    this.builtinReturnType = builtinReturnType;
    this.isVaArg = isVaArg;
    this.builtinParamTypes = builtinParamTypes;
    this.builtinParamNames = new String[builtinParamTypes.length];
    for (int i = 0; i < builtinParamTypes.length; ++i)
      this.builtinParamNames[i] = "argument "+(i+1);
    this.constantResult = null;
  }

  /**
   * Create a new function builtin that takes no parameters and that
   * translates to a constant rvalue.
   * 
   * @param builtinName
   *          the function builtin's name in the source code after
   *          preprocessing
   * @param constantResult
   *          the return value, which must be a constant rvalue
   */
  public SrcFunctionBuiltin(String builtinName, ValueAndType constantResult) {
    assert(!constantResult.isLvalue());
    assert(constantResult.getLLVMValue() instanceof LLVMConstant);
    this.builtinName = builtinName;
    this.targetBaseName = null;
    this.builtinReturnTypeOverload = false;
    this.builtinParamTypeOverloads = null;
    this.targetInstruction = null;
    this.isVaArg = false;
    this.builtinReturnType = constantResult.getSrcType();
    this.builtinParamTypes = new SrcParamType[0];
    this.builtinParamNames = null;
    this.constantResult = constantResult;
  }

  /** Get the function builtin's name. */
  public String getName() {
    return builtinName;
  }

  /**
   * Get the function builtin's parameter types as {@link SrcType}s, omit
   * any discarded parameters, and substitute null where no {@link SrcType}
   * can express the type.
   */
  private SrcType[] getParamTypes(
    Traversable callNode, ValueAndType args[], SrcType typeArg,
    SrcSymbolTable srcSymbolTable, LLVMModule llvmModule,
    boolean warningsAsErrors)
  {
    int paramCount = 0;
    for (int i = 0; i < builtinParamTypes.length; ++i)
      if (!builtinParamTypes[i].discardsArg())
        ++paramCount;
    final SrcType[] paramTypes = new SrcType[paramCount];
    paramCount = 0;
    for (int i = 0; i < builtinParamTypes.length; ++i) {
      if (!builtinParamTypes[i].discardsArg())
        paramTypes[paramCount++] = builtinParamTypes[i].getCallSrcType(
          callNode, builtinName, builtinParamNames[i], args, typeArg,
          srcSymbolTable, llvmModule, warningsAsErrors);
    }
    return paramTypes;
  }

  /**
   * Get the target function's parameter types translated to LLVM types.
   */
  private LLVMType[] getLLVMParamTypes(
    SrcType paramTypes[], SrcType typeArg, LLVMModule llvmModule)
  {
    final LLVMType[] res = new LLVMType[paramTypes.length];
    int targetParamIdx = 0;
    for (int i = 0; i < builtinParamTypes.length; ++i) {
      if (builtinParamTypes[i].discardsArg())
        continue;
      res[targetParamIdx]
        = builtinParamTypes[i].getTargetCallLLVMType(
            paramTypes[targetParamIdx], typeArg, llvmModule);
      ++targetParamIdx;
    }
    return res;
  }

  /** Fetch or declare the target LLVM function for {@code llvmModule}. */
  private LLVMFunction getLLVMFn(
    LLVMType llvmReturnType, LLVMType[] llvmParamTypes,
    boolean llvmReturnTypeOverload, LLVMModule llvmModule)
  {
    assert(targetBaseName != null);
    final StringBuilder str = new StringBuilder(targetBaseName);
    {
      final String suffix = builtinReturnType
                            .getSpecialOverloadSuffix(llvmReturnType);
      if (suffix != null)
        str.append(suffix);
    }
    int targetParamIdx = 0;
    for (int i = 0; i < builtinParamTypes.length; ++i) {
      if (builtinParamTypes[i].discardsArg())
        continue;
      final String suffix = builtinParamTypes[i].getSpecialOverloadSuffix(
        llvmParamTypes[targetParamIdx]);
      if (suffix != null)
        str.append(suffix);
      ++targetParamIdx;
    }
    if (builtinReturnType.overloadsWithLLVMSuffix(
          llvmReturnType, llvmReturnTypeOverload)
        || llvmReturnTypeOverload)
    {
      str.append(".");
      str.append(llvmReturnType.toStringForIntrinsic());
    }
    targetParamIdx = 0;
    for (int i = 0; i < builtinParamTypes.length; ++i) {
      if (builtinParamTypes[i].discardsArg())
        continue;
      boolean overload = builtinParamTypeOverloads == null
                         ? false : builtinParamTypeOverloads[i];
      if (builtinParamTypes[i].overloadsWithLLVMSuffix(
            llvmParamTypes[targetParamIdx], overload))
        overload = true;
      if (overload) {
        str.append(".");
        final LLVMType overloadType;
        if (llvmParamTypes[targetParamIdx] instanceof LLVMPointerType)
          overloadType = ((LLVMPointerType)llvmParamTypes[targetParamIdx])
                         .getElementType();
        else
          overloadType = llvmParamTypes[targetParamIdx];
        str.append(overloadType.toStringForIntrinsic());
      }
      ++targetParamIdx;
    }
    final String targetName = str.toString();
    final LLVMFunction old = llvmModule.getNamedFunction(targetName);
    if (old.getInstance() != null)
      return old;
    return new LLVMFunction(llvmModule, targetName,
                            LLVMFunctionType.get(llvmReturnType, isVaArg,
                                                 llvmParamTypes));
  }

  /**
   * Generate a call to this function builtin.
   * 
   * @param callNode
   *          the AST node for the call. Used as the context for any special
   *          symbol table lookups.
   * @param args
   *          the actual source-level arguments to the call, none of which
   *          should be the result of a {@link #prepareForOp} call as that
   *          will be called here where appropriate. For target-only
   *          parameter types (whether explicit like
   *          {@link #SRC_PARAM_METADATA} or implicit like
   *          {@link #SRC_PARAM_TYPE_CHECKSUM_FROM_PTR_ARG}), no argument
   *          should be specified here.
   * @param targetOnlyArgs
   *          the explicit target-only arguments, as described in
   *          {@code args} documentation
   * @param typeArg
   *          a type argument, or null if none was supplied. A type argument
   *          is expected if and only if the return type (such as
   *          {@link #SRC_RET_TYPE_IS_ARG}) is based on a type argument. For
   *          all function builtins so far, it's a syntax error if whether a
   *          type argument is supplied does not match whether it is
   *          expected, so just fail an assertion here in that case. So far,
   *          we've had no need for multiple type arguments, and any type
   *          argument always appears at the end of the argument list.
   * @param srcSymbolTable
   *          the {@link SrcSymbolTable} for the C source
   * @param llvmModule
   *          the LLVM module into which any required declarations should be
   *          generated
   * @param llvmModuleIndex
   *          index of {@code llvmModule} within the array of modules being
   *          constructed by {@link BuildLLVM}
   * @param llvmTargetData
   *          the LLVMTargetData used while building llvmModule
   * @param llvmBuilder
   *          the IR builder with which any LLVM instructions should be
   *          generated
   * @param warningsAsErrors
   *          whether to report warnings as errors
   * @return the call's rvalue result (void expression if void return type)
   */
  public ValueAndType call(
    Traversable callNode, ValueAndType[] args,
    LLVMValue[] targetOnlyArgs, SrcType typeArg,
    SrcSymbolTable srcSymbolTable, LLVMModule llvmModule,
    int llvmModuleIndex, LLVMTargetData llvmTargetData, 
    LLVMInstructionBuilder llvmBuilder, boolean warningsAsErrors)
  {
    final LLVMContext ctxt = llvmModule.getContext();
    final SrcType returnType = builtinReturnType.getCallSrcType(
      callNode, builtinName, args, typeArg, srcSymbolTable, llvmModule,
      warningsAsErrors);
    final SrcType[] paramTypes = getParamTypes(
      callNode, args, typeArg, srcSymbolTable, llvmModule,
      warningsAsErrors);
    final LLVMType[] llvmParamTypes = getLLVMParamTypes(paramTypes, typeArg,
                                                        llvmModule);
    final LLVMType llvmReturnType = builtinReturnType.getTargetCallLLVMType(
      returnType, typeArg, llvmModule);
    final ArrayList<LLVMValue> llvmArgs = new ArrayList<LLVMValue>();
    final List<ValueAndType> argList = Arrays.asList(args);
    final List<LLVMValue> targetOnlyArgList = Arrays.asList(targetOnlyArgs);
    final ListIterator<ValueAndType> argItr = argList.listIterator();
    final ListIterator<LLVMValue> targetOnlyArgItr = targetOnlyArgList
                                                     .listIterator();
    int paramIdx = 0;
    for (int i = 0; i < builtinParamTypes.length; ++i) {
      final String builtinParamName = builtinParamNames[i];
      final SrcParamType builtinParamType = builtinParamTypes[i];
      try {
        if (builtinParamType.discardsArg()) {
          if (!argItr.hasNext())
            throw new SrcInsufficientArgsException();
          argItr.next();
          continue;
        }
        final SrcType paramType = paramTypes[paramIdx];
        final LLVMType llvmParamType = llvmParamTypes[paramIdx];
        final LLVMValue llvmArg = builtinParamType.processArg(
          builtinName, builtinParamName, returnType, paramType,
          llvmParamType, argItr, targetOnlyArgItr,
          argItr.previousIndex()==-1 ? null : args[argItr.previousIndex()],
          llvmArgs.isEmpty() ? null : llvmArgs.get(llvmArgs.size()-1),
          srcSymbolTable, llvmModule, llvmModuleIndex, llvmBuilder,
          warningsAsErrors);
        assert(llvmArg.typeOf() == llvmParamType);
        llvmArgs.add(llvmArg);
        ++paramIdx;
      }
      catch (SrcInsufficientArgsException e) {
        throw new SrcRuntimeException("too few arguments in call to "
                                      +builtinName);
      }
    }
    if (targetOnlyArgItr.hasNext())
      throw new IllegalStateException();
    if (!isVaArg && argItr.hasNext())
      throw new SrcRuntimeException("too many arguments in call to "
                                    +builtinName);
    while (argItr.hasNext()) {
      final ValueAndType arg = argItr.next();
      final LLVMValue promoted = arg.defaultArgumentPromote(
        "argument "+(argItr.nextIndex())+" to "+builtinName+" is invalid",
        llvmModule, llvmBuilder);
      llvmArgs.add(promoted);
    }
    // See postcondition on getCallSrcType.
    assert(returnType != null);
    final LLVMValue[] llvmArgArr
      = llvmArgs.toArray(new LLVMValue[llvmArgs.size()]);
    final String retName = llvmReturnType == LLVMVoidType.get(ctxt)
                           ? "" : builtinName + ".ret";
    final LLVMValue resultValue;
    if (constantResult != null) {
      assert(llvmArgArr.length == 0);
      return constantResult;
    }
    if (targetInstruction == null) {
      final LLVMFunction llvmFn = getLLVMFn(
        llvmReturnType, llvmParamTypes, builtinReturnTypeOverload,
        llvmModule);
      resultValue = new LLVMCallInstruction(llvmBuilder, retName, llvmFn,
                                            llvmArgArr);
    }
    else if (targetInstruction == Instruction.ALLOCA) {
      final LLVMIntegerType i8 = SrcCharType.getLLVMType(ctxt);
      assert(llvmArgArr.length == 1);
      resultValue = new LLVMStackAllocation(llvmBuilder, retName, i8,
                                            llvmArgArr[0]);
    }
    else if (targetInstruction == Instruction.VA_ARG) {
      assert(llvmArgArr.length == 1);
      resultValue = new LLVMVariableArgumentInstruction(
        llvmBuilder, retName, llvmArgArr[0], returnType.getLLVMType(ctxt));
    }
    else
      throw new IllegalStateException();
    if (returnType.iso(SrcVoidType))
      return new ValueAndType();
    if (resultValue.typeOf() == returnType.getLLVMType(ctxt))
      return new ValueAndType(resultValue, returnType, false);
    return new ValueAndType(
      LLVMBitCast.create(llvmBuilder, retName+".bitcast", resultValue,
                         returnType.getLLVMType(ctxt)),
      returnType, false);
  }

  /**
   * Same as {@link #call(ValueAndType[], LLVMValue[], SrcType, SrcSymbolTable, LLVMModule, int, LLVMTargetData, LLVMInstructionBuilder, boolean)}
   * except {@code targetOnlyArgs} is empty.
   */
  public ValueAndType call(
    Traversable callNode, ValueAndType[] args, SrcType typeArg,
    SrcSymbolTable srcSymbolTable, LLVMModule llvmModule,
    int llvmModuleIndex, LLVMTargetData llvmTargetData,
    LLVMInstructionBuilder llvmBuilder, boolean warningsAsErrors)
  {
    return call(callNode, args, new LLVMValue[0], typeArg, srcSymbolTable,
                llvmModule, llvmModuleIndex, llvmTargetData, llvmBuilder,
                warningsAsErrors);
  }

  /**
   * Same as {@link #call(ValueAndType[], LLVMValue[], SrcType, SrcSymbolTable, LLVMModule, int, LLVMTargetData, LLVMInstructionBuilder, boolean)}
   * except {@code typeArg} is null.
   */
  public ValueAndType call(
    Traversable callNode, ValueAndType[] args,
    LLVMValue[] targetOnlyArgs, SrcSymbolTable srcSymbolTable,
    LLVMModule llvmModule, int llvmModuleIndex,
    LLVMTargetData llvmTargetData, LLVMInstructionBuilder llvmBuilder,
    boolean warningsAsErrors)
  {
    return call(callNode, args, targetOnlyArgs, null, srcSymbolTable,
                llvmModule, llvmModuleIndex, llvmTargetData, llvmBuilder,
                warningsAsErrors);
  }

  /**
   * Same as
   * {@link #call(ValueAndType[], LLVMValue[], SrcType, SrcSymbolTable, LLVMModule, int, LLVMTargetData, LLVMInstructionBuilder, boolean)}
   * except {@code targetOnlyArgs} is empty and {@code typeArg} is null.
   */
  public ValueAndType call(
    Traversable callNode, ValueAndType[] args,
    SrcSymbolTable srcSymbolTable, LLVMModule llvmModule,
    int llvmModuleIndex, LLVMTargetData llvmTargetData,
    LLVMInstructionBuilder llvmBuilder, boolean warningsAsErrors)
  {
    return call(callNode, args, new LLVMValue[0], null, srcSymbolTable,
                llvmModule, llvmModuleIndex, llvmTargetData, llvmBuilder,
                warningsAsErrors);
  }
}
