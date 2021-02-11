package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcCharType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcIntType;
import static openacc.codegen.llvmBackend.SrcPrimitiveNonNumericType.SrcVoidType;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import openacc.codegen.llvmBackend.SrcQualifiedType.SrcTypeQualifier;
import openacc.codegen.llvmBackend.ValueAndType.ArgAssignKind;

import org.jllvm.LLVMBitCast;
import org.jllvm.LLVMConstant;
import org.jllvm.LLVMConstantInteger;
import org.jllvm.LLVMContext;
import org.jllvm.LLVMInstructionBuilder;
import org.jllvm.LLVMIntegerComparison;
import org.jllvm.LLVMIntegerType;
import org.jllvm.LLVMMDNode;
import org.jllvm.LLVMMDString;
import org.jllvm.LLVMMetadataType;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMPointerType;
import org.jllvm.LLVMType;
import org.jllvm.LLVMValue;
import org.jllvm.bindings.LLVMIntPredicate;

import cetus.hir.Specifier;
import cetus.hir.SymbolTools;
import cetus.hir.Traversable;
import cetus.hir.UserSpecifier;

/**
 * The LLVM backend's interface for source types that can serve as parameter
 * types.
 * 
 * <p>
 * WARNING: See {@link SrcParamOrReturnType} header comments before
 * extending.
 * </p>
 * 
 * <p>
 * Currently, this is used solely for function builtins. See
 * {@link SrcFunctionBuiltin}.
 * </p>
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public interface SrcParamType extends SrcParamOrReturnType {
  /**
   * For a call where this type is a parameter type, should the actual
   * argument be discarded without examination?
   * 
   * <p>
   * The default implementation returns false.
   * </p>
   * 
   * @return true iff the the actual argument should be discarded without
   *         examination
   */
  public boolean discardsArg();

  /**
   * For a call where this type is a parameter type, process the next
   * corresponding actual source-level argument or the next corresponding
   * actual explicit target-only argument.
   * 
   * <p>
   * The default implementation throws a
   * {@link SrcInsufficientArgsException} if {@code argItr.hasNext()} is
   * false. Otherwise, it calls {@code argItr.next()}, calls
   * {@link ValueAndType#convertForAssign} to convert the result to
   * {@code callSrcType}, bitcasts the result to {@code targetCallLLVMType}
   * if that's not already the type, and then returns the result.
   * </p>
   * 
   * @param calleeName
   *          name to use for the callee in error messages. Usually it's
   *          something like "foo" or "foo pragma".
   * @param paramName
   *          name to use for the actual argument in error messages, or null
   *          if the type is being used as a return type. Usually it's
   *          something like "argument 2" or "foo clause".
   * @param callReturnSrcType
   *          the source-level return type for this call
   * @param callSrcType
   *          the type previously returned by {@link #getCallSrcType} for
   *          this call
   * @param targetCallLLVMType
   *          the type previously returned by {@link #getTargetCallLLVMType}
   *          for this call
   * @param argItr
   *          iterator for which the next item is the call's next actual
   *          source-level argument, if any. Advanced by one iff this is an
   *          explicit source-level parameter type. Otherwise, the cursor
   *          position is left where it started. No item in this list should
   *          be the result of a {@link #prepareForOp} call as that will be
   *          called here if appropriate.
   * @param targetOnlyArgItr
   *          iterator for which the next item is the call's next actual
   *          explicit target-only argument, if any. Advanced by one iff
   *          this is an explicit target-only parameter type. Otherwise, the
   *          cursor position is left where it started.
   * @param prevArg
   *          the previous actual source-level argument, or null if none
   * @param prevArgLLVM
   *          the previous actual LLVM-level argument, or null if none.
   *          {@code prevArg} and {@code prevArgLLVM} are not guaranteed to
   *          represent the same actual argument (and thus just one of them
   *          might be null) when there are discarded arguments or implicit
   *          target-only arguments.
   * @param srcSymbolTable
   *          the {@link SrcSymbolTable} for the C source
   * @param llvmModule
   *          the LLVM module into which any required declarations should be
   *          generated
   * @param llvmModuleIndex
   *          index of {@code llvmModule} within the array of modules being
   *          constructed by {@link BuildLLVM}
   * @param llvmBuilder
   *          the IR builder with which any LLVM instructions should be
   *          generated
   * @param warningsAsErrors
   *          whether to report warnings as errors
   * @return the argument to pass to the target LLVM call. The result is
   *         computed without advancing {@code argItr} or
   *         {@code targetOnlyArgItr} iff this is an implicit target-only
   *         parameter type.
   * @throws SrcInsufficientArgsException
   *           if this is an explicit source-level parameter type but
   *           {@code argItr.hasNext()} is false
   * @throws IllegalStateException
   *           if this is an explicit target-only parameter type but
   *           {@code targetOnlyArgItr.hasNext()} is false
   * @throws IllegalStateException
   *           if {@link #discardsArg} returns true
   */
  public LLVMValue processArg(
    String calleeName, String paramName, SrcType callReturnSrcType,
    SrcType callSrcType, LLVMType targetCallLLVMType,
    Iterator<ValueAndType> argItr, Iterator<LLVMValue> targetOnlyArgItr,
    ValueAndType prevArg, LLVMValue prevArgLLVM,
    SrcSymbolTable srcSymbolTable, LLVMModule llvmModule,
    int llvmModuleIndex, LLVMInstructionBuilder llvmBuilder,
    boolean warningsAsErrors)
    throws SrcInsufficientArgsException;

  /**
   * The implementation for {@link SrcParamType}. See header comments there
   * for details.
   */
  public static abstract class Impl extends SrcParamOrReturnType.Impl
                                    implements SrcParamType
  {
    @Override
    public boolean discardsArg() {
      return false;
    }
    /**
     * Keep this implementation in sync with
     * {@link SrcParamAndReturnType.Impl#processArg}'s implementation.
     */
    @Override
    public LLVMValue processArg(
      String calleeName, String paramName, SrcType callReturnSrcType,
      SrcType callSrcType, LLVMType targetCallLLVMType,
      Iterator<ValueAndType> argItr, Iterator<LLVMValue> targetOnlyArgItr,
      ValueAndType prevArg, LLVMValue prevArgLLVM,
      SrcSymbolTable srcSymbolTable, LLVMModule llvmModule,
      int llvmModuleIndex, LLVMInstructionBuilder llvmBuilder,
      boolean warningsAsErrors)
      throws SrcInsufficientArgsException
    {
      if (!argItr.hasNext())
        throw new SrcInsufficientArgsException();
      final LLVMValue converted = argItr.next().convertForAssign(
        callSrcType.toIso(SrcBaldType.class),
        new ArgAssignKind(calleeName, paramName), llvmModule, llvmBuilder,
        warningsAsErrors);
      if (converted.typeOf() == targetCallLLVMType)
        return converted;
      return LLVMBitCast.create(llvmBuilder, ".argBitcast", converted,
                                targetCallLLVMType);
    }
  }

  /**
   * Specifies a parameter for which, in any function call, the corresponding
   * actual argument should not be type-checked and should be discarded after
   * evaluation instead of being passed to the target call.
   */
  public static final SrcParamType SRC_PARAM_DISCARD = new Impl(){
    @Override
    public SrcType getCallSrcType(
      Traversable callNode, String calleeName, String paramName,
      ValueAndType args[], SrcType typeArg, SrcSymbolTable srcSymbolTable,
      LLVMModule llvmModule, boolean warningsAsErrors)
    {
      throw new IllegalStateException();
    }
    @Override
    public LLVMType getTargetCallLLVMType(
      SrcType callSrcType, SrcType typeArg, LLVMModule llvmModule)
    {
      throw new IllegalStateException();
    }
    @Override
    public boolean discardsArg() {
      return true;
    }
    @Override
    public LLVMValue processArg(
      String calleeName, String paramName, SrcType callReturnSrcType,
      SrcType callSrcType, LLVMType targetCallLLVMType,
      Iterator<ValueAndType> argItr, Iterator<LLVMValue> targetOnlyArgItr,
      ValueAndType prevArg, LLVMValue prevArgLLVM,
      SrcSymbolTable srcSymbolTable, LLVMModule llvmModule,
      int llvmModuleIndex, LLVMInstructionBuilder llvmBuilder,
      boolean warningsAsErrors)
      throws SrcInsufficientArgsException
    {
      throw new IllegalStateException();
    }
  };

  /**
   * Specifies a pass-by-reference parameter of type {@code va_list}, for
   * which the underlying {@link SrcStructType} isn't built unless needed. In
   * any function call, the actual value must be a {@code va_list} lvalue,
   * which will be cast to a void pointer for the target call.
   */
  public static final SrcParamType SRC_PARAM_VA_LIST_LVALUE = new Impl(){
    @Override
    public SrcType getCallSrcType(
      Traversable callNode, String calleeName, String paramName,
      ValueAndType args[], SrcType typeArg, SrcSymbolTable srcSymbolTable,
      LLVMModule llvmModule, boolean warningsAsErrors)
    {
      return SrcPointerType.get(SrcVoidType);
    }
    @Override
    public LLVMValue processArg(
      String calleeName, String paramName, SrcType callReturnSrcType,
      SrcType callSrcType, LLVMType targetCallLLVMType,
      Iterator<ValueAndType> argItr, Iterator<LLVMValue> targetOnlyArgItr,
      ValueAndType prevArg, LLVMValue prevArgLLVM,
      SrcSymbolTable srcSymbolTable, LLVMModule llvmModule,
      int llvmModuleIndex, LLVMInstructionBuilder llvmBuilder,
      boolean warningsAsErrors)
      throws SrcInsufficientArgsException
    {
      if (!argItr.hasNext())
        throw new IllegalStateException();
      final ValueAndType arg = argItr.next();
      if (!arg.isLvalue())
        throw new SrcRuntimeException(paramName+" to "+calleeName
                                      +" is not an lvalue");
      final SrcType vaList = srcSymbolTable.getBuiltinTypeTable()
                             .getVaList(llvmModule, warningsAsErrors);
      if (!arg.getSrcType().iso(vaList))
        throw new SrcRuntimeException(calleeName+" to "+paramName
                                      +" is not of type "+vaList);
      return LLVMBitCast.create(llvmBuilder, "", arg.getLLVMValue(),
                                targetCallLLVMType);
    }
  };

  /**
   * Specifies a pass-by-value parameter of type {@code va_list}, for which
   * the underlying {@link SrcStructType} isn't built unless needed. In any
   * function call, the actual value must be a {@code va_list}, which will be
   * evaluated in the same manner as any other argument.
   */
  public static final SrcParamType SRC_PARAM_VA_LIST_RVALUE = new Impl(){
    @Override
    public SrcType getCallSrcType(
      Traversable callNode, String calleeName, String paramName,
      ValueAndType args[], SrcType typeArg, SrcSymbolTable srcSymbolTable,
      LLVMModule llvmModule, boolean warningsAsErrors)
    {
      return srcSymbolTable.getBuiltinTypeTable()
             .getVaList(llvmModule, warningsAsErrors);
    }
  };

  /**
   * Specifies a parameter of scalar type for which, in any function call, the
   * corresponding actual argument should be compared not equal
   * zero/null-pointer and the resulting i1 passed to the target call. That
   * is, the actual argument is evaluated using
   * {@link ValueAndType#evalAsCond}.
   */
  public static final SrcParamType SRC_PARAM_COND = new Impl(){
    @Override
    public SrcType getCallSrcType(
      Traversable callNode, String calleeName, String paramName,
      ValueAndType args[], SrcType typeArg, SrcSymbolTable srcSymbolTable,
      LLVMModule llvmModule, boolean warningsAsErrors)
    {
      return null; // i1
    }
    @Override
    public LLVMType getTargetCallLLVMType(
      SrcType callSrcType, SrcType typeArg, LLVMModule llvmModule)
    {
      return LLVMIntegerType.get(llvmModule.getContext(), 1);
    }
    @Override
    public LLVMValue processArg(
      String calleeName, String paramName, SrcType callReturnSrcType,
      SrcType callSrcType, LLVMType targetCallLLVMType,
      Iterator<ValueAndType> argItr, Iterator<LLVMValue> targetOnlyArgItr,
      ValueAndType prevArg, LLVMValue prevArgLLVM,
      SrcSymbolTable srcSymbolTable, LLVMModule llvmModule,
      int llvmModuleIndex, LLVMInstructionBuilder llvmBuilder,
      boolean warningsAsErrors)
      throws SrcInsufficientArgsException
    {
      if (!argItr.hasNext())
        throw new IllegalStateException();
      final ValueAndType arg = argItr.next();
      return arg.evalAsCond(paramName+" to "+calleeName, llvmModule,
                            llvmBuilder);
    }
  };

  /**
   * Specifies a parameter of type {@link SrcIntType} for which, in any
   * function call, the corresponding actual argument should be compared
   * greater than 1 and the resulting i1 passed to the target call.
   */
  public static final SrcParamType SRC_PARAM_INT_GT1 = new Impl(){
    @Override
    public SrcType getCallSrcType(
      Traversable callNode, String calleeName, String paramName,
      ValueAndType args[], SrcType typeArg, SrcSymbolTable srcSymbolTable,
      LLVMModule llvmModule, boolean warningsAsErrors)
    {
      return null; // i1
    }
    @Override
    public LLVMType getTargetCallLLVMType(
      SrcType callSrcType, SrcType typeArg, LLVMModule llvmModule)
    {
      return LLVMIntegerType.get(llvmModule.getContext(), 1);
    }
    @Override
    public LLVMValue processArg(
      String calleeName, String paramName, SrcType callReturnSrcType,
      SrcType callSrcType, LLVMType targetCallLLVMType,
      Iterator<ValueAndType> argItr, Iterator<LLVMValue> targetOnlyArgItr,
      ValueAndType prevArg, LLVMValue prevArgLLVM,
      SrcSymbolTable srcSymbolTable, LLVMModule llvmModule,
      int llvmModuleIndex, LLVMInstructionBuilder llvmBuilder,
      boolean warningsAsErrors)
      throws SrcInsufficientArgsException
    {
      final LLVMContext ctxt = llvmModule.getContext();
      if (!argItr.hasNext())
        throw new IllegalStateException();
      final ValueAndType arg = argItr.next();
      final LLVMValue converted = arg.convertForAssign(
        SrcIntType, new ArgAssignKind(calleeName, paramName),
        llvmModule, llvmBuilder, warningsAsErrors);
      return LLVMIntegerComparison.create(
        llvmBuilder, ".gt1", LLVMIntPredicate.LLVMIntSGT, converted,
        LLVMConstantInteger.get(SrcIntType.getLLVMType(ctxt), 1, false));
    }
  };

  /**
   * Specifies a metadata parameter. In any function call, the corresponding
   * actual argument is passed unmodified to the target LLVM intrinsic call.
   * That actual argument must be generated by the compiler as there is no
   * corresponding source type for a source-level call. That is, this is an
   * explicit target-only parameter type.
   */
  public static final SrcParamType SRC_PARAM_METADATA = new Impl(){
    @Override
    public SrcType getCallSrcType(
      Traversable callNode, String calleeName, String paramName,
      ValueAndType args[], SrcType typeArg, SrcSymbolTable srcSymbolTable,
      LLVMModule llvmModule, boolean warningsAsErrors)
    {
      return null; // metadata
    }
    @Override
    public LLVMType getTargetCallLLVMType(
      SrcType callSrcType, SrcType typeArg, LLVMModule llvmModule)
    {
      return LLVMMetadataType.get(llvmModule.getContext());
    }
    @Override
    public LLVMValue processArg(
      String calleeName, String paramName, SrcType callReturnSrcType,
      SrcType callSrcType, LLVMType targetCallLLVMType,
      Iterator<ValueAndType> argItr, Iterator<LLVMValue> targetOnlyArgItr,
      ValueAndType prevArg, LLVMValue prevArgLLVM,
      SrcSymbolTable srcSymbolTable, LLVMModule llvmModule,
      int llvmModuleIndex, LLVMInstructionBuilder llvmBuilder,
      boolean warningsAsErrors)
      throws SrcInsufficientArgsException
    {
      if (!targetOnlyArgItr.hasNext())
        throw new IllegalStateException();
      final LLVMValue res = targetOnlyArgItr.next();
      if (!(res instanceof LLVMMDNode) && !(res instanceof LLVMMDString))
        throw new IllegalStateException();
      return res;
    }
  };

  /**
   * Specifies a parameter to receive the checksum for the target type of
   * the preceding actual argument's type, which must be a pointer type. If
   * the preceding actual argument is a null pointer constant, it's assumed
   * to have type {@code void*}, so the checksum is for {@code void}. For
   * the {@link #SRC_PARAM_TYPE_CHECKSUM_FROM_PTR_ARG} parameter, there is
   * no corresponding actual argument at the source level, and the argument
   * at the target level is generated automatically. That is, this is an
   * implicit target-only parameter type.
   */
  public static final SrcParamType SRC_PARAM_TYPE_CHECKSUM_FROM_PTR_ARG
    = new Impl()
  {
    @Override
    public SrcType getCallSrcType(
      Traversable callNode, String calleeName, String paramName,
      ValueAndType args[], SrcType typeArg, SrcSymbolTable srcSymbolTable,
      LLVMModule llvmModule, boolean warningsAsErrors)
    {
      return SrcPointerType.get(
        SrcQualifiedType.get(SrcCharType, SrcTypeQualifier.CONST));
    }
    @Override
    public LLVMValue processArg(
      String calleeName, String paramName, SrcType callReturnSrcType,
      SrcType callSrcType, LLVMType targetCallLLVMType,
      Iterator<ValueAndType> argItr, Iterator<LLVMValue> targetOnlyArgItr,
      ValueAndType prevArg, LLVMValue prevArgLLVM,
      SrcSymbolTable srcSymbolTable, LLVMModule llvmModule,
      int llvmModuleIndex, LLVMInstructionBuilder llvmBuilder,
      boolean warningsAsErrors)
      throws SrcInsufficientArgsException
    {
      assert(prevArg != null);
      final SrcType targetType;
      if (prevArg.isNullPointerConstant())
        targetType = SrcVoidType;
      else {
        final SrcType type = prevArg.getSrcType().prepareForOp();
        assert(type instanceof SrcPointerType);
        targetType = ((SrcPointerType)type).getTargetType();
      }
      return srcSymbolTable.computeTypeChecksumPointer(
               targetType, llvmModule, llvmModuleIndex, llvmBuilder);
    }
  };

  /**
   * Specifies a parameter to receive the checksum for the target type of
   * the function's return type, which must be a pointer type, and which
   * might be computed from a type argument. For any function call, there is
   * no corresponding actual argument at the source level, and the argument
   * at the target level is generated automatically. That is, this is an
   * implicit target-only parameter type.
   */
  public static final SrcParamType SRC_PARAM_TYPE_CHECKSUM_FROM_RET_TYPE
    = new Impl()
  {
    @Override
    public SrcType getCallSrcType(
      Traversable callNode, String calleeName, String paramName,
      ValueAndType args[], SrcType typeArg, SrcSymbolTable srcSymbolTable,
      LLVMModule llvmModule, boolean warningsAsErrors)
    {
      return SrcPointerType.get(
        SrcQualifiedType.get(SrcCharType, SrcTypeQualifier.CONST));
    }
    @Override
    public LLVMValue processArg(
      String calleeName, String paramName, SrcType callReturnSrcType,
      SrcType callSrcType, LLVMType targetCallLLVMType,
      Iterator<ValueAndType> argItr, Iterator<LLVMValue> targetOnlyArgItr,
      ValueAndType prevArg, LLVMValue prevArgLLVM,
      SrcSymbolTable srcSymbolTable, LLVMModule llvmModule,
      int llvmModuleIndex, LLVMInstructionBuilder llvmBuilder,
      boolean warningsAsErrors)
      throws SrcInsufficientArgsException
    {
      assert(callReturnSrcType instanceof SrcPointerType);
      final SrcPointerType returnPtrType
        = (SrcPointerType)callReturnSrcType;
      return srcSymbolTable.computeTypeChecksumPointer(
               returnPtrType.getTargetType(), llvmModule,
               llvmModuleIndex, llvmBuilder);
    }
  };

  /**
   * Specifies a parameter to receive a metadata node containing a null
   * pointer constant. For any function call, there is no corresponding
   * actual argument at the source level, and the argument at the target
   * level is generated automatically. That is, this is an implicit
   * target-only parameter type. The type of the null pointer constant is
   * the source-level return type, which must be a pointer type.
   */
  public static final SrcParamType SRC_PARAM_METADATA_NULL_PTR_FROM_RET_TYPE
    = new Impl()
  {
    @Override
    public SrcType getCallSrcType(
      Traversable callNode, String calleeName, String paramName,
      ValueAndType args[], SrcType typeArg, SrcSymbolTable srcSymbolTable,
      LLVMModule llvmModule, boolean warningsAsErrors)
    {
      return null; // metadata
    }
    @Override
    public LLVMType getTargetCallLLVMType(
      SrcType callSrcType, SrcType typeArg, LLVMModule llvmModule)
    {
      return LLVMMetadataType.get(llvmModule.getContext());
    }
    @Override
    public LLVMValue processArg(
      String calleeName, String paramName, SrcType callReturnSrcType,
      SrcType callSrcType, LLVMType targetCallLLVMType,
      Iterator<ValueAndType> argItr, Iterator<LLVMValue> targetOnlyArgItr,
      ValueAndType prevArg, LLVMValue prevArgLLVM,
      SrcSymbolTable srcSymbolTable, LLVMModule llvmModule,
      int llvmModuleIndex, LLVMInstructionBuilder llvmBuilder,
      boolean warningsAsErrors)
      throws SrcInsufficientArgsException
    {
      assert(callReturnSrcType instanceof SrcPointerType);
      final LLVMContext ctxt = llvmModule.getContext();
      return LLVMMDNode.get(
        ctxt, LLVMConstant.constNull(callReturnSrcType.getLLVMType(ctxt)));
    }
  };

  /**
   * Specifies a parameter to receive a metadata node containing a null
   * pointer constant. For any function call, there is no corresponding
   * actual argument at the source level, and the argument at the target
   * level is generated automatically. That is, this is an implicit
   * target-only parameter type. The type of the null pointer constant is
   * the source-level type of the preceding actual argument, which must be a
   * pointer type. If the preceding actual argument is a null pointer
   * constant, its actual LLVM argument type is used, which must be of
   * pointer type. In that way, it's up to the preceding formal parameter
   * type to dictate the translation of that null pointer constant to an
   * LLVM pointer type.
   */
  public static final SrcParamType SRC_PARAM_METADATA_NULL_PTR_FROM_ARG_TYPE
    = new Impl()
  {
    @Override
    public SrcType getCallSrcType(
      Traversable callNode, String calleeName, String paramName,
      ValueAndType args[], SrcType typeArg, SrcSymbolTable srcSymbolTable,
      LLVMModule llvmModule, boolean warningsAsErrors)
    {
      return null; // metadata
    }
    @Override
    public LLVMType getTargetCallLLVMType(
      SrcType callSrcType, SrcType typeArg, LLVMModule llvmModule)
    {
      return LLVMMetadataType.get(llvmModule.getContext());
    }
    @Override
    public LLVMValue processArg(
      String calleeName, String paramName, SrcType callReturnSrcType,
      SrcType callSrcType, LLVMType targetCallLLVMType,
      Iterator<ValueAndType> argItr, Iterator<LLVMValue> targetOnlyArgItr,
      ValueAndType prevArg, LLVMValue prevArgLLVM,
      SrcSymbolTable srcSymbolTable, LLVMModule llvmModule,
      int llvmModuleIndex, LLVMInstructionBuilder llvmBuilder,
      boolean warningsAsErrors)
      throws SrcInsufficientArgsException
    {
      final LLVMContext ctxt = llvmModule.getContext();
      assert(prevArg != null);
      final LLVMPointerType llvmPtrTy;
      if (prevArg.isNullPointerConstant()) {
        assert(prevArgLLVM != null);
        final LLVMType prevTy = prevArgLLVM.typeOf();
        assert(prevTy instanceof LLVMPointerType);
        llvmPtrTy = (LLVMPointerType)prevTy;
      }
      else {
        final SrcType type = prevArg.getSrcType().prepareForOp();
        assert(type instanceof SrcPointerType);
        llvmPtrTy = ((SrcPointerType)type).getLLVMType(ctxt);
      }
      return LLVMMDNode.get(ctxt, LLVMConstant.constNull(llvmPtrTy));
    }
  };

  /**
   * Specifies a parameter to receive a null pointer constant of type
   * {@code i8*}. For any function call, there is no corresponding actual
   * argument at the source level, and the argument at the target level is
   * generated automatically. That is, this is an implicit target-only
   * parameter type.
   */
  public static final SrcParamType SRC_PARAM_NULL_VOID_PTR = new Impl(){
    @Override
    public SrcType getCallSrcType(
      Traversable callNode, String calleeName, String paramName,
      ValueAndType args[], SrcType typeArg, SrcSymbolTable srcSymbolTable,
      LLVMModule llvmModule, boolean warningsAsErrors)
    {
      return SrcPointerType.get(SrcVoidType);
    }
    @Override
    public LLVMValue processArg(
      String calleeName, String paramName, SrcType callReturnSrcType,
      SrcType callSrcType, LLVMType targetCallLLVMType,
      Iterator<ValueAndType> argItr, Iterator<LLVMValue> targetOnlyArgItr,
      ValueAndType prevArg, LLVMValue prevArgLLVM,
      SrcSymbolTable srcSymbolTable, LLVMModule llvmModule,
      int llvmModuleIndex, LLVMInstructionBuilder llvmBuilder,
      boolean warningsAsErrors)
      throws SrcInsufficientArgsException
    {
      return LLVMConstant.constNull(targetCallLLVMType);
    }
  };

  /**
   * Specifies a parameter that is either (1) of pointer type with an
   * nvl-qualified target type, or (2) a null pointer constant. In any
   * function call, the corresponding actual argument will be cast to a
   * {@code i8 addrspace({@link #LLVM_ADDRSPACE_NVL})*} for the target call.
   */
  public static final SrcParamType SRC_PARAM_PTR_TO_NVL = new Impl(){
    @Override
    public SrcType getCallSrcType(
      Traversable callNode, String calleeName, String paramName,
      ValueAndType args[], SrcType typeArg, SrcSymbolTable srcSymbolTable,
      LLVMModule llvmModule, boolean warningsAsErrors)
    {
      return null; // i8 addrspace(LLVM_ADDRSPACE_NVL)*
    }
    @Override
    public LLVMType getTargetCallLLVMType(
      SrcType callSrcType, SrcType typeArg, LLVMModule llvmModule)
    {
      return LLVMPointerType.get(
        SrcVoidType.getLLVMTypeAsPointerTarget(llvmModule.getContext()),
        SrcPointerType.LLVM_ADDRSPACE_NVL);
    }
    @Override
    public LLVMValue processArg(
      String calleeName, String paramName, SrcType callReturnSrcType,
      SrcType callSrcType, LLVMType targetCallLLVMType,
      Iterator<ValueAndType> argItr, Iterator<LLVMValue> targetOnlyArgItr,
      ValueAndType prevArg, LLVMValue prevArgLLVM,
      SrcSymbolTable srcSymbolTable, LLVMModule llvmModule,
      int llvmModuleIndex, LLVMInstructionBuilder llvmBuilder,
      boolean warningsAsErrors)
      throws SrcInsufficientArgsException
    {
      if (!argItr.hasNext())
        throw new IllegalStateException();
      final ValueAndType arg = argItr.next();
      final ValueAndType argPrep = arg.prepareForOp(llvmModule,
                                                    llvmBuilder);
      if (argPrep.isNullPointerConstant())
        return LLVMConstant.constNull(targetCallLLVMType);
      final SrcPointerType ptrType = argPrep.getSrcType()
                                     .toIso(SrcPointerType.class);
      if (ptrType == null || !ptrType.getTargetType()
                             .hasEffectiveQualifier(SrcTypeQualifier.NVL))
        throw new SrcRuntimeException(
          paramName+" to "+calleeName+" is not a null pointer constant or a"
          +" pointer to an nvl-qualified type");
      return LLVMBitCast.create(llvmBuilder, ".arg2voidPtr",
                                argPrep.getLLVMValue(), targetCallLLVMType);
    }
  };

  /**
   * Specifies a parameter of type {@code MPI_Group}, however that is
   * defined at each function call. At each function call, {@code MPI_Group}
   * must be defined (hopefully by inclusion of a standard-conforming
   * {@code mpi.h}) as a typedef to an integer type or pointer type. In the
   * case of a pointer type, the corresponding actual argument will be cast
   * to an {@code i8*} for the target call. Also, if the target call is to a
   * function, then either {@code .mpiGroupPtr} or {@code .mpiGroupInt} is
   * appended to the name of that function to indicate whether
   * {@code MPI_Group} is a pointer type or integer type at that call.
   * Moreover, this parameter must not be specified as overloaded to a
   * {@link SrcFunctionBuiltin} constructor as it will be implicitly
   * overloaded if and only if {@code MPI_Group} is an integer type, which
   * is then appended as well.
   */
  public static final SrcParamType SRC_PARAM_MPI_GROUP = new Impl(){
    @Override
    public SrcType getCallSrcType(
      Traversable callNode, String calleeName, String paramName,
      ValueAndType args[], SrcType typeArg, SrcSymbolTable srcSymbolTable,
      LLVMModule llvmModule, boolean warningsAsErrors)
    {
      final SrcType mpiGroupType;
      {
        final List<Specifier> spec = new ArrayList<>(1);
        spec.add(new UserSpecifier(SymbolTools.getOrphanID("MPI_Group")));
        try {
          mpiGroupType
            = srcSymbolTable.typeSpecifiersAndQualifiersToSrcType(
                spec, callNode, llvmModule, warningsAsErrors);
        }
        catch (SrcRuntimeException e) {
          throw new SrcRuntimeException("at "+paramName+" to "+calleeName
                                        +", "+e.getMessage());
        }
      }
      assert(mpiGroupType != null);
      if (!mpiGroupType.iso(SrcIntegerType.class)
          && !mpiGroupType.iso(SrcPointerType.class))
        throw new SrcRuntimeException(
          "at call to "+calleeName+", MPI_Group is a typedef to a"
          +" non-integral, non-pointer type: "+mpiGroupType);
      return mpiGroupType;
    }
    @Override
    public LLVMType getTargetCallLLVMType(
      SrcType callSrcType, SrcType typeArg, LLVMModule llvmModule)
    {
      final SrcType targetCallSrcType;
      if (callSrcType.iso(SrcPointerType.class))
        targetCallSrcType = SrcPointerType.get(SrcVoidType);
      else
        targetCallSrcType = callSrcType;
      return targetCallSrcType.getLLVMType(llvmModule.getContext());
    }
    @Override
    public boolean overloadsWithLLVMSuffix(LLVMType targetCallLLVMType,
                                           boolean overloadsCall)
    {
      if (overloadsCall)
        throw new IllegalStateException();
      return targetCallLLVMType instanceof LLVMIntegerType;
    }
    @Override
    public String getSpecialOverloadSuffix(LLVMType targetCallLLVMType) {
      if (targetCallLLVMType instanceof LLVMPointerType)
        return ".mpiGroupPtr";
      assert(targetCallLLVMType instanceof LLVMIntegerType);
      return ".mpiGroupInt";
    }
  };
}
