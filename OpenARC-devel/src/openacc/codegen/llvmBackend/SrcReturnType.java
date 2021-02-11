package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcPrimitiveNonNumericType.SrcVoidType;
import openacc.codegen.llvmBackend.SrcQualifiedType.SrcTypeQualifier;

import org.jllvm.LLVMContext;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMPointerType;
import org.jllvm.LLVMType;

import cetus.hir.Traversable;

/**
 * The LLVM backend's interface for source types that can serve as return
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
public interface SrcReturnType extends SrcParamOrReturnType {
  /**
   * Subclasses override this method to provide an implementation for
   * {@link SrcParamOrReturnType#getCallSrcType} when the type is being used
   * as a return type. These methods are the same except that this method
   * has no {@code paramName} parameter and it never throws
   * {@link IllegalStateException} due to discarding an argument.
   */
  public SrcType getCallSrcType(
    Traversable callNode, String calleeName, ValueAndType args[],
    SrcType typeArg, SrcSymbolTable srcSymbolTable,
    LLVMModule llvmModule, boolean warningsAsErrors);

  /**
   * The implementation for {@link SrcReturnType}. See header comments there
   * for details.
   */
  public static abstract class Impl extends SrcParamOrReturnType.Impl
                                    implements SrcReturnType
  {
    @Override
    public final SrcType getCallSrcType(
      Traversable callNode, String calleeName, String paramName,
      ValueAndType args[], SrcType typeArg, SrcSymbolTable srcSymbolTable,
      LLVMModule llvmModule, boolean warningsAsErrors)
    {
      assert(paramName == null);
      return getCallSrcType(callNode, calleeName, args, typeArg,
                        srcSymbolTable, llvmModule, warningsAsErrors);
    }
  }

  /**
   * Specifies a return type that is copied verbatim from a type argument
   * passed to the function call.
   */
  public static final SrcReturnType SRC_RET_TYPE_IS_TYPE_ARG = new Impl(){
    @Override
    public SrcType getCallSrcType(
      Traversable callNode, String calleeName, ValueAndType args[],
      SrcType typeArg, SrcSymbolTable srcSymbolTable, LLVMModule llvmModule,
      boolean warningsAsErrors)
    {
      return typeArg;
    }
  };
  /**
   * Specifies a return type that is a pointer to an nvl-qualified version
   * of a type argument passed to the function call. Of course, that type
   * argument must be valid as an nvl-qualified type, and so it cannot be
   * already nvl_wp-qualified. The actual return value from the target call
   * is assumed to be a pointer to an nvl-qualified void (which is permitted
   * at the LLVM level but not the source level) and will be cast to the
   * formal return type.
   */
  public static final SrcReturnType SRC_RET_NVL_PTR_TYPE_FROM_TYPE_ARG
    = new Impl()
  {
    @Override
    public SrcType getCallSrcType(
      Traversable callNode, String calleeName, ValueAndType args[],
      SrcType typeArg, SrcSymbolTable srcSymbolTable, LLVMModule llvmModule,
      boolean warningsAsErrors)
    {
      return SrcPointerType.get(SrcQualifiedType.get(typeArg,
                                                     SrcTypeQualifier.NVL));
    }
    @Override
    public LLVMType getTargetCallLLVMType(
      SrcType callSrcType, SrcType typeArg, LLVMModule llvmModule)
    {
      final LLVMContext ctxt = llvmModule.getContext();
      // i8 addrspace(LLVM_ADDRSPACE_NVL)*
      return LLVMPointerType.get(
        SrcVoidType.getLLVMTypeAsPointerTarget(ctxt),
        SrcPointerType.LLVM_ADDRSPACE_NVL);
    }
  };

  /**
   * Same as {@link SRC_RET_NVL_PTR_TYPE_FROM_TYPE_ARG} except that the type
   * argument can be nvl_wp-qualified, and then the return type is a pointer
   * to the type argument without any additional type qualifier. Also, if
   * the target call is to a function, then either {@code .nv} or
   * {@code .wp} is appended to the name of that function to indicate
   * whether the return pointer type is nvl-qualified or nvl_wp-qualified.
   */
  public static final SrcReturnType SRC_RET_NVM_PTR_TYPE_FROM_TYPE_ARG
    = new Impl()
  {
    @Override
    public SrcType getCallSrcType(
      Traversable callNode, String calleeName, ValueAndType args[],
      SrcType typeArg, SrcSymbolTable srcSymbolTable, LLVMModule llvmModule,
      boolean warningsAsErrors)
    {
      if (typeArg.hasEffectiveQualifier(SrcTypeQualifier.NVL_WP))
        return SrcPointerType.get(typeArg);
      return SrcPointerType.get(SrcQualifiedType.get(typeArg,
                                                     SrcTypeQualifier.NVL));
    }
    @Override
    public LLVMType getTargetCallLLVMType(
      SrcType callSrcType, SrcType typeArg, LLVMModule llvmModule)
    {
      final LLVMContext ctxt = llvmModule.getContext();
      // i8 addrspace(LLVM_ADDRSPACE_NVL_WP)*
      if (typeArg.hasEffectiveQualifier(SrcTypeQualifier.NVL_WP))
        return LLVMPointerType.get(
          SrcVoidType.getLLVMTypeAsPointerTarget(ctxt),
          SrcPointerType.LLVM_ADDRSPACE_NVL_WP);
      // i8 addrspace(LLVM_ADDRSPACE_NVL)*
      return LLVMPointerType.get(
        SrcVoidType.getLLVMTypeAsPointerTarget(ctxt),
        SrcPointerType.LLVM_ADDRSPACE_NVL);
    }
    @Override
    public String getSpecialOverloadSuffix(LLVMType targetCallLLVMType) {
      final long addrspace = ((LLVMPointerType)targetCallLLVMType)
                             .getAddressSpace();
      if (addrspace == SrcPointerType.LLVM_ADDRSPACE_NVL)
        return ".nv";
      assert(addrspace == SrcPointerType.LLVM_ADDRSPACE_NVL_WP);
      return ".wp";
    }
  };

  /**
   * Specifies a return type that is the same as the type of the function
   * call's last explicit actual argument, which must be a pointer to an
   * nvl-qualified type, but the return type's target type drops the nvl
   * qualifier. The actual return value from the target call is assumed to
   * be a pointer to void and will be cast to the formal return type. If the
   * last explicit actual argument is a null pointer constant, an error is
   * reported. If the last explicit actual argument is not a pointer to an
   * nvl-qualified type or is missing, then it is assumed that parameter
   * validation will report the error later, and the resulting return type
   * is undefined.
   */
  public static final SrcReturnType SRC_RET_BARE_PTR_TYPE_FROM_ARG
    = new Impl()
  {
    @Override
    public SrcType getCallSrcType(
      Traversable callNode, String calleeName, ValueAndType args[],
      SrcType typeArg, SrcSymbolTable srcSymbolTable, LLVMModule llvmModule,
      boolean warningsAsErrors)
    {
      if (args.length == 0)
        return null;
      final ValueAndType lastArg = args[args.length - 1];
      if (lastArg.isNullPointerConstant())
        throw new SrcRuntimeException("cannot compute bare pointer from"
                                      +" null pointer constant");
      final SrcType lastArgTy = args[args.length - 1].getSrcType()
                                .prepareForOp();
      if (!(lastArgTy instanceof SrcPointerType))
        return null;
      final SrcPointerType lastArgPtrTy = (SrcPointerType)lastArgTy;
      final SrcType lastArgTargetTy = lastArgPtrTy.getTargetType();
      return SrcPointerType.get(
        lastArgTargetTy.withoutEffectiveQualifier(SrcTypeQualifier.NVL));
    }
    @Override
    public LLVMType getTargetCallLLVMType(
      SrcType callSrcType, SrcType typeArg, LLVMModule llvmModule)
    {
      return SrcPointerType.get(SrcVoidType)
             .getLLVMType(llvmModule.getContext());
    }
  };
  /**
   * Specifies a return type that is the same as the type of the function
   * call's first explicit actual argument, which must be a pointer to an
   * NVM-stored type. The actual return value from the target call is
   * assumed to be a pointer to void in the same address space and will be
   * cast to the formal return type.
   */
  public static final SrcReturnType SRC_RET_NVM_PTR_FROM_ARG = new Impl(){
    @Override
    public SrcType getCallSrcType(
      Traversable callNode, String calleeName, ValueAndType args[],
      SrcType typeArg, SrcSymbolTable srcSymbolTable, LLVMModule llvmModule,
      boolean warningsAsErrors)
    {
      assert(args.length > 0);
      return args[0].getSrcType().prepareForOp();
    }
    @Override
    public LLVMType getTargetCallLLVMType(
      SrcType callSrcType, SrcType typeArg, LLVMModule llvmModule)
    {
      final LLVMContext ctxt = llvmModule.getContext();
      assert(callSrcType instanceof SrcPointerType);
      final SrcType ptrTargetTy = ((SrcPointerType)callSrcType)
                                  .getTargetType();
      // i8 addrspace(LLVM_ADDRSPACE_NVL_WP)*
      if (ptrTargetTy.hasEffectiveQualifier(SrcTypeQualifier.NVL_WP))
        return LLVMPointerType.get(
          SrcVoidType.getLLVMTypeAsPointerTarget(ctxt),
          SrcPointerType.LLVM_ADDRSPACE_NVL_WP);
      // i8 addrspace(LLVM_ADDRSPACE_NVL)*
      return LLVMPointerType.get(
        SrcVoidType.getLLVMTypeAsPointerTarget(ctxt),
        SrcPointerType.LLVM_ADDRSPACE_NVL);
    }
  };
}
