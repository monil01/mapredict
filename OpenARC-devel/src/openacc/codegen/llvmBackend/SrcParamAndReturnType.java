package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcPrimitiveNonNumericType.SrcVoidType;

import java.util.Iterator;

import openacc.codegen.llvmBackend.SrcQualifiedType.SrcTypeQualifier;
import openacc.codegen.llvmBackend.ValueAndType.ArgAssignKind;

import org.jllvm.LLVMBitCast;
import org.jllvm.LLVMInstructionBuilder;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMType;
import org.jllvm.LLVMValue;

import cetus.hir.Traversable;

/**
 * The LLVM backend's base type for source types each of whose set of
 * possible roles must include both parameter type and return type. In other
 * words, the set of source types included is the intersection of the sets
 * included by {@link SrcParamType} and {@link SrcReturnType}.
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
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public interface SrcParamAndReturnType extends SrcParamType, SrcReturnType {
  /**
   * The implementation for {@link SrcParamAndReturnType}. See header
   * comments there for details.
   */
  public static abstract class Impl extends SrcParamOrReturnType.Impl
                                    implements SrcParamAndReturnType
  {
    @Override
    public final SrcType getCallSrcType(
      Traversable callNode, String calleeName, ValueAndType args[],
      SrcType typeArg, SrcSymbolTable srcSymbolTable,
      LLVMModule llvmModule, boolean warningsAsErrors)
    {
      return getCallSrcType(callNode, calleeName, null, args, typeArg,
                        srcSymbolTable, llvmModule, warningsAsErrors);
    }
    @Override
    public final boolean discardsArg() {
      return false;
    }
    /**
     * Keep this implementation in sync with
     * {@link SrcParamType.Impl#processArg}'s implementation.
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
   * Specifies a parameter/return type that is a pointer to
   * {@code nvl_heap_t}. In any function call, the corresponding actual
   * value will be cast to/from a void pointer for the target call.
   */
  public static final SrcParamAndReturnType SRC_PARAMRET_NVL_HEAP_PTR
    = new Impl()
  {
    @Override
    public SrcType getCallSrcType(
      Traversable callNode, String calleeName, String paramName,
      ValueAndType args[], SrcType typeArg, SrcSymbolTable srcSymbolTable,
      LLVMModule llvmModule, boolean warningsAsErrors)
    {
      return SrcPointerType.get(srcSymbolTable.getBuiltinTypeTable()
                                .getNVLHeapT(llvmModule, warningsAsErrors));
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
   * Same as {@link #SRC_PARAMRET_NVL_HEAP_PTR} except the pointer target
   * type is const-qualified.
   */
  public static final SrcParamAndReturnType SRC_PARAMRET_CONST_NVL_HEAP_PTR
    = new Impl()
  {
    @Override
    public SrcType getCallSrcType(
      Traversable callNode, String calleeName, String paramName,
      ValueAndType args[], SrcType typeArg, SrcSymbolTable srcSymbolTable,
      LLVMModule llvmModule, boolean warningsAsErrors)
    {
      return SrcPointerType.get(SrcQualifiedType.get(
        srcSymbolTable.getBuiltinTypeTable().getNVLHeapT(llvmModule,
                                                         warningsAsErrors),
        SrcTypeQualifier.CONST));
    }
    @Override
    public LLVMType getTargetCallLLVMType(
      SrcType callSrcType, SrcType typeArg, LLVMModule llvmModule)
    {
      return SrcPointerType.get(
               SrcQualifiedType.get(SrcVoidType, SrcTypeQualifier.CONST))
             .getLLVMType(llvmModule.getContext());
    }
  };
}
