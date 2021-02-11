package openacc.codegen.llvmBackend;

import java.util.Iterator;

import org.jllvm.LLVMConstant;
import org.jllvm.LLVMInstructionBuilder;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMType;
import org.jllvm.LLVMValue;

import cetus.hir.Traversable;

/**
 * The LLVM backend's class for parameter types that require constant
 * expression actual arguments. Currently, this is used solely for function
 * builtins. See {@link SrcFunctionBuiltin}.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class SrcParamConstExprType extends SrcParamType.Impl {
  private final SrcType srcType;
  public SrcParamConstExprType(SrcType srcType) {
    this.srcType = srcType;
  }
  @Override
  public SrcType getCallSrcType(
    Traversable callNode, String calleeName, String paramName,
    ValueAndType args[], SrcType typeArg, SrcSymbolTable srcSymbolTable,
    LLVMModule llvmModule, boolean warningsAsErrors)
  {
    return srcType;
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
    final LLVMValue llvmArg = super.processArg(
      calleeName, paramName, callReturnSrcType, callSrcType,
      targetCallLLVMType, argItr, targetOnlyArgItr, prevArg, prevArgLLVM,
      srcSymbolTable, llvmModule, llvmModuleIndex, llvmBuilder,
      warningsAsErrors);
    if (!(llvmArg instanceof LLVMConstant))
      throw new SrcRuntimeException(paramName+" to "+calleeName+" is not a"
                                    +" constant");
    return llvmArg;
  }
}
