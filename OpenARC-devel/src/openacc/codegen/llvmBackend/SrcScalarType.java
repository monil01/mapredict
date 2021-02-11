package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcIntType;
import openacc.codegen.llvmBackend.SrcQualifiedType.SrcTypeQualifier;

import org.jllvm.LLVMContext;
import org.jllvm.LLVMExtendCast;
import org.jllvm.LLVMExtendCast.ExtendType;
import org.jllvm.LLVMInstructionBuilder;
import org.jllvm.LLVMIntegerType;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMTargetData;
import org.jllvm.LLVMValue;

/**
 * The LLVM backend's base class for all scalar types from the C source
 * (ISO C99 sec. 6.2.5p21).
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public abstract class SrcScalarType extends SrcBaldType {
  @Override
  public boolean hasEffectiveQualifier(SrcTypeQualifier... quals) {
    return false;
  }

  @Override
  public SrcScalarType withoutEffectiveQualifier(SrcTypeQualifier... quals) {
    return this;
  }

  @Override
  public abstract LLVMValue evalAsCondNoPrep(String operand, LLVMValue value,
                                             LLVMModule module,
                                             LLVMInstructionBuilder builder);

  @Override
  public ValueAndType unaryNotNoPrep(LLVMValue value, LLVMModule module,
                                     LLVMInstructionBuilder builder)
  {
    final LLVMContext ctxt = module.getContext();
    value = unaryNotI1NoPrep(value, module, builder);
    assert(value.typeOf() == LLVMIntegerType.get(ctxt, (long)1));
    final LLVMIntegerType resultLLVMType = SrcIntType.getLLVMType(ctxt);
    value = LLVMExtendCast.create(builder, ".not", value, resultLLVMType,
                                  ExtendType.ZERO);
    return new ValueAndType(value, SrcIntType, false);
  }
  /** This is meant to be called only by {@link #unaryNotNoPrep}. */
  public abstract LLVMValue unaryNotI1NoPrep(LLVMValue value,
                                             LLVMModule module,
                                             LLVMInstructionBuilder builder);
  /** This is meant to be called only by {@link SrcType#addNoPrep}. */
  public abstract LLVMValue addNoPrep(
    LLVMValue op1, LLVMValue op2,
    LLVMModule module, LLVMInstructionBuilder builder);
  /** This is meant to be called only by {@link SrcType#subtractNoPrep}. */
  public abstract LLVMValue subtractNoPrep(
    LLVMValue op1, LLVMValue op2, LLVMTargetData targetData,
    LLVMModule module, LLVMInstructionBuilder builder);
  /** This is meant to be called only by {@link SrcType#relationalNoPrep}. */
  public abstract LLVMValue relationalI1NoPrep(
    LLVMValue op1, LLVMValue op2, boolean greater, boolean equals,
    LLVMModule module, LLVMInstructionBuilder builder);
  /** This is meant to be called only by {@link SrcType#equalityNoPrep}. */
  public abstract LLVMValue equalityI1NoPrep(
    LLVMValue op1, LLVMValue op2, boolean equals,
    LLVMModule module, LLVMInstructionBuilder builder);
}
