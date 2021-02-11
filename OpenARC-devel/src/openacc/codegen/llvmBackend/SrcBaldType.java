package openacc.codegen.llvmBackend;

import java.util.EnumSet;
import java.util.Iterator;
import java.util.Set;

import openacc.codegen.llvmBackend.SrcQualifiedType.SrcTypeQualifier;

import org.jllvm.LLVMContext;
import org.jllvm.LLVMInstructionBuilder;
import org.jllvm.LLVMLoadInstruction;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMStoreInstruction;
import org.jllvm.LLVMType;
import org.jllvm.LLVMValue;

/**
 * The LLVM backend's base class for all types from the C source that are not
 * typedefs and that have no top-level type qualifiers.
 * 
 * <p>
 * These types might have type qualifiers or typedefs for target types,
 * element types, or member types.
 * </p>
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public abstract class SrcBaldType extends SrcUnqualifiedType {
  /**
   * Same as {@link #eqv(SrcType)} except each type is always a
   * {@link SrcBaldType}.
   */
  public abstract boolean eqvBald(SrcBaldType other);

  @Override
  protected final SrcTopLevelExpandedType expandTopLevelTypedefs(
    EnumSet<SrcTypeQualifier> addQuals)
  {
    // Calling SrcQualifiedType.get is important to ensure that qualifiers
    // are merged into the type correctly (in the case of an array type).
    final SrcType hairy = SrcQualifiedType.get(this, addQuals);
    return new SrcTopLevelExpandedType(
      (SrcBaldType)hairy.unqualifiedUnexpanded(),
      hairy.getSrcTypeQualifiers());
  }

  @Override
  public abstract boolean isIncompleteType();

  @Override
  public abstract boolean hasEffectiveQualifier(SrcTypeQualifier... quals);

  @Override
  public abstract SrcBaldType withoutEffectiveQualifier(
    SrcTypeQualifier... quals);

  /**
   * Same as {@link #iterator} except, before returning the iterator,
   * effectively advance the iterator past this type to the first component
   * type. Thus, the first call to {@link Iterator#hasNext} on the iterator
   * after it is returned might not return true.
   */
  public abstract SrcTypeIterator componentIterator(boolean storageOnly,
                                                    Set<SrcType> skipTypes);

  /**
   * Same as {@link #buildComposite} except each type is always a
   * {@link SrcBaldType}.
   */
  public abstract SrcBaldType buildCompositeBald(
    SrcBaldType other, boolean warn, boolean warningsAsErrors,
    String errMsg);

  /**
   * Same as {@link #getLLVMType(LLVMContext)} except that top-level typedefs
   * have been expanded on the type and the remaining top-level qualifiers are
   * specified as the first parameter.
   */
  public LLVMType getLLVMType(
    EnumSet<SrcTypeQualifier> srcTypeQualifiers, LLVMContext context)
  {
    return getLLVMType(context);
  }

  /**
   * Same as {@link #getLLVMTypeAsPointerTarget(LLVMContext)} except that
   * top-level typedefs have been expanded on the type and the remaining
   * top-level qualifiers are specified as the first parameter.
   * 
   * TODO: See todo on getLLVMType.
   */
  public LLVMType getLLVMTypeAsPointerTarget(
    EnumSet<SrcTypeQualifier> srcTypeQualifiers, LLVMContext context)
  {
    return getLLVMTypeAsPointerTarget(context);
  }

  /**
   * Same as {@link #prepareForOp()} except that top-level typedefs have been
   * expanded on the type and the remaining top-level type qualifiers are
   * specified as the first parameter.
   */
  public SrcType prepareForOp(EnumSet<SrcTypeQualifier> srcTypeQualifiers) {
    return this;
  }
  /**
   * Same as
   * {@link #prepareForOp(LLVMValue, boolean, LLVMModule, LLVMInstructionBuilder)}
   * except that top-level typedefs have been expanded on the type and the
   * remaining top-level type qualifiers are specified as the first parameter.
   */
  public ValueAndType prepareForOp(
    EnumSet<SrcTypeQualifier> srcTypeQualifiers, LLVMValue value,
    boolean lvalueOrFnDesignator, LLVMModule module,
    LLVMInstructionBuilder builder)
  {
    if (!lvalueOrFnDesignator)
      return new ValueAndType(value, prepareForOp(), false);
    // ISO C99 sec. 6.3.2.1p2.
    final ValueAndType rvalue = load(srcTypeQualifiers, value, module,
                                     builder);
    assert(rvalue.getSrcType().eqv(this));
    return rvalue;
  }

  /**
   * Same as {@link #load(LLVMValue, LLVMModule, LLVMInstructionBuilder)}
   * except that top-level typedefs have been expanded on the type and the
   * remaining top-level qualifiers are specified as the first parameter.
   * The result type is thus always exactly this type.
   */
  public ValueAndType load(EnumSet<SrcTypeQualifier> srcTypeQualifiers,
                           LLVMValue lvalue, LLVMModule module,
                           LLVMInstructionBuilder builder)
  {
    return new ValueAndType(new LLVMLoadInstruction(builder, ".load", lvalue),
                            this, false);
  }

  /**
   * Same as
   * {@link #store(LLVMValue, LLVMValue, LLVMModule, LLVMInstructionBuilder)}
   * except that top-level typedefs have been expanded on the type and the
   * remaining top-level qualifiers are specified as the first parameter.
   */
  public void store(EnumSet<SrcTypeQualifier> srcTypeQualifiers,
                    boolean forInit, LLVMValue lvalue, LLVMValue rvalue,
                    LLVMModule module, LLVMInstructionBuilder builder)
  {
    // The caller should have validated this, as documented in preconditions
    // on this method.
    assert(forInit || !srcTypeQualifiers.contains(SrcTypeQualifier.CONST));
    new LLVMStoreInstruction(builder, rvalue, lvalue);
  }

  @Override
  public abstract SrcType defaultArgumentPromoteNoPrep();

  @Override
  public abstract LLVMValue defaultArgumentPromoteNoPrep(
    LLVMValue value, LLVMModule module, LLVMInstructionBuilder builder);

  @Override
  public abstract LLVMValue convertFromNoPrep(
    ValueAndType from, String operation, LLVMModule module,
    LLVMInstructionBuilder builder);

  /**
   * This method is meant to be called only by {@link ValueAndType#evalAsCond}.
   * This method should be overridden only by {@link SrcScalarType} and its
   * subtypes.
   */
  @Override
  public LLVMValue evalAsCondNoPrep(String operand, LLVMValue value,
                                    LLVMModule module,
                                    LLVMInstructionBuilder builder)
  {
    throw new SrcRuntimeException(operand + " is not of scalar type");
  }

  /**
   * This method is meant to be called only by
   * {@link ValueAndType#indirection}. This method should be overridden only
   * by {@link SrcPointerType}.
   */
  @Override
  public ValueAndType indirectionNoPrep(String operand, LLVMValue value,
                                        LLVMModule module,
                                        LLVMInstructionBuilder builder)
  {
    throw new SrcRuntimeException(operand + " is not of pointer type");
  }

  /**
   * This method is meant to be called only by {@link ValueAndType#unaryPlus}.
   * This method should be overridden only by {@link SrcArithmeticType} and
   * its subtypes.
   */
  @Override
  public ValueAndType unaryPlusNoPrep(LLVMValue value, LLVMModule module,
                                      LLVMInstructionBuilder builder)
  {
    throw new SrcRuntimeException(
      "operand to unary \"+\" is not of arithmetic type");
  }

  /**
   * This method is meant to be called only by {@link ValueAndType#unaryMinus}.
   * This method should be overridden only by {@link SrcArithmeticType} and
   * its subtypes.
   */
  @Override
  public ValueAndType unaryMinusNoPrep(LLVMValue value, LLVMModule module,
                                       LLVMInstructionBuilder builder)
  {
    throw new SrcRuntimeException(
      "operand to unary \"-\" is not of arithmetic type");
  }

  /**
   * This method is meant to be called only by
   * {@link ValueAndType#unaryBitwiseComplement}. This method should be
   * overridden only by {@link SrcIntegerType} and its subtypes.
   */
  public ValueAndType unaryBitwiseComplementNoPrep(
    LLVMValue value, LLVMModule module, LLVMInstructionBuilder builder)
  {
    throw new SrcRuntimeException(
      "operand to unary \"~\" is not of integer type");
  }

  /**
   * This method is meant to be called only by {@link ValueAndType#unaryNot}.
   * This method should be overridden only by {@link SrcScalarType} and its
   * subtypes.
   */
  public ValueAndType unaryNotNoPrep(LLVMValue value, LLVMModule module,
                                     LLVMInstructionBuilder builder)
  {
    throw new SrcRuntimeException(
      "operand to unary \"!\" is not of scalar type");
  }
}
