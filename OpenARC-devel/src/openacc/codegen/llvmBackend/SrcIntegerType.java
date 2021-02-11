package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcBoolType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcIntType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedIntType;

import org.jllvm.LLVMAddInstruction;
import org.jllvm.LLVMConstant;
import org.jllvm.LLVMConstantInteger;
import org.jllvm.LLVMContext;
import org.jllvm.LLVMDivideInstruction;
import org.jllvm.LLVMDivideInstruction.DivisionType;
import org.jllvm.LLVMExtendCast;
import org.jllvm.LLVMExtendCast.ExtendType;
import org.jllvm.LLVMFloatComparison;
import org.jllvm.LLVMFloatToIntegerCast;
import org.jllvm.LLVMFloatToIntegerCast.FPToIntCastType;
import org.jllvm.LLVMInstructionBuilder;
import org.jllvm.LLVMIntegerComparison;
import org.jllvm.LLVMIntegerType;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMMultiplyInstruction;
import org.jllvm.LLVMPtrIntCast;
import org.jllvm.LLVMPtrIntCast.PtrIntCastType;
import org.jllvm.LLVMRemainderInstruction;
import org.jllvm.LLVMRemainderInstruction.RemainderType;
import org.jllvm.LLVMSubtractInstruction;
import org.jllvm.LLVMTruncateCast;
import org.jllvm.LLVMTruncateCast.TruncateType;
import org.jllvm.LLVMValue;
import org.jllvm.LLVMXorInstruction;
import org.jllvm.bindings.LLVMIntPredicate;
import org.jllvm.bindings.LLVMRealPredicate;

/**
 * The LLVM backend's base class for all integer types from the C source.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public abstract class SrcIntegerType extends SrcArithmeticType {
  /** What is the integer conversion rank for this type? */
  public abstract int getIntegerConversionRank();
  /** Is this a signed integer type? */
  public abstract boolean isSigned();
  /**
   * What is the unsigned type corresponding to this signed type, or
   * vice-versa? Returns null if there is no such type.
   */
  public abstract SrcIntegerType getSignednessToggle();

  /**
   * How many bits are available to represent a value in two's complement?
   * 
   * <p>
   * This is not necessarily the number of bits in the LLVM storage unit,
   * which is always equal or larger and is returned by {@link #getLLVMWidth}
   * instead. (Specifically, for {@link SrcPrimitiveIntegerType#SrcBoolType},
   * the result of this method is 1 while the result of {@link #getLLVMWidth}
   * is 8. For a bit-field, the result of this method is the bit-field's
   * declared width.)
   * </p>
   */
  public long getWidth() {
    return eqv(SrcBoolType) ? 1 : getLLVMWidth();
  }
  /**
   * How many bits are available to represent a positive integer value? That
   * is, compute the result of {@link #getWidth} minus any sign bit.
   */
  public long getPosWidth() {
    return getWidth() - (isSigned() ? 1 : 0);
  }
  /**
   * How many bits is the LLVM storage unit?
   * 
   * <p>
   * That is, apply {@link LLVMIntegerType#getWidth} to the result of
   * {@link #getLLVMType} but without the need to specify any
   * {@link LLVMContext} to {@link #getLLVMType}. See {@link #getLLVMType}
   * documentation for a special note about bit-fields.
   * <p>
   * 
   * <p>
   * This is not necessarily the number of bits available to represent a
   * value, which is returned by {@link #getWidth} or {@link #getPosWidth}
   * instead and which is always equal or smaller.
   * </p>
   */
  public abstract long getLLVMWidth();

  @Override
  public boolean isIncompleteType() {
    return false;
  }

  /**
   * Same as {@link #checkCompatibility} except the other type is always a
   * {@link SrcIntegerType}.
   */
  public abstract boolean checkIntegerCompatibility(
    SrcIntegerType other, boolean warn, boolean warningsAsErrors,
    String msgPrefix);

  @Override
  public final SrcIntegerType buildCompositeBald(
    SrcBaldType other, boolean warn, boolean warningsAsErrors,
    String msgPrefix)
  {
    final SrcIntegerType otherIntType = other.toEqv(SrcIntegerType.class);
    if (otherIntType == null) {
      BuildLLVM.warnOrError(warn, warningsAsErrors,
        msgPrefix+": integer type is incompatible with non-integer type: "
        +other);
      return null;
    }
    if (!checkIntegerCompatibility(otherIntType, warn, warningsAsErrors,
                                   msgPrefix))
      return null;
    return this;
  }

  @Override
  public LLVMIntegerType getLLVMType(LLVMContext context) {
    return LLVMIntegerType.get(context, getLLVMWidth());
  }
  @Override
  public LLVMIntegerType getLLVMTypeAsPointerTarget(LLVMContext context) {
    return getLLVMType(context);
  }

  /**
   * Same as
   * {@link #integerPromoteNoPrep(LLVMValue, LLVMModule, LLVMInstructionBuilder)}
   * except only compute the new type.
   */
  public SrcIntegerType integerPromoteNoPrep() {
    assert(SrcIntType.getIntegerConversionRank()
           == SrcUnsignedIntType.getIntegerConversionRank());
    assert(SrcIntType.isSigned());
    // ISO C99 says how to promote bit-fields of type _Bool, int, signed
    // int, or unsigned int. The target type is always an int or unsigned
    // int. It doesn't say how to promote other bit-field types that a
    // compiler chooses to support, but we should certainly not convert a
    // 33-bit-long bit-field to an int, for example, so we choose to keep
    // the storage unit type if the rank is greater than int.
    final SrcIntegerType storageUnitType;
    final SrcBitFieldType bitFieldType = toEqv(SrcBitFieldType.class);
    if (bitFieldType != null)
      storageUnitType = bitFieldType.getStorageUnitType();
    else
      storageUnitType = this;
    if (storageUnitType.getIntegerConversionRank()
        <= SrcIntType.getIntegerConversionRank())
      return getPosWidth() <= SrcIntType.getPosWidth() ? SrcIntType
                                                       : SrcUnsignedIntType;
    return storageUnitType;
  }

  /**
   * Perform the integer promotions.
   * 
   * <p>
   * That is, ISO C99 sec. 6.3.1.1p2-3.
   * <p>
   * 
   * <p>
   * {@link ValueAndType#prepareForOp} should have already been called on the
   * value and type if appropriate as it will not be called here. Moreover,
   * this must be an rvalue ({@link ValueAndType#prepareForOp} would ensure
   * that) as the result is always an rvalue.
   * </p>
   * 
   * @param llvmModule
   *          the LLVM module into which any required declarations should be
   *          generated for the conversions
   * @param llvmBuilder
   *          the IR builder with which any LLVM instructions should be
   *          generated for the conversion
   * @return the new rvalue and type, which together might be the same as the
   *         contents of {@code this} if no LLVM operations were required to
   *         perform the conversions and no type changes were required.
   */
  public ValueAndType integerPromoteNoPrep(LLVMValue value, LLVMModule module,
                                           LLVMInstructionBuilder builder)
  {
    final SrcIntegerType toType = integerPromoteNoPrep();
    final LLVMValue resultValue
      = toType.convertFromIntegerTypeNoPrep(this, value, module, builder);
    return new ValueAndType(resultValue, toType, false);
  }

  @Override
  public SrcIntegerType defaultArgumentPromoteNoPrep() {
    return integerPromoteNoPrep();
  }
  @Override
  public LLVMValue defaultArgumentPromoteNoPrep(LLVMValue value,
                                                LLVMModule module,
                                                LLVMInstructionBuilder builder)
  {
    return integerPromoteNoPrep(value, module, builder).getLLVMValue();
  }

  @Override
  public LLVMValue convertFromNoPrep(
    ValueAndType from, final String operation, LLVMModule module,
    LLVMInstructionBuilder builder)
  {
    final LLVMContext context = module.getContext();
    // ISO C99 sec. 6.3p2.
    if (this == from.getSrcType())
      return from.getLLVMValue();
    // ISO C99 sec. 6.3.1.2.
    if (iso(SrcBoolType)) {
      LLVMValue cmp;
      if (from.getSrcType().iso(SrcIntegerType.class)
          || from.getSrcType().iso(SrcPointerType.class))
        cmp = LLVMIntegerComparison.create(
                builder, ".cmp", LLVMIntPredicate.LLVMIntNE,
                from.getLLVMValue(),
                LLVMConstant.constNull(from.getSrcType().getLLVMType(context)));
      else if (from.getSrcType().iso(SrcPrimitiveFloatingType.class))
        cmp = LLVMFloatComparison.create(
                builder, ".cmp", LLVMRealPredicate.LLVMRealONE,
                from.getLLVMValue(),
                LLVMConstant.constNull(from.getSrcType().getLLVMType(context)));
      else
        throw new SrcRuntimeException(operation + " requires conversion from <"
                                      + from.getSrcType() + "> to <" + this
                                      + ">");
      return LLVMExtendCast.create(builder, ".ext", cmp, getLLVMType(context),
                                   ExtendType.ZERO);
    }
    final SrcIntegerType fromIntegerType
      = from.getSrcType().toIso(SrcIntegerType.class);
    if (fromIntegerType != null) {
      final LLVMValue val = from.getLLVMValue();
      return convertFromIntegerTypeNoPrep(fromIntegerType, val, module,
                                          builder);
    }
    // ISO C9 6.3.1.4p1.
    if (from.getSrcType().iso(SrcPrimitiveFloatingType.class)) {
      // When using clang (clang-600.0.56) or the methods below, the behavior
      // when converting a negative floating point to unsigned int is different
      // based on whether the original expression is a constant expression.
      // If it's a constant expression, result is always zero (-0.1 => 0 and
      // -3.5 => 0). If it's not a constant expression, then it truncates
      // towards zero as if signed, and then the bits are reinterpreted as
      // unsigned (-0.1 => 0 and -3.5 => (unsigned)-3). So, in the non-constant
      // expression case, we really could always use signed conversion below
      // regardless of whether the target type is signed. (ISO C99 says that,
      // any case where the value after truncating towards zero is negative has
      // undefined behavior.)
      return LLVMFloatToIntegerCast.create(
        builder, ".fptoi", from.getLLVMValue(), getLLVMType(context),
        isSigned() ? FPToIntCastType.SIGNED : FPToIntCastType.UNSIGNED);
    }
    final SrcPointerType fromPtrType = from.getSrcType()
                                       .toIso(SrcPointerType.class);
    if (fromPtrType != null) {
      new SrcTypeInvolvesNoNVMCheck(
        fromPtrType.getTargetType(),
        operation+" requires conversion to non-_Bool integer type from"
        +" pointer type involving NVM storage")
      .run();
      return LLVMPtrIntCast.create(builder, ".ptr2int", from.getLLVMValue(),
                                   getLLVMType(context),
                                   PtrIntCastType.PTR_TO_INT);
    }
    throw new SrcRuntimeException(operation + " requires conversion from <"
                                  + from.getSrcType() + "> to integer type");
  }

  /**
   * Does not handle destination type of {@link SrcBoolType}. See
   * {@link #convertFromNoPrep} for that.
   */
  private LLVMValue convertFromIntegerTypeNoPrep(
    SrcIntegerType fromIntegerType, LLVMValue val, LLVMModule module,
    LLVMInstructionBuilder builder)
  {
    assert(!iso(SrcBoolType));
    final LLVMContext context = module.getContext();
    // getWidth would be fine in place of getLLVMWidth everywhere below except
    // when _Bool or bit-fields are involved, for which getLLVMWidth can be
    // bigger than getWidth. In that case, LLVM would fail if we
    // chose to truncate, extend, or stay the same based on getWidth while
    // getLLVMWidth required a different decision. Fortunately, basing the
    // decision on getLLVMWidth still manages to produce the correct result
    // value. That is, if the from type is _Bool or a bit-field, any extra bits
    // from the from value are already the result of zero-extending or
    // sign-extending, so any truncation or extension required for the to type
    // will produce the same result value as if the from value didn't have
    // extra bits. If the to type is _Bool, we don't handle it here. If the to
    // type is a bit-field, any extra bits required by the to type will be
    // truncated before storing the result.

    // ISO C99 sec. 6.3.1.3 where the widths are the same. For p2 and p3, the
    // signedness is then different and the value cannot be represented in
    // the new type, so the sign changes but the bit representation stays the
    // same.
    if (getLLVMWidth() == fromIntegerType.getLLVMWidth())
      return val;
    // ISO C99 sec. 6.3.1.3, where the new width is smaller, so we just
    // truncate. For p1, that doesn't change the value. For p2, the integer
    // basically overflows. p3 is implementation defined, so we don't worry
    // about it too much.
    if (getLLVMWidth() < fromIntegerType.getLLVMWidth())
      return LLVMTruncateCast.create(builder, ".trunc", val,
                                     getLLVMType(context),
                                     TruncateType.INTEGER);
    // ISO C99 sec. 6.3.1.3, where the new width is larger. For p1, we just
    // extend while maintaining the same value. For p2 and p3, the signedness
    // is then different and the value cannot be represented in the new type,
    // so the sign changes, but the bit representation becomes the result of
    // sign/zero-extending as if the signedness of the new type were the
    // signedness of the original type.
    return LLVMExtendCast.create(builder, ".ext", val, getLLVMType(context),
                                 fromIntegerType.isSigned()
                                 ? ExtendType.SIGN : ExtendType.ZERO);
  }

  @Override
  public LLVMValue evalAsCondNoPrep(String operand, LLVMValue value,
                                    LLVMModule module,
                                    LLVMInstructionBuilder builder)
  {
    final LLVMConstant zero
      = LLVMConstant.constNull(getLLVMType(module.getContext()));
    return LLVMIntegerComparison.create(
      builder, ".cond", LLVMIntPredicate.LLVMIntNE, value, zero);
  }

  @Override
  public ValueAndType unaryPlusNoPrep(
    LLVMValue value, LLVMModule module, LLVMInstructionBuilder builder)
  {
    return integerPromoteNoPrep(value, module, builder);
  }

  @Override
  public ValueAndType unaryMinusNoPrep(
    LLVMValue value, LLVMModule module, LLVMInstructionBuilder builder)
  {
    final ValueAndType promoted = integerPromoteNoPrep(value, module,
                                                       builder);
    final LLVMConstant zero
      = LLVMConstant.constNull(promoted.getSrcType()
                               .getLLVMType(module.getContext()));
    return new ValueAndType(
      LLVMSubtractInstruction.create(builder, ".neg", zero,
                                     promoted.getLLVMValue(), false),
      promoted.getSrcType(), false);
  }

  @Override
  public ValueAndType unaryBitwiseComplementNoPrep(
    LLVMValue value, LLVMModule module, LLVMInstructionBuilder builder)
  {
    final ValueAndType promoted = integerPromoteNoPrep(value, module,
                                                       builder);
    value = promoted.getLLVMValue();
    final LLVMConstant ones
      = LLVMConstantInteger.allOnes(promoted.getSrcType()
                                    .getLLVMType(module.getContext()));
    value = LLVMXorInstruction.create(builder, ".bitwiseNot", value, ones);
    return new ValueAndType(value, promoted.getSrcType(), false);
  }

  @Override
  public LLVMValue unaryNotI1NoPrep(LLVMValue value, LLVMModule module,
                                    LLVMInstructionBuilder builder)
  {
    final LLVMConstant zero
      = LLVMConstant.constNull(getLLVMType(module.getContext()));
    return LLVMIntegerComparison.create(
      builder, ".notAsI1", LLVMIntPredicate.LLVMIntEQ, value, zero);
  }

  @Override
  public LLVMValue multiplyNoPrep(LLVMValue op1, LLVMValue op2,
                                  LLVMModule module,
                                  LLVMInstructionBuilder builder)
  {
    return LLVMMultiplyInstruction.create(builder, ".mul", op1, op2, false);
  }
  @Override
  public LLVMValue divideNoPrep(LLVMValue op1, LLVMValue op2,
                                LLVMModule module,
                                LLVMInstructionBuilder builder)
  {
    return LLVMDivideInstruction.create(
      builder, ".div", op1, op2,
      isSigned() ? DivisionType.SIGNEDINT : DivisionType.UNSIGNEDINT);
  }
  @Override
  public LLVMValue remainderNoPrep(LLVMValue op1, LLVMValue op2,
                                   LLVMModule module,
                                   LLVMInstructionBuilder builder)
  {
    return LLVMRemainderInstruction.create(
      builder, ".rem", op1, op2,
      isSigned() ? RemainderType.SIGNEDINT : RemainderType.UNSIGNEDINT);
  }
  @Override
  public LLVMValue addNoPrep(LLVMValue op1, LLVMValue op2,
                             LLVMModule module,
                             LLVMInstructionBuilder builder)
  {
    return LLVMAddInstruction.create(builder, ".add", op1, op2, false);
  }
  @Override
  public LLVMValue subtractNoPrep(LLVMValue op1, LLVMValue op2,
                                  LLVMModule module,
                                  LLVMInstructionBuilder builder)
  {
    return LLVMSubtractInstruction.create(builder, ".sub", op1, op2, false);
  }
  @Override
  public LLVMValue relationalI1NoPrep(LLVMValue op1, LLVMValue op2,
                                      boolean greater, boolean equals,
                                      LLVMModule module,
                                      LLVMInstructionBuilder builder)
  {
    final String oper = (greater?"g":"l")+(equals?"e":"t");
    final LLVMIntPredicate pred;
    if (isSigned()) {
      if (greater) pred = equals ? LLVMIntPredicate.LLVMIntSGE
                                 : LLVMIntPredicate.LLVMIntSGT;
      else         pred = equals ? LLVMIntPredicate.LLVMIntSLE
                                 : LLVMIntPredicate.LLVMIntSLT;
    }
    else {
      if (greater) pred = equals ? LLVMIntPredicate.LLVMIntUGE
                                 : LLVMIntPredicate.LLVMIntUGT;
      else         pred = equals ? LLVMIntPredicate.LLVMIntULE
                                 : LLVMIntPredicate.LLVMIntULT;
    }
    return LLVMIntegerComparison.create(builder, "."+oper+"AsI1", pred,
                                        op1, op2);
  }
  @Override
  public LLVMValue equalityI1NoPrep(LLVMValue op1, LLVMValue op2,
                                    boolean equals, LLVMModule module,
                                    LLVMInstructionBuilder builder)
  {
    final String oper = equals ? "eq" : "ne";
    final LLVMIntPredicate pred = equals ? LLVMIntPredicate.LLVMIntEQ
                                         : LLVMIntPredicate.LLVMIntNE;
    return LLVMIntegerComparison.create(builder, "."+oper+"AsI1", pred,
                                        op1, op2);
  }
}
