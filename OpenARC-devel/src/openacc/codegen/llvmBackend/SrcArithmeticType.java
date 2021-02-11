package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcPrimitiveFloatingType.SrcDoubleType;
import static openacc.codegen.llvmBackend.SrcPrimitiveFloatingType.SrcFloatType;
import static openacc.codegen.llvmBackend.SrcPrimitiveFloatingType.SrcLongDoubleType;

import java.util.Set;

import org.jllvm.LLVMInstructionBuilder;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMTargetData;
import org.jllvm.LLVMValue;

/**
 * The LLVM backend's base class for all arithmetic types from the C
 * source (ISO C99 sec. 6.2.5p18).
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public abstract class SrcArithmeticType extends SrcScalarType {
  /**
   * Compute the type resulting from the usual arithmetic conversions (ISO C99
   * sec. 6.3.1.8).
   * 
   * @param operator
   *          the name of the operator for which the usual arithmetic
   *          conversions will be performed. For example,
   *          "{@code binary \"*\"}".
   * @param op1
   *          the first operand's type, which should be the result of a
   *          {@link SrcType#prepareForOp} call
   * @param op2
   *          the second operand's type, which should be the result of a
   *          {@link SrcType#prepareForOp} call
   * @return the resulting arithmetic type, to which both operands should be
   *         converted by the caller
   * @throws SrcRuntimeException
   *           if either {@code op1} or {@code op2} is not of arithmetic type
   */
  public static SrcArithmeticType getTypeFromUsualArithmeticConversionsNoPrep(
    String operator, SrcType op1, SrcType op2)
  {
    if (!op1.iso(SrcArithmeticType.class))
      throw new SrcRuntimeException(
        "first operand to " + operator + " is not of arithmetic type");
    if (!op2.iso(SrcArithmeticType.class))
      throw new SrcRuntimeException(
        "second operand to " + operator + " is not of arithmetic type");
    for (SrcPrimitiveFloatingType toType
         : new SrcPrimitiveFloatingType[]{SrcLongDoubleType, SrcDoubleType,
                                          SrcFloatType})
      if (op1.iso(toType) || op2.iso(toType))
        return toType;
    final SrcIntegerType
      op1Promoted = op1.toIso(SrcIntegerType.class).integerPromoteNoPrep(),
      op2Promoted = op2.toIso(SrcIntegerType.class).integerPromoteNoPrep();
    if (op1Promoted.eqv(op2Promoted))
      return op1Promoted;
    final boolean op1IsSigned = op1Promoted.isSigned();
    if (op1IsSigned == op2Promoted.isSigned()) {
      if (op1Promoted.getIntegerConversionRank()
          > op2Promoted.getIntegerConversionRank())
        return op1Promoted;
      return op2Promoted;
    }
    final SrcIntegerType signedType = op1IsSigned ? op1Promoted : op2Promoted;
    final SrcIntegerType unsignedType = op1IsSigned ? op2Promoted : op1Promoted;
    if (unsignedType.getIntegerConversionRank()
        >= signedType.getIntegerConversionRank())
      return unsignedType;
    if (signedType.getPosWidth() >= unsignedType.getWidth())
      return signedType;
    return signedType.getSignednessToggle();
  }

  @Override
  public boolean eqvBald(SrcBaldType other) {
    return this == other;
  }

  @Override
  public SrcTypeIterator componentIterator(boolean storageOnly,
                                           Set<SrcType> skipTypes)
  {
    return new SrcTypeComponentIterator(storageOnly, skipTypes);
  }

  @Override
  public abstract ValueAndType unaryPlusNoPrep(
    LLVMValue value, LLVMModule module, LLVMInstructionBuilder builder);

  @Override
  public abstract ValueAndType unaryMinusNoPrep(
    LLVMValue value, LLVMModule module, LLVMInstructionBuilder builder);

  /** This is meant to be called only by {@link SrcType#multiplyNoPrep}. */
  public abstract LLVMValue multiplyNoPrep(
    LLVMValue op1, LLVMValue op2,
    LLVMModule module, LLVMInstructionBuilder builder);
  /** This is meant to be called only by {@link SrcType#divideNoPrep}. */
  public abstract LLVMValue divideNoPrep(
    LLVMValue op1, LLVMValue op2,
    LLVMModule module, LLVMInstructionBuilder builder);
  /** This is meant to be called only by {@link SrcType#remainderNoPrep}. */
  public abstract LLVMValue remainderNoPrep(
    LLVMValue op1, LLVMValue op2,
    LLVMModule module, LLVMInstructionBuilder builder);
  @Override
  public LLVMValue subtractNoPrep(
    LLVMValue op1, LLVMValue op2, LLVMTargetData targetData,
    LLVMModule module, LLVMInstructionBuilder builder)
  {
    return subtractNoPrep(op1, op2, module, builder);
  }
  public abstract LLVMValue subtractNoPrep(
    LLVMValue op1, LLVMValue op2,
    LLVMModule module, LLVMInstructionBuilder builder);
}
