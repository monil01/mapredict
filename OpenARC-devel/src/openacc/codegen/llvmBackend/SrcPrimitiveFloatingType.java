package openacc.codegen.llvmBackend;

import static org.jllvm.bindings.LLVMTypeKind.LLVMDoubleTypeKind;
import static org.jllvm.bindings.LLVMTypeKind.LLVMFloatTypeKind;
import static org.jllvm.bindings.LLVMTypeKind.LLVMX86_FP80TypeKind;

import java.util.Set;

import org.jllvm.LLVMAddInstruction;
import org.jllvm.LLVMConstant;
import org.jllvm.LLVMConstantReal;
import org.jllvm.LLVMContext;
import org.jllvm.LLVMDivideInstruction;
import org.jllvm.LLVMDivideInstruction.DivisionType;
import org.jllvm.LLVMDoubleType;
import org.jllvm.LLVMExtendCast;
import org.jllvm.LLVMExtendCast.ExtendType;
import org.jllvm.LLVMFloatComparison;
import org.jllvm.LLVMFloatType;
import org.jllvm.LLVMInstructionBuilder;
import org.jllvm.LLVMIntegerToFloatCast;
import org.jllvm.LLVMIntegerToFloatCast.IntCastType;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMMultiplyInstruction;
import org.jllvm.LLVMRealType;
import org.jllvm.LLVMRemainderInstruction;
import org.jllvm.LLVMRemainderInstruction.RemainderType;
import org.jllvm.LLVMSubtractInstruction;
import org.jllvm.LLVMTruncateCast;
import org.jllvm.LLVMTruncateCast.TruncateType;
import org.jllvm.LLVMValue;
import org.jllvm.LLVMX86FP80Type;
import org.jllvm.bindings.LLVMRealPredicate;
import org.jllvm.bindings.LLVMTypeKind;

/**
 * The LLVM backend's class for all primitive floating types from the C
 * source.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public class SrcPrimitiveFloatingType extends SrcArithmeticType {
  public static final SrcPrimitiveFloatingType SrcFloatType      = new SrcPrimitiveFloatingType(LLVMFloatTypeKind,    "float");
  public static final SrcPrimitiveFloatingType SrcDoubleType     = new SrcPrimitiveFloatingType(LLVMDoubleTypeKind,   "double");
  public static final SrcPrimitiveFloatingType SrcLongDoubleType = new SrcPrimitiveFloatingType(LLVMX86_FP80TypeKind, "long double");

  private final LLVMTypeKind llvmTypeKind;
  private final String cSyntax;

  private SrcPrimitiveFloatingType(LLVMTypeKind llvmTypeKind, String cSyntax) {
    this.llvmTypeKind = llvmTypeKind;
    this.cSyntax = cSyntax;
  }

  /** Get the constant value for infinity. */
  public LLVMConstantReal getInfinity(LLVMContext context) {
    // For float, double, and x86_fp80 as long double, this produces the
    // values that clang (3.5.1) produces.
    return LLVMConstantReal.get(getLLVMType(context),
                                Double.POSITIVE_INFINITY);
  }

  @Override
  public boolean isIncompleteType() {
    return false;
  }
  @Override
  public SrcPrimitiveFloatingType buildCompositeBald(
    SrcBaldType other, boolean warn, boolean warningsAsErrors,
    String msgPrefix)
  {
    if (!(other.eqv(SrcPrimitiveFloatingType.class))) {
      BuildLLVM.warnOrError(warn, warningsAsErrors,
        msgPrefix+": floating type is incompatible with non-floating type: "
        +other);
      return null;
    }
    if (!eqvBald(other)) {
      BuildLLVM.warnOrError(warn, warningsAsErrors,
        msgPrefix+": floating types are incompatible because they are not"
        +" the same type: "+this+" and "+other);
      return null;
    }
    return this;
  }

  @Override
  public LLVMRealType getLLVMType(LLVMContext context) {
    if (llvmTypeKind == LLVMFloatTypeKind)
      return LLVMFloatType.get(context);
    if (llvmTypeKind == LLVMDoubleTypeKind)
      return LLVMDoubleType.get(context);
    assert(llvmTypeKind == LLVMX86_FP80TypeKind);
    return LLVMX86FP80Type.get(context);
  }
  @Override
  public LLVMRealType getLLVMTypeAsPointerTarget(LLVMContext context) {
    return getLLVMType(context);
  }

  @Override
  public SrcPrimitiveFloatingType defaultArgumentPromoteNoPrep() {
    if (eqv(SrcFloatType))
      return SrcDoubleType;
    return this;
  }
  @Override
  public LLVMValue defaultArgumentPromoteNoPrep(LLVMValue value,
                                                LLVMModule module,
                                                LLVMInstructionBuilder builder)
  {
    if (eqv(SrcFloatType))
      return SrcDoubleType.convertFromFloatingTypeNoPrep(this, value, module,
                                                         builder);
    return value;
  }

  @Override
  public LLVMValue convertFromNoPrep(ValueAndType from, String operation,
                                     LLVMModule module,
                                     LLVMInstructionBuilder builder)
  {
    final LLVMContext context = module.getContext();
    // ISO C99 sec. 6.3p2.
    if (this == from.getSrcType())
      return from.getLLVMValue();
    // ISO C99 sec. 6.3.1.4p2.
    final SrcIntegerType fromIntegerType
      = from.getSrcType().toIso(SrcIntegerType.class);
    if (fromIntegerType != null) {
      final LLVMValue val = from.getLLVMValue();
      return LLVMIntegerToFloatCast.create(
        builder, ".itofp", val, getLLVMType(context),
        fromIntegerType.isSigned() ? IntCastType.SIGNED
                                   : IntCastType.UNSIGNED);
    }
      final SrcPrimitiveFloatingType fromFloatType
      = from.getSrcType().toIso(SrcPrimitiveFloatingType.class);
    if (fromFloatType != null) {
      final LLVMValue val = from.getLLVMValue();
      return convertFromFloatingTypeNoPrep(fromFloatType, val, module,
                                           builder);
    }
    throw new SrcRuntimeException(operation + " requires conversion from <"
                                  + from.getSrcType() + "> to floating type");
  }

  private LLVMValue convertFromFloatingTypeNoPrep(
    SrcPrimitiveFloatingType fromFloatType, LLVMValue val, LLVMModule module,
    LLVMInstructionBuilder builder)
  {
    final LLVMContext context = module.getContext();
    // ISO C99 sec. 6.3.1.5p1.
    if (getLLVMType(context).getPrimitiveSizeInBits()
        > fromFloatType.getLLVMType(context).getPrimitiveSizeInBits())
      return LLVMExtendCast.create(builder, ".ext", val, getLLVMType(context),
                                   ExtendType.FLOAT);
    // ISO C99 sec. 6.3.1.5p2.
    if (getLLVMType(context).getPrimitiveSizeInBits()
        < fromFloatType.getLLVMType(context).getPrimitiveSizeInBits())
      return LLVMTruncateCast.create(builder, ".trunc", val,
                                     getLLVMType(context), TruncateType.FLOAT);
    return val;
  }

  @Override
  public LLVMValue evalAsCondNoPrep(String operand, LLVMValue value,
                                    LLVMModule module,
                                    LLVMInstructionBuilder builder)
  {
    final LLVMConstant zero
      = LLVMConstant.constNull(getLLVMType(module.getContext()));
    return LLVMFloatComparison.create(
      builder, ".cond", LLVMRealPredicate.LLVMRealUNE, value, zero);
  }

  @Override
  public ValueAndType unaryPlusNoPrep(
    LLVMValue value, LLVMModule module, LLVMInstructionBuilder builder)
  {
    return new ValueAndType(value, this, false);
  }

  @Override
  public ValueAndType unaryMinusNoPrep(LLVMValue value, LLVMModule module,
                                       LLVMInstructionBuilder builder)
  {
    final LLVMConstant zero = LLVMConstant.constNull(value.typeOf());
    return new ValueAndType(
      LLVMSubtractInstruction.create(builder, ".neg", zero, value, true),
      this, false);
  }

  @Override
  public LLVMValue unaryNotI1NoPrep(LLVMValue value, LLVMModule module,
                                    LLVMInstructionBuilder builder)
  {
    final LLVMConstant zero
      = LLVMConstant.constNull(getLLVMType(module.getContext()));
    return LLVMFloatComparison.create(builder, ".notAsI1",
                                      LLVMRealPredicate.LLVMRealOEQ,
                                      value, zero);
  }

  @Override
  public LLVMValue multiplyNoPrep(LLVMValue op1, LLVMValue op2,
                                  LLVMModule module,
                                  LLVMInstructionBuilder builder)
  {
    return LLVMMultiplyInstruction.create(builder, ".mul", op1, op2, true);
  }
  @Override
  public LLVMValue divideNoPrep(LLVMValue op1, LLVMValue op2,
                                LLVMModule module,
                                LLVMInstructionBuilder builder)
  {
    return LLVMDivideInstruction.create(builder, ".div", op1, op2,
                                        DivisionType.FLOAT);
  }
  @Override
  public LLVMValue remainderNoPrep(LLVMValue op1, LLVMValue op2,
                                   LLVMModule module,
                                   LLVMInstructionBuilder builder)
  {
    return LLVMRemainderInstruction.create(builder, ".rem", op1, op2,
                                           RemainderType.FLOAT);
  }
  @Override
  public LLVMValue addNoPrep(LLVMValue op1, LLVMValue op2,
                             LLVMModule module,
                             LLVMInstructionBuilder builder)
  {
    return LLVMAddInstruction.create(builder, ".add", op1, op2, true);
  }
  @Override
  public LLVMValue subtractNoPrep(LLVMValue op1, LLVMValue op2,
                                  LLVMModule module,
                                  LLVMInstructionBuilder builder)
  {
    return LLVMSubtractInstruction.create(builder, ".sub", op1, op2, true);
  }
  @Override
  public LLVMValue relationalI1NoPrep(LLVMValue op1, LLVMValue op2,
                                      boolean greater, boolean equals,
                                      LLVMModule module,
                                      LLVMInstructionBuilder builder)
  {
    final String oper = (greater?"g":"l")+(equals?"e":"t");
    final LLVMRealPredicate pred;
    if (greater) pred = equals ? LLVMRealPredicate.LLVMRealOGE
                               : LLVMRealPredicate.LLVMRealOGT;
    else         pred = equals ? LLVMRealPredicate.LLVMRealOLE
                               : LLVMRealPredicate.LLVMRealOLT;
    return LLVMFloatComparison.create(builder, "."+oper+"AsI1", pred,
                                      op1, op2);
  }
  @Override
  public LLVMValue equalityI1NoPrep(LLVMValue op1, LLVMValue op2,
                                    boolean equals, LLVMModule module,
                                    LLVMInstructionBuilder builder)
  {
    final String oper = equals ? "eq" : "ne";
    final LLVMRealPredicate pred = equals ? LLVMRealPredicate.LLVMRealOEQ
                                          : LLVMRealPredicate.LLVMRealUNE;
    return LLVMFloatComparison.create(builder, "."+oper+"AsI1", pred,
                                      op1, op2);
  }

  @Override
  public String toString(String nestedDecl, Set<SrcType> skipTypes) {
    StringBuilder res = new StringBuilder(cSyntax);
    if (!nestedDecl.isEmpty()) {
      res.append(" ");
      res.append(nestedDecl);
    }
    return res.toString();
  }

  @Override
  public String toCompatibilityString(Set<SrcStructOrUnionType> structSet,
                                      LLVMContext ctxt)
  {
    return getLLVMType(ctxt).toString();
  }
}
