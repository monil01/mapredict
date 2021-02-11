package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SRC_PTRDIFF_T_TYPE;
import static openacc.codegen.llvmBackend.SrcPrimitiveNonNumericType.SrcVoidType;

import java.lang.ref.WeakReference;
import java.util.HashMap;
import java.util.Set;

import openacc.codegen.llvmBackend.SrcQualifiedType.SrcTypeQualifier;

import org.jllvm.LLVMBitCast;
import org.jllvm.LLVMCallInstruction;
import org.jllvm.LLVMConstant;
import org.jllvm.LLVMConstantInteger;
import org.jllvm.LLVMContext;
import org.jllvm.LLVMDivideInstruction;
import org.jllvm.LLVMDivideInstruction.DivisionType;
import org.jllvm.LLVMExtendCast;
import org.jllvm.LLVMExtendCast.ExtendType;
import org.jllvm.LLVMFunction;
import org.jllvm.LLVMFunctionType;
import org.jllvm.LLVMGetElementPointerInstruction;
import org.jllvm.LLVMInstructionBuilder;
import org.jllvm.LLVMIntegerComparison;
import org.jllvm.LLVMIntegerType;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMPointerType;
import org.jllvm.LLVMPtrIntCast;
import org.jllvm.LLVMPtrIntCast.PtrIntCastType;
import org.jllvm.LLVMSubtractInstruction;
import org.jllvm.LLVMTargetData;
import org.jllvm.LLVMTruncateCast;
import org.jllvm.LLVMTruncateCast.TruncateType;
import org.jllvm.LLVMUser;
import org.jllvm.LLVMValue;
import org.jllvm.LLVMVoidType;
import org.jllvm.bindings.LLVMIntPredicate;

/**
 * The LLVM backend's class for all pointer types from the C source.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class SrcPointerType extends SrcScalarType {
  /**
   * The address spaces used here must be kept in sync with OpenARC's
   * make.header (LLVM_TARGET_DATA_LAYOUT specifies pointer sizes), NVL's
   * intrinsic definitions (IntrinsicsNVL.td) and the LLVM pass (NVL.cpp).
   */
  public static final int LLVM_ADDRSPACE_DEFAULT = 0;
  public static final int LLVM_ADDRSPACE_NVL     = 1;
  public static final int LLVM_ADDRSPACE_NVL_WP  = 2;

  private static final HashMap<SrcType, WeakReference<SrcPointerType>> cache
    = new HashMap<>();

  private final SrcType targetType;

  private SrcPointerType(SrcType targetType) {
    assert(targetType != null); // otherwise, hashCode needs updating
    this.targetType = targetType;
  }

  /**
   * Get the specified pointer type.
   * 
   * <p>
   * After this method has been called once for a particular type of pointer,
   * it is guaranteed to return the same object for the same type of pointer
   * until that object can be garbage-collected because it is no longer
   * referenced outside this class's internal cache.
   * </p>
   * 
   * @param targetType
   *          the pointer type's target type
   * @return the specified pointer type
   */
  public static SrcPointerType get(SrcType targetType) {
    WeakReference<SrcPointerType> ref;
    synchronized (cache) {
      ref = cache.get(targetType);
    }
    SrcPointerType type;
    if (ref == null || (type = ref.get()) == null) {
      type = new SrcPointerType(targetType);
      ref = new WeakReference<>(type);
      synchronized (cache) {
        cache.put(targetType, ref);
      }
    }
    return type;
  }
  @Override
  protected void finalize() {
    synchronized (cache) {
      if (cache.get(targetType).get() == null)
        cache.remove(targetType);
    }
  }

  /** Get the pointer type's target type. */
  public SrcType getTargetType() {
    return targetType;
  }

  @Override
  public boolean eqvBald(SrcBaldType other) {
    // Optimize for trivial eqv relation.
    if (this == other)
      return true;
    if (!(other instanceof SrcPointerType))
      return false;
    final SrcPointerType otherPointerType = (SrcPointerType)other;
    return targetType.eqv(otherPointerType.targetType);
  }

  @Override
  public boolean isIncompleteType() {
    return false;
  }

  @Override
  public SrcTypeIterator componentIterator(boolean storageOnly,
                                           Set<SrcType> skipTypes)
  {
    if (storageOnly)
      return new SrcTypeComponentIterator(storageOnly, skipTypes);
    return new SrcTypeComponentIterator(storageOnly, skipTypes,
                                        new SrcType[]{targetType});
  }

  @Override
  public SrcPointerType buildCompositeBald(
    SrcBaldType other, boolean warn, boolean warningsAsErrors,
    String msgPrefix)
  {
    // ISO C99 6.7.5.1p2.
    if (this == other)
      return this;
    final SrcPointerType otherPtr = other.toEqv(SrcPointerType.class);
    if (otherPtr == null) {
      BuildLLVM.warnOrError(warn, warningsAsErrors,
        msgPrefix+": pointer type is incompatible with non-pointer type: "
        +other);
      return null;
    }
    final SrcType compositeTargetType = targetType.buildComposite(
      otherPtr.targetType, warn, warningsAsErrors,
      msgPrefix+": pointer types are incompatible because their target"
      +" types are incompatible");
    if (compositeTargetType == null)
      return null;
    final SrcPointerType res = SrcPointerType.get(compositeTargetType);
    return res.eqv(this) ? this : res.eqv(otherPtr) ? otherPtr : res;
  }

  @Override
  public LLVMPointerType getLLVMType(LLVMContext context) {
    final int addrspace;
    if (targetType.hasEffectiveQualifier(SrcTypeQualifier.NVL))
      addrspace = LLVM_ADDRSPACE_NVL;
    else if (targetType.hasEffectiveQualifier(SrcTypeQualifier.NVL_WP))
      addrspace = LLVM_ADDRSPACE_NVL_WP;
    else
      addrspace = LLVM_ADDRSPACE_DEFAULT;
    return LLVMPointerType.get(targetType.getLLVMTypeAsPointerTarget(context),
                               addrspace);
  }
  @Override
  public LLVMPointerType getLLVMTypeAsPointerTarget(LLVMContext context) {
    return getLLVMType(context);
  }

  @Override
  public SrcType defaultArgumentPromoteNoPrep() {
    return this;
  }
  @Override
  public LLVMValue defaultArgumentPromoteNoPrep(LLVMValue value,
                                                LLVMModule module,
                                                LLVMInstructionBuilder builder)
  {
    return value;
  }

  @Override
  public LLVMValue convertFromNoPrep(
    ValueAndType from, final String operation, LLVMModule module,
    LLVMInstructionBuilder builder)
  {
    final LLVMContext context = module.getContext();

    // Handle null pointer constant.
    if (from.isNullPointerConstant())
      return LLVMConstant.constNull(getLLVMType(context));

    // Handle pointer types. Explicit pointer casts appear to be
    // unconstrained, according to ISO C99 sec. 6.5.4p3.
    final SrcPointerType fromPtr = from.getSrcType()
                                   .toIso(SrcPointerType.class);
    if (fromPtr != null) {
      // The first NVL check overlaps but is clearer than the second one in the
      // case of an array with an NVM-stored element type. In other cases, the
      // first NVL check extends the second one to examine top-level NVM
      // storage of the target type.
      if (targetType.hasEffectiveQualifier(SrcTypeQualifier.NVL)
          != fromPtr.targetType.hasEffectiveQualifier(SrcTypeQualifier.NVL)
          || targetType.hasEffectiveQualifier(SrcTypeQualifier.NVL_WP)
             != fromPtr.targetType
                .hasEffectiveQualifier(SrcTypeQualifier.NVL_WP))
        throw new SrcRuntimeException(
          operation+" changes NVM storage of pointer target type");
      new SrcTypesInvolveNoNVMOrUnqualCompatCheck(
        targetType, fromPtr.targetType,
        operation+" requires conversion between pointer types with target"
        +" types that involve NVM storage and that have incompatible"
        +" unqualified versions")
      .run();
      final LLVMValue val = from.getLLVMValue();
      if (getLLVMType(context) == from.getSrcType().getLLVMType(context))
        return val;
      return LLVMBitCast.create(builder, ".ptr2ptr", val,
                                getLLVMType(context));
    }

    // Handle integer to pointer conversion.
    if (from.getSrcType().iso(SrcIntegerType.class)) {
      new SrcTypeInvolvesNoNVMCheck(
        targetType,
        operation+" requires conversion from integer type to pointer type"
        +" involving NVM storage")
      .run();
      return LLVMPtrIntCast.create(builder, ".int2ptr", from.getLLVMValue(),
                                   getLLVMType(context),
                                   PtrIntCastType.INT_TO_PTR);
    }

    throw new SrcRuntimeException(operation + " requires conversion from <"
                                  + from.getSrcType() + "> to pointer type");
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
  public ValueAndType indirectionNoPrep(String operand, LLVMValue value,
                                        LLVMModule module,
                                        LLVMInstructionBuilder builder)
  {
    return new ValueAndType(value, targetType, true);
  }

  @Override
  public LLVMValue unaryNotI1NoPrep(LLVMValue value, LLVMModule module,
                                    LLVMInstructionBuilder builder)
  {
    final LLVMConstant zero
      = LLVMConstant.constNull(getLLVMType(module.getContext()));
    return LLVMIntegerComparison.create(builder, ".notAsI1",
                                        LLVMIntPredicate.LLVMIntEQ, value,
                                        zero);
  }

  @Override
  public LLVMValue addNoPrep(LLVMValue op1, LLVMValue op2,
                             LLVMModule module,
                             LLVMInstructionBuilder builder)
  {
    final boolean op1IsPtr = op1.typeOf() instanceof LLVMPointerType;
    final LLVMValue ptr = op1IsPtr ? op1 : op2;
    final LLVMValue idx = op1IsPtr ? op2 : op1;
    assert(idx.typeOf() instanceof LLVMIntegerType);
    return LLVMGetElementPointerInstruction.create(builder, ".add", ptr, idx);
  }

  @Override
  public LLVMValue subtractNoPrep(LLVMValue op1, LLVMValue op2,
                                  LLVMTargetData targetData,
                                  LLVMModule module,
                                  LLVMInstructionBuilder builder)
  {
    assert(op1.typeOf() instanceof LLVMPointerType);
    if (op2.typeOf() instanceof LLVMIntegerType) {
      final LLVMValue op2neg
        = LLVMSubtractInstruction.create(builder, ".neg",
                                         LLVMConstant.constNull(op2.typeOf()),
                                         op2, false);
      return LLVMGetElementPointerInstruction.create(builder, ".sub", op1,
                                                     op2neg);
    }
    final LLVMContext ctxt = module.getContext();
    final LLVMPointerType op1PtrType = (LLVMPointerType)op1.typeOf();
    final LLVMPointerType op2PtrType = (LLVMPointerType)op2.typeOf();
    final long addrspace = op1PtrType.getAddressSpace();
    // TODO: Currently this assertion can fail. We need to extend the
    // front end not to allow subtraction across address spaces.
    assert(addrspace == op2PtrType.getAddressSpace());
    if (isNVLAddrspace(addrspace)) {
      if (!(op1 instanceof LLVMConstant) || !(op2 instanceof LLVMConstant))
      {
        final String ptrKind;
        switch ((int)addrspace) {
        case LLVM_ADDRSPACE_NVL:    ptrKind = "v2nv"; break;
        case LLVM_ADDRSPACE_NVL_WP: ptrKind = "v2wp"; break;
        default: throw new IllegalStateException();
        }
        final String intrName = "llvm.nvl.check.heapAlloc."+ptrKind;
        final LLVMPointerType opVoidPtrType = LLVMPointerType.get(
          SrcVoidType.getLLVMTypeAsPointerTarget(ctxt), addrspace);
        LLVMFunction intr = module.getNamedFunction(intrName);
        if (intr.getInstance() == null)
          intr = new LLVMFunction(
            module, intrName,
            LLVMFunctionType.get(LLVMVoidType.get(ctxt), false,
                                 opVoidPtrType, opVoidPtrType));
        new LLVMCallInstruction(
          builder, "", intr,
          LLVMBitCast.create(builder, ".sub.ptrOp1.toVoidPtr", op1,
                             opVoidPtrType),
          LLVMBitCast.create(builder, ".sub.ptrOp2.toVoidPtr", op2,
                             opVoidPtrType));
      }
    }
    final LLVMIntegerType ptrdiffTType = SRC_PTRDIFF_T_TYPE.getLLVMType(ctxt);
    final LLVMIntegerType intPtrType = targetData.intPtrType(ctxt);
    final LLVMConstant targetTypeSize
      = LLVMConstantInteger.get(
          intPtrType, targetData.abiSizeOfType(targetType.getLLVMType(ctxt)),
          true);
    final LLVMUser op1AsInt
      = LLVMPtrIntCast.create(builder, ".sub.ptr.op1.toInt", op1, intPtrType,
                              PtrIntCastType.PTR_TO_INT);
    final LLVMUser op2AsInt
      = LLVMPtrIntCast.create(builder, ".sub.ptr.op2.toInt", op2, intPtrType,
                              PtrIntCastType.PTR_TO_INT);
    final LLVMUser sub
      = LLVMSubtractInstruction.create(builder, ".sub.ptr.sub", op1AsInt,
                                       op2AsInt, false);
    final LLVMUser div
      = LLVMDivideInstruction.create(builder, ".sub.ptr.div", sub,
                                     targetTypeSize, DivisionType.SIGNEDINT);
    if (intPtrType.getWidth() == ptrdiffTType.getWidth())
      return div;
    else if (intPtrType.getWidth() < ptrdiffTType.getWidth())
      return LLVMExtendCast.create(builder, ".2ptrdifft", div, ptrdiffTType,
                                   ExtendType.SIGN);
    else
      return LLVMTruncateCast.create(builder, ".2ptrdifft", div, ptrdiffTType,
                                     TruncateType.INTEGER);
  }
  @Override
  public LLVMValue relationalI1NoPrep(LLVMValue op1, LLVMValue op2,
                                      boolean greater, boolean equals,
                                      LLVMModule module,
                                      LLVMInstructionBuilder builder)
  {
    final String oper = (greater?"g":"l")+(equals?"e":"t");
    final LLVMIntPredicate pred;
    if (greater) pred = equals ? LLVMIntPredicate.LLVMIntUGE
                               : LLVMIntPredicate.LLVMIntUGT;
    else         pred = equals ? LLVMIntPredicate.LLVMIntULE
                               : LLVMIntPredicate.LLVMIntULT;
    return LLVMIntegerComparison.create(builder, "."+oper+".asI1", pred,
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
    return LLVMIntegerComparison.create(builder, "."+oper+".asI1", pred,
                                        op1, op2);
  }

  private static boolean isNVLAddrspace(long addrspace) {
    return addrspace == LLVM_ADDRSPACE_NVL
           || addrspace == LLVM_ADDRSPACE_NVL_WP;
  }

  @Override
  public String toString(String nestedDecl, Set<SrcType> skipTypes) {
    StringBuilder str = new StringBuilder();
    final boolean parens = targetType instanceof SrcArrayType
                           || targetType instanceof SrcFunctionType;
    if (parens) str.append("(");
    str.append("*");
    str.append(nestedDecl);
    if (parens) str.append(")");
    return targetType.toString(str.toString(), skipTypes);
  }

  @Override
  public String toCompatibilityString(Set<SrcStructOrUnionType> structSet,
                                      LLVMContext ctxt)
  {
    StringBuilder res = new StringBuilder(
      targetType.toCompatibilityString(structSet, ctxt));
    res.append(" *");
    return res.toString();
  }
}
