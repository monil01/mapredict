package openacc.codegen.llvmBackend;

import java.util.EnumSet;
import java.util.Set;

import openacc.codegen.llvmBackend.SrcQualifiedType.SrcTypeQualifier;

import org.jllvm.LLVMAndInstruction;
import org.jllvm.LLVMConstantInteger;
import org.jllvm.LLVMContext;
import org.jllvm.LLVMInstructionBuilder;
import org.jllvm.LLVMIntegerType;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMOrInstruction;
import org.jllvm.LLVMShiftInstruction;
import org.jllvm.LLVMShiftInstruction.ShiftType;
import org.jllvm.LLVMSubtractInstruction;
import org.jllvm.LLVMUser;
import org.jllvm.LLVMValue;
import org.jllvm.LLVMXorInstruction;

/**
 * The LLVM backend's class for the types of bit-fields from the C source.
 * 
 * <p>
 * The type of a bit-field is only vaguely defined in ISO C99, which does not
 * permit bit-fields in most places where types are important. For example,
 * you cannot have a pointer to a bit-field because (1) the address operator
 * explicitly disallows a bit-field as an operand, and (2) it is syntactically
 * impossible to declare a pointer type whose target type is a bit-field
 * (also, see ISO C99 sec. 6.7.2.1p8 footnote 106). Thus, some of the
 * operations that are common among C types should never be needed for
 * bit-fields, so the methods here that normally implement those operations
 * should be unreachable and so always throw exceptions.
 * </p>
 * 
 * <p>
 * Despite these differences between a bit-field's type and other C types, our
 * design models a bit-field's type as a sub-type of {@link SrcType} so that
 * we can more easily integrate bit-fields into operations that <em>are</em>
 * needed for them. First, we need to be able to assign to and initialize
 * bit-fields, which requires a type conversion from the assigned value to the
 * {@link SrcBitFieldType}. Second, in all other use cases, a bit-field
 * becomes an rvalue and then undergoes integer promotions from the
 * {@link SrcBitFieldType} to a non-bit-field integer type before any other
 * operations. In order to become an rvalue, a bit-field either is loaded from
 * an lvalue of type {@link SrcBitFieldType} or is accessed in a struct or
 * union rvalue. The next two paragraphs describe how all of these operations
 * handle bit-fields.
 * </p>
 * 
 * <p>
 * Because a bit-field is possibly stored at some offset within a storage unit
 * that is shared with other bit-fields, the bit-field's value must be
 * inserted into or extracted from its storage unit for the sake of the
 * operations mentioned in the previous paragraph. Insertion (in
 * {@link #insertIntoStorageUnit}) is always performed immediately before the
 * bit-field's storage unit is written, either by a store (in {@link #store})
 * or by the creation of a struct or union constant initializer (in
 * {@link SrcStructOrUnionType#buildConstant}). Extraction (in
 * {@link #extractFromStorageUnit}) is always performed immediately after the
 * bit-field's storage unit is read, either by a load (in {@link #load}) or by
 * member access on a struct or union rvalue (in
 * {@link SrcStructOrUnionType#accessMember}).
 * </p>
 * 
 * <p>
 * For any bit-field, the value expected by insertion and the value produced
 * by extraction is an {@link LLVMValue} whose type is the full storage unit
 * type but whose full value, including any bits that normally store other
 * bit-fields, exactly equals just the single bit-field's value. Because
 * insertion and extraction are always performed immediately on a bit-field's
 * storage unit as described above, any rvalue of type {@link SrcBitFieldType}
 * is always represented as such an {@link LLVMValue}. As a result, type
 * conversions (in
 * {@link SrcType#convertFromNoPrep(ValueAndType, String, org.jllvm.LLVMModule, org.jllvm.LLVMInstructionBuilder)}
 * ) performed for assignment, initialization, or integer promotions need not
 * be concerned with insertion and extraction, but they do have to consider
 * that, as for {@link SrcBoolType}, the width of the values that the
 * bit-field can actually represent might be smaller than the width of the
 * LLVM value and type.
 * </p>
 * 
 * <p>
 * Unless a bit-field <it>x</it> appears in a struct (but not a union, which
 * doesn't pack bit-fields) following another bit-field whose storage unit has
 * enough room left to pack <it>x</it> too, we choose the type of <it>x</it>'s
 * storage unit to be <it>x</it>'s declared type disregarding <it>x</it>'s
 * declared width (ISO C99 sec. 6.7.2.1p3 and p10 allow us to make that
 * decision). (TODO: Clang (3.5.1) seems to choose the smallest C integer type
 * that fits the contained bit-fields, but we haven't implemented that.)
 * {@link #getLLVMType} returns a bit-field's storage unit type.
 * </p>
 * 
 * <p>
 * See {@link SrcType}'s documentation for notes about {@link SrcBitFieldType}
 * object uniqueness.
 * </p>
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public class SrcBitFieldType extends SrcIntegerType {
  private final SrcIntegerType storageUnitType;
  private final long width;
  private long highOrderOffset;

  /**
   * Construct a new {@link SrcBitFieldType} from a declaration.
   * 
   * <p>
   * The bit-field's offset and storage unit type are set as if this is the
   * first or only bit-field packed into a storage unit, so a new
   * {@link SrcBitFieldType} must be constructed if that turns out not to be
   * true when the bit-field is added as a member of a {@link SrcStructType}.
   * </p>
   * 
   * @param storageUnitType
   *          the type specified in the bit-field's declaration disregarding
   *          the specified width
   * @param width
   *          the number of bits declared for the bit-field
   */
  public SrcBitFieldType(SrcIntegerType storageUnitType, long width) {
    this.storageUnitType = storageUnitType;
    this.width = width;
    this.highOrderOffset = 0;
  }

  /**
   * Construct a new {@link SrcBitFieldType} based on packing within a struct.
   * 
   * @param storageUnitType
   *          the storage unit type computed for a struct based on the
   *          declared type and the type of any preceding bit-field
   * @param width
   *          the number of bits declared for the bit-field
   * @param highOrderOffset
   *          the number of bits between the bit-field's highest order bit and
   *          its storage unit's highest-order bit
   */
  public SrcBitFieldType(SrcIntegerType storageUnitType, long width,
                         long highOrderOffset)
  {
    this.storageUnitType = storageUnitType;
    this.width = width;
    this.highOrderOffset = highOrderOffset;
  }

  /** Get the storage unit type. */
  public SrcIntegerType getStorageUnitType() {
    return storageUnitType;
  }

  /**
   * Get the number of bits between the bit-field's highest order bit and its
   * storage unit's highest-order bit.
   */
  public long getHighOrderOffset() {
    return highOrderOffset;
  }

  @Override
  public ValueAndType load(EnumSet<SrcTypeQualifier> srcTypeQualifiers,
                           LLVMValue lvalue, LLVMModule module,
                           LLVMInstructionBuilder builder)
  {
    final ValueAndType storageUnit = super.load(srcTypeQualifiers, lvalue,
                                                module, builder);
    assert(storageUnit.getSrcType().eqv(this));
    return new ValueAndType(
      extractFromStorageUnit(storageUnit.getLLVMValue(), module, builder),
      this, false);
  }

  @Override
  public void store(EnumSet<SrcTypeQualifier> srcTypeQualifiers,
                    boolean forInit, LLVMValue lvalue, LLVMValue rvalue,
                    LLVMModule module, LLVMInstructionBuilder builder)
  {
    final ValueAndType storageUnitOld
      = super.load(srcTypeQualifiers, lvalue, module, builder);
    final LLVMUser storageUnitNew
      = insertIntoStorageUnit(storageUnitOld.getLLVMValue(), rvalue, module,
                              builder);
    super.store(srcTypeQualifiers, forInit, lvalue, storageUnitNew, module,
                builder);
  }

  /**
   * Extract a bit-field's value from its storage unit.
   * 
   * @param storageUnit
   *          the storage unit's current bits
   * @param module
   *          the LLVM module into which any required declarations should be
   *          generated
   * @param builder
   *          the IR builder with which any LLVM instructions should be
   *          generated
   * @return an {@link LLVMValue} whose type is the storage unit's type and
   *         whose value equals the bit-field's value
   */
  public LLVMValue extractFromStorageUnit(
    LLVMValue storageUnit, LLVMModule module, LLVMInstructionBuilder builder)
  {
    final LLVMIntegerType lty
      = storageUnitType.getLLVMType(module.getContext());
    final LLVMValue shl = LLVMShiftInstruction.create(
      builder, ".bitfield.shl", ShiftType.SHL, storageUnit,
      LLVMConstantInteger.get(lty, getHighOrderOffset(), false));
    final LLVMValue shr = LLVMShiftInstruction.create(
      builder, ".bitfield.shr",
      isSigned() ? ShiftType.ARITHMETIC_SHR : ShiftType.LOGICAL_SHR, shl,
      LLVMConstantInteger.get(lty, getLLVMWidth()-width, false));
    return shr;
  }

  /**
   * Insert a new value for a bit-field into its storage unit.
   * 
   * @param storageUnit
   *          the storage unit's current bits
   * @param value
   *          an {@link LLVMValue} whose type is the storage unit's type and
   *          whose value equals the bit-field's new value. Actually, if the
   *          bit-field is <it>n</it> bits wide, all but the lowest-order
   *          <it>n</it> bits in {@code value} are discarded.
   * @param module
   *          the LLVM module into which any required declarations should be
   *          generated
   * @param builder
   *          the IR builder with which any LLVM instructions should be
   *          generated
   * @return the storage unit's new bits
   */
  public LLVMUser insertIntoStorageUnit(
    LLVMValue storageUnit, LLVMValue value,
    LLVMModule module, LLVMInstructionBuilder builder)
  {
    final LLVMContext ctxt = module.getContext();
    final LLVMIntegerType lty = storageUnitType.getLLVMType(ctxt);
    final LLVMConstantInteger
      one     = LLVMConstantInteger.get(lty, 1,      false),
      ones    = LLVMConstantInteger.allOnes(lty),
      width_  = LLVMConstantInteger.get(lty, width, false),
      lowOff  = LLVMConstantInteger.get(lty,
                                        getLLVMWidth()-highOrderOffset-width,
                                        false);
    final LLVMUser mask;
    {
      LLVMUser v;
      v = LLVMShiftInstruction.create(builder, "", ShiftType.SHL, one, width_);
      v = LLVMSubtractInstruction.create(builder, "", v, one, false);
      v = LLVMShiftInstruction.create(builder, "", ShiftType.SHL, v, lowOff);
      mask = v;
    }
    final LLVMUser notMask = LLVMXorInstruction.create(builder, "", mask, ones);
    final LLVMUser valueMasked;
    {
      LLVMUser v;
      v = LLVMShiftInstruction.create(builder, ".bitfield.shl", ShiftType.SHL,
                                      value, lowOff);
      v = LLVMAndInstruction.create(builder, ".bitfield.mask", v, mask);
      valueMasked = v;
    }
    final LLVMUser storageUnitMasked
      = LLVMAndInstruction.create(builder, ".bitfield.load.mask", storageUnit,
                                  notMask);
    return LLVMOrInstruction.create(builder, ".bitfield.or", storageUnitMasked,
                                    valueMasked);
  }

  /**
   * ISO C99 sec. 6.2.7p1 is the only specification of compatibility between
   * bit-fields, but it only talks about bit-fields from structs or unions
   * from separate translation units, and we haven't implemented such
   * compatibility checks, as discussed in {@link SrcType#checkCompatibility}'s
   * documentation. Within a single translation unit, it appears to be
   * impossible to ever need to check compatibility of a bit-field's type with
   * another type, so this method should be unreachable and thus it always
   * throws an exception.
   */
  @Override
  public boolean checkIntegerCompatibility(
    SrcIntegerType other, boolean warn, boolean warningsAsErrors,
    String msgPrefix)
  {
    throw new IllegalStateException(
      "attempt to compute type compatibility with bit-field");
  }

  /**
   * ISO C99 does not define an integer conversion rank for bit-fields. This
   * method should be unreachable because bit-fields appear to always undergo
   * integer promotions before having their ranks checked. Thus, this method
   * always throws an exception.
   */
  @Override
  public int getIntegerConversionRank() {
    throw new IllegalStateException(
      "attempt to compute integer conversion rank of bit-field");
  }

  /**
   * ISO C99 sec. 6.7.2p5 and 6.7.2.1p9 footnote 107 say we get to choose
   * whether this is a signed int or an unsigned int when the declared type is
   * int, so we make the most obvious and consistent choice: signed int, as in
   * the rest of C.
   */
  @Override
  public boolean isSigned() {
    return storageUnitType.isSigned();
  }

  @Override
  public SrcIntegerType getSignednessToggle() {
    return storageUnitType.getSignednessToggle();
  }
  @Override
  public long getWidth() {
    return width;
  }
  @Override
  public long getLLVMWidth() {
    return storageUnitType.getLLVMWidth();
  }

  @Override
  public String toString(String nestedDecl, Set<SrcType> skipTypes) {
    return storageUnitType.toString(nestedDecl + " : " + width, skipTypes);
  }

  @Override
  public String toCompatibilityString(Set<SrcStructOrUnionType> structSet,
                                      LLVMContext ctxt)
  {
    final StringBuilder res = new StringBuilder(
      storageUnitType.toCompatibilityString(structSet, ctxt));
    res.append(" : ");
    res.append(width);
    return res.toString();
  }
}
