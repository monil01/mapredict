package openacc.codegen.llvmBackend;

import java.util.Set;

import org.jllvm.LLVMContext;

/**
 * The LLVM backend's class for all primitive integer types from the C source.
 * 
 * <p>
 * TODO: The mapping here from C integer types to integer sizes is
 * based on the LLVM IR output of clang on x86_64-apple-macosx10.9.0. We
 * should find a way to determine the correct sizes for the target
 * platform.
 * </p>
 * 
 * <p>
 * Integer conversion ranks are based on ISO C99 sec. 6.3.1.1.
 * </p>
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class SrcPrimitiveIntegerType extends SrcIntegerType {
  public static final SrcPrimitiveIntegerType SrcBoolType             = new SrcPrimitiveIntegerType(8,  0,                               null,                    "_Bool");

  /**
   * The type {@code char}. This is distinct from {@link #SrcCharConstType}.
   * 
   * ISO C99 6.2.5p15 says it's implementation-defined whether {@code char}
   * is signed or unsigned.
   * 
   * TODO: If you have to change these bitwidths for some platforms,
   * consider swapping these defintions with the definitions of types like
   * {@link #SRC_UINT64_T_TYPE}. That way, {@link #SRC_UINT64_T_TYPE} is
   * explicitly 64 bits, and {@link #SrcUnsignedLongType} is a
   * platform-specific alias for it.
   */
  public static final SrcPrimitiveIntegerType SrcUnsignedCharType     = new SrcPrimitiveIntegerType(8,  SrcBoolType.intRank + 1,         null,                    "unsigned char");
  public static final SrcPrimitiveIntegerType SrcSignedCharType       = new SrcPrimitiveIntegerType(8,  SrcUnsignedCharType.intRank,     SrcUnsignedCharType,     "signed char");
  public static final SrcPrimitiveIntegerType SrcCharType             = new SrcPrimitiveIntegerType(8,  SrcUnsignedCharType.intRank,     SrcUnsignedCharType,     "char");

  public static final SrcPrimitiveIntegerType SrcUnsignedShortType    = new SrcPrimitiveIntegerType(16, SrcCharType.intRank + 1,         null,                    "unsigned short");
  public static final SrcPrimitiveIntegerType SrcShortType            = new SrcPrimitiveIntegerType(16, SrcUnsignedShortType.intRank,    SrcUnsignedShortType,    "short");

  public static final SrcPrimitiveIntegerType SrcUnsignedIntType      = new SrcPrimitiveIntegerType(32, SrcShortType.intRank + 1,        null,                    "unsigned");
  public static final SrcPrimitiveIntegerType SrcIntType              = new SrcPrimitiveIntegerType(32, SrcUnsignedIntType.intRank,      SrcUnsignedIntType,      "int");

  public static final SrcPrimitiveIntegerType SrcUnsignedLongType     = new SrcPrimitiveIntegerType(64, SrcIntType.intRank + 1,          null,                    "unsigned long");
  public static final SrcPrimitiveIntegerType SrcLongType             = new SrcPrimitiveIntegerType(64, SrcUnsignedLongType.intRank,     SrcUnsignedLongType,     "long");

  public static final SrcPrimitiveIntegerType SrcUnsignedLongLongType = new SrcPrimitiveIntegerType(64, SrcLongType.intRank + 1,         null,                    "unsigned long long");
  public static final SrcPrimitiveIntegerType SrcLongLongType         = new SrcPrimitiveIntegerType(64, SrcUnsignedLongLongType.intRank, SrcUnsignedLongLongType, "long long");

  /**
   * The following types are not new types in C. Each is merely an alias for
   * another type, either via a typedef or as the type of a kind of literal.
   */
  /**
   * The type of an integer character constant. This is distinct from
   * {@link #SrcCharType}.
   */
  public static final SrcPrimitiveIntegerType SRC_CHAR_CONST_TYPE
    = SrcIntType;
  /** wchar_t and the type of wide character constants. */
  public static final SrcPrimitiveIntegerType SRC_WCHAR_TYPE = SrcIntType;
  /**
   * The type of an enumeration constant, which is always int according to ISO
   * C99 6.4.4.3p2 and 6.7.2.2p3. This could potentially be different than
   * {@link SrcEnumType#COMPATIBLE_INTEGER_TYPE}, so we're careful to use the
   * right one.
   */
  public static final SrcPrimitiveIntegerType SRC_ENUM_CONST_TYPE
    = SrcIntType;
  /**
   * TODO: What should the type for size_t be? ISO C99 7.17p2 says size_t is
   * unsigned, and p4 suggests the conversion rank should usually not be
   * larger than long, so we've chosen unsigned long, whose LLVM type is i64.
   * But how do we find what type it actually is in the headers we've parsed?
   * Another possibility for the LLVM type is
   * "postOrderLLVMValues.add(LLVMConstantExpression.getSizeOf(type));" but
   * that assumes i64 anyway, and it generates a complicated gep expression
   * instead of a constant integer (or maybe constant-folding in IRBuilder
   * could handle that?).
   */
  public static final SrcPrimitiveIntegerType SRC_SIZE_T_TYPE
    = SrcUnsignedLongType;
  /** TODO: What should the type for ptrdiff_t be? */
  public static final SrcPrimitiveIntegerType SRC_PTRDIFF_T_TYPE
    = SrcLongType;

  public static final SrcPrimitiveIntegerType SRC_UINT16_T_TYPE
    = SrcUnsignedShortType;
  public static final SrcPrimitiveIntegerType SRC_UINT32_T_TYPE
    = SrcUnsignedIntType;
  public static final SrcPrimitiveIntegerType SRC_UINT64_T_TYPE
    = SrcUnsignedLongType;

  private final int llvmWidth;
  private final int intRank;
  private final boolean signed;
  private SrcPrimitiveIntegerType signednessToggle;
  private final String cSyntax;

  private SrcPrimitiveIntegerType(int llvmWidth, int intRank,
                                  SrcPrimitiveIntegerType unsignedType,
                                  String cSyntax)
  {
    this.llvmWidth = llvmWidth;
    this.intRank = intRank;
    this.signednessToggle = unsignedType;
    this.signed = unsignedType != null;
    if (this.signed)
      unsignedType.signednessToggle = this;
    this.cSyntax = cSyntax;
  }

  @Override
  public boolean checkIntegerCompatibility(
    SrcIntegerType other, boolean warn, boolean warningsAsErrors,
    String msgPrefix)
  {
    if (!other.eqv(SrcPrimitiveIntegerType.class))
      return other.checkIntegerCompatibility(this, warn, warningsAsErrors,
                                             msgPrefix);
    if (!eqvBald(other)) {
      BuildLLVM.warnOrError(warn, warningsAsErrors,
        msgPrefix+": primitive integer types are incompatible because they"
        +" are not the same type: "+this+" and "+other);
      return false;
    }
    return true;
  }

  @Override
  public int getIntegerConversionRank() {
    return intRank;
  }
  @Override
  public boolean isSigned() {
    return signed;
  }
  @Override
  public SrcPrimitiveIntegerType getSignednessToggle() {
    return signednessToggle;
  }
  @Override
  public long getLLVMWidth() {
    return llvmWidth;
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
    StringBuilder res = new StringBuilder(signed ? "si" : "ui");
    res.append(llvmWidth);
    return res.toString();
  }
}
