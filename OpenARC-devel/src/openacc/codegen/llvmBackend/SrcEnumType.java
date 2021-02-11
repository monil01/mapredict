package openacc.codegen.llvmBackend;

import java.util.HashMap;
import java.util.Set;

import org.jllvm.LLVMConstant;
import org.jllvm.LLVMContext;

/**
 * The LLVM backend's class for all enumerated types from the C source.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class SrcEnumType extends SrcIntegerType {
  /**
   * The compatible integer type for an enumerated type is
   * implementation-defined according to ISO C99 6.7.2.2p4, so for now we
   * choose to mimic clang. This could potentially be different than
   * {@link SrcPrimitiveIntegerType#SRC_ENUM_CONST_TYPE}, so we're careful to
   * use the right one. TODO: Actually, clang (3.5.1) uses unsigned int if
   * there are no negative enumerator initializers, so maybe we should
   * consider selecting the compatible integer type based on the enumerators.
   */
  public static final SrcPrimitiveIntegerType COMPATIBLE_INTEGER_TYPE
    = SrcPrimitiveIntegerType.SrcIntType;

  private final String tag;
  private final HashMap<String, LLVMConstant> memberHash;
  private final String memberArray[];
  private int nextMember;

  /**
   * Construct a new {@link SrcEnumType}.
   * 
   * @param tag
   *          the enum tag
   * @param memberCount
   *          the number of members
   */
  public SrcEnumType(String tag, int memberCount) {
    this.tag = tag;
    this.memberHash = new HashMap<>(memberCount);
    this.memberArray = new String[memberCount];
    this.nextMember = 0;
  }

  /**
   * Add an enum member.
   * 
   * @param name
   *          the name of the enum member
   * @param value
   *          the value of the enum member
   */
  public void addMember(String name, LLVMConstant value) {
    assert(nextMember < memberArray.length);
    memberHash.put(name, value);
    memberArray[nextMember++] = name;
  }

  /**
   * Get an enum member's value.
   * 
   * @param name
   *          the name of the enum member
   * @return the value of the enum member
   */
  public LLVMConstant getMember(String name) {
    return memberHash.get(name);
  }

  /**
   * Get the enum tag.
   * 
   * @return the enum tag
   */
  public String getTag() {
    return tag;
  }

  @Override
  public boolean checkIntegerCompatibility(
    SrcIntegerType other, boolean warn, boolean warningsAsErrors,
    String msgPrefix)
  {
    // ISO C99 sec. 6.7.2.2p4.
    if (!other.eqv(SrcEnumType.class)) {
      if (!other.eqv(COMPATIBLE_INTEGER_TYPE)) {
        BuildLLVM.warnOrError(warn, warningsAsErrors,
          msgPrefix+": "+this+" is incompatible with "+other);
        return false;
      }
      return true;
    }
    // Type compatibility is apparently not transitive: just because two enum
    // types have the same compatible integer type does not mean they are
    // compatible. At least that's how clang (3.5.1) and gcc (4.2.1) behave
    // when assigning pointers whose target types are enums.
    if (!eqvBald(other)) {
      BuildLLVM.warnOrError(warn, warningsAsErrors,
        msgPrefix+": enum types are incompatible because they are not the"
        +" same enum: "+this+" and "+other);
      return false;
    }
    return true;
  }

  @Override
  public int getIntegerConversionRank() {
    return COMPATIBLE_INTEGER_TYPE.getIntegerConversionRank();
  }
  @Override
  public boolean isSigned() {
    return COMPATIBLE_INTEGER_TYPE.isSigned();
  }
  @Override
  public SrcIntegerType getSignednessToggle() {
    return COMPATIBLE_INTEGER_TYPE.getSignednessToggle();
  }
  @Override
  public long getLLVMWidth() {
    return COMPATIBLE_INTEGER_TYPE.getLLVMWidth();
  }

  @Override
  public String toString(String nestedDecl, Set<SrcType> skipTypes) {
    StringBuilder res = new StringBuilder("enum ");
    res.append(tag);
    res.append(" {");
    for (int i = 0; i < memberArray.length; ++i) {
      if (i > 0) res.append(", ");
      final String memberName = memberArray[i];
      // We check for null in case this method is called for debugging
      // before addMember has been called for every member.
      res.append(memberName == null ? "?" : memberName);
      res.append("=<");
      res.append(memberName == null ? "?" : memberHash.get(memberName));
      res.append(">");
    }
    res.append("}");
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
    final StringBuilder res = new StringBuilder("enum ");
    res.append(tag);
    res.append(" <");
    res.append(COMPATIBLE_INTEGER_TYPE.toCompatibilityString(structSet,
                                                             ctxt));
    res.append("> {");
    for (int i = 0; i < memberArray.length; ++i) {
      if (i > 0) res.append(", ");
      // This method is not for debugging the compiler, so memberName
      // shouldn't be null by the time this method is called.
      final String memberName = memberArray[i];
      res.append(memberName);
      res.append("=<");
      res.append(memberHash.get(memberName));
      res.append(">");
    }
    res.append("}");
    return res.toString();
  }
}
