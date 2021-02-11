package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SRC_SIZE_T_TYPE;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import openacc.codegen.llvmBackend.SrcQualifiedType.SrcTypeQualifier;

import org.jllvm.LLVMConstant;
import org.jllvm.LLVMConstantExpression;
import org.jllvm.LLVMContext;
import org.jllvm.LLVMIdentifiedStructType;
import org.jllvm.LLVMInstructionBuilder;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMTargetData;
import org.jllvm.LLVMType;
import org.jllvm.LLVMValue;

/**
 * The LLVM backend's abstract super class for {@link SrcStructType} and
 * {@link SrcUnionType}.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public abstract class SrcStructOrUnionType extends SrcBaldType {
  private final String keyword;
  private final String tag;
  private LLVMIdentifiedStructType llvmIdentifiedStructType;
  protected static class DirectMemberIndex {
    public DirectMemberIndex(int index, boolean isAnonymous) {
      this.index = index;
      this.isAnonymous = isAnonymous;
    }
    public final int index;
    public final boolean isAnonymous;
  }
  // Does not include direct members that are anonymous structures/unions,
  // but does include named members from them.
  private Map<String, DirectMemberIndex> memberNameToIndex = null;
  // As empty strings, includes direct members that are anonymous
  // structures/unions, but does not include members from them.
  private String[] memberNames = null;
  private SrcType[] memberTypes = null;
  private List<SrcCheck> checksDeferredUntilCompletion
    = new ArrayList<>();
  private boolean hasNVMStoredUses = false;

  public SrcStructOrUnionType(String keyword, String tag,
                              LLVMContext llvmContext)
  {
    this.keyword = keyword;
    this.tag = tag;
    this.llvmIdentifiedStructType
      = new LLVMIdentifiedStructType(llvmContext, keyword + "." + tag);
  }

  public final void setBody(String[] memberNames, SrcType[] memberTypes,
                            LLVMTargetData llvmTargetData,
                            LLVMContext context)
  {
    // It shouldn't be possible to have set the body before.
    assert(isIncompleteType());

    // A struct or union type is not considered complete while its members
    // are being declared, so we perform member validation before setting its
    // members and thus before marking it complete.
    boolean hasNVMStoredMember = false;
    for (int i = 0; i < memberNames.length; ++i) {
      if (memberTypes[i].iso(SrcFunctionType.class))
        throw new SrcRuntimeException(keyword+" member has function type");
      // When the struct or union is constructed in BuildLLVM, the size of a
      // flexible array member is set to zero, and so it is not considered to
      // have an incomplete type.
      if (memberTypes[i].isIncompleteType())
        throw new SrcRuntimeException(
          keyword+" member has incomplete type"
          +(eqv(SrcStructType.class) ? " but is not flexible array member"
                                     : ""));
      if (memberTypes[i].hasEffectiveQualifier(SrcTypeQualifier.NVL,
                                               SrcTypeQualifier.NVL_WP))
        hasNVMStoredMember = true;
    }
    // If there's any NVM-stored member type, make sure they all can be
    // NVM-stored.
    if (hasNVMStoredMember) {
      for (int i = 0; i < memberNames.length; ++i) {
        final SrcType memberType = memberTypes[i];
        if (!memberType.hasEffectiveQualifier(SrcTypeQualifier.NVL,
                                              SrcTypeQualifier.NVL_WP))
          SrcQualifiedType.get(memberType, SrcTypeQualifier.NVL);
      }
    }

    // Now set the body and make it a complete type, possibly performing
    // more validation first according to the subclass.
    setValidatedBody(memberNames, memberTypes, llvmTargetData, context);

    // These checks were deferred because the type was incomplete, so run
    // them now that it is complete.
    for (SrcCheck deferredCheck : checksDeferredUntilCompletion)
      deferredCheck.run();
    checksDeferredUntilCompletion.clear();
  }

  protected abstract void setValidatedBody(
    String[] memberNames, SrcType[] memberTypes, LLVMTargetData llvmTargetData,
    LLVMContext context);

  protected final void setValidatedBody(
    String[] memberNames, SrcType[] memberTypes, LLVMType[] memberLLVMTypes)
  {
    assert(memberNames.length == memberTypes.length);
    this.memberNames = memberNames;
    this.memberTypes = memberTypes;
    memberNameToIndex = new HashMap<>(memberNames.length);
    for (int i = 0; i < memberNames.length; ++i) {
      if (!memberNames[i].isEmpty()) {
        if (memberNameToIndex.containsKey(memberNames[i]))
          throw new SrcRuntimeException(
            "member name is ambiguous within "+keyword+": "+memberNames[i]);
        memberNameToIndex.put(memberNames[i],
                              new DirectMemberIndex(i, false));
      }
      else {
        final SrcStructOrUnionType memberStructOrUnionType
          = memberTypes[i].toIso(SrcStructOrUnionType.class);
        // We can't check that there is no tag because Cetus adds a tag
        // when there isn't one. gcc permits this too when you use
        // -fms-extensions.
        if (memberStructOrUnionType == null)
          throw new SrcRuntimeException(
            "unnamed member is not a bitfield, struct, or union");
        for (String memberName : memberStructOrUnionType
                                 .memberNameToIndex.keySet())
        {
          if (memberNameToIndex.containsKey(memberName))
            throw new SrcRuntimeException(
              "member name is ambiguous within "+keyword+": "+memberName);
          memberNameToIndex.put(memberName, new DirectMemberIndex(i, true));
        }
      }
    }
    // ISO C11 sec. 6.7.2.1p8, ISO C99 sec. 6.7.2.1p7.
    if (memberNameToIndex.isEmpty())
      throw new SrcRuntimeException("struct or union has no named members");
    llvmIdentifiedStructType.setBody(false, memberLLVMTypes);
  }

  /** "{@code struct}" or "{@code union}". */
  public String getKeyword() {
    return keyword;
  }
  /** What is the struct or union tag? */
  public String getTag() {
    return tag;
  }
  /**
   * What are the direct member names? Unnamed bit-fields are omitted.
   * Direct members that are anonymous structures/unions are recorded as
   * empty strings. Members from them are omitted.
   */
  public String[] getMemberNames() {
    return memberNames;
  }
  /**
   * What are the direct member types? Unnamed bit-fields are omitted.
   * Direct members that are anonymous structures/unions are included.
   * Members from them are omitted.
   */
  public SrcType[] getMemberTypes() {
    return memberTypes;
  }
  /** Record that this type is sometimes used for NVM storage. */
  public void recordNVMStoredUse() {
    hasNVMStoredUses = true;
  }
  /** Has {@link #recordNVMStoredUse} been called on this type? */
  public boolean hasNVMStoredUses() {
    return hasNVMStoredUses;
  }

  /**
   * Store a check, and run it later when the struct or union type's member
   * list is set.
   * 
   * <p>
   * If the member list is never set, the check will never be run.
   * </p>
   * 
   * @param c the check
   */
  public final void checkAtTypeCompletion(SrcCheck c) {
    checksDeferredUntilCompletion.add(c);
  }

  public abstract ValueAndType buildConstant(List<LLVMConstant> inits,
                                             LLVMModule module,
                                             LLVMInstructionBuilder builder);

  @Override
  public boolean eqvBald(SrcBaldType other) {
    return this == other;
  }

  @Override
  public boolean isIncompleteType() {
    return memberNames == null;
  }

  @Override
  public boolean hasEffectiveQualifier(SrcTypeQualifier... quals) {
    return false;
  }

  @Override
  public SrcStructOrUnionType withoutEffectiveQualifier(
    SrcTypeQualifier... quals)
  {
    return this;
  }

  @Override
  public SrcTypeIterator componentIterator(boolean storageOnly,
                                           Set<SrcType> skipTypes)
  {
    if (isIncompleteType())
      return new SrcCompletableTypeIterator(SrcStructOrUnionType.this);
    if (skipTypes.contains(this))
      return new SrcTypeComponentIterator(storageOnly, skipTypes);
    skipTypes.add(this);
    return new SrcTypeComponentIterator(storageOnly, skipTypes, memberTypes);
  }

  @Override
  public SrcStructOrUnionType buildCompositeBald(
    SrcBaldType other, boolean warn, boolean warningsAsErrors,
    String msgPrefix)
  {
    if (getClass() != other.getClass()) {
      BuildLLVM.warnOrError(warn, warningsAsErrors,
        msgPrefix+": "+keyword+" "+tag+" is incompatible with other type: "
        +other);
      return null;
    }
    if (!eqvBald(other)) {
      BuildLLVM.warnOrError(warn, warningsAsErrors,
        msgPrefix+": "+keyword + " types are incompatible because they are"
        +" not the same "+keyword+": "+this+" and "+other);
      return null;
    }
    return this;
  }

  @Override
  public LLVMIdentifiedStructType getLLVMType(LLVMContext context) {
    return llvmIdentifiedStructType;
  }
  @Override
  public LLVMIdentifiedStructType getLLVMTypeAsPointerTarget(
    LLVMContext context)
  {
    return getLLVMType(context);
  }

  @Override
  public SrcStructOrUnionType defaultArgumentPromoteNoPrep() {
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
  public LLVMValue convertFromNoPrep(ValueAndType from, String operation,
                                     LLVMModule module,
                                     LLVMInstructionBuilder builder)
  {
    // ISO C99 sec. 6.3p2.
    if (eqv(from.getSrcType()))
      return from.getLLVMValue();
    throw new SrcRuntimeException(operation + " requires conversion to <"
                                  + this + "> from a different type");
  }

  private DirectMemberIndex accessDirectMemberIndex(String member) {
    if (isIncompleteType())
      throw new SrcRuntimeException(keyword + " is incomplete");
    final DirectMemberIndex directMemberIndex = memberNameToIndex
                                                .get(member);
    if (directMemberIndex == null)
      throw new SrcRuntimeException(keyword + " has no member named "
                                    + member);
    return directMemberIndex;
  }

  /**
   * Access a member.
   * 
   * @param structOrUnionTypeQualifiers
   *          type qualifiers from the struct or union via which this member
   *          is being accessed
   * @param expr
   *          the {@link LLVMValue} for the struct or union. Its type must be
   *          {@link LLVMIdentifiedStructType} or a pointer to one.
   * @param memberName
   *          the name of the member
   * @param resultIsLvalue
   *          whether the result should be an lvalue. Must be true if and only
   *          if {@code expr} is a pointer. (As for other types in LLVM, a
   *          struct can be stored directly in a register. For example,
   *          {@code expr} might be the temporary returned by a function.)
   * @param module
   *          the LLVM module into which any required declarations should be
   *          generated
   * @param builder
   *          the IR builder with which any non-alloca LLVM instructions
   *          should be generated
   * @param allocaBuilder
   *          the IR builder with which any LLVM alloca instructions should be
   *          generated. Can be null if {@code expr} is an LLVM pointer
   *          (perhaps from a pointer rvalue or a struct or union lvalue) not
   *          a struct or union. This method will not generate any
   *          instructions with {@code allocaBuilder} after generating any
   *          instructions with {@code builder} (see
   *          {@link BuildLLVM.Visitor#getLLVMAllocaBuilder()} comments for
   *          why).
   * @return the value and type for the member
   * @throws SrcRuntimeException
   *           if {@code memberName} does not name a member in this
   */
  public ValueAndType accessMember(
    EnumSet<SrcTypeQualifier> structOrUnionTypeQualifiers, LLVMValue expr,
    String memberName, boolean resultIsLvalue, LLVMModule module,
    LLVMInstructionBuilder builder, LLVMInstructionBuilder allocaBuilder)
  {
    final DirectMemberIndex directMemberIndex
      = accessDirectMemberIndex(memberName);
    final ValueAndType directMember
      = accessMember(structOrUnionTypeQualifiers, expr,
                     directMemberIndex.index, resultIsLvalue, module,
                     builder, allocaBuilder);
    if (!directMemberIndex.isAnonymous)
      return directMember;
    final SrcType directMemberType = directMember.getSrcType();
    final SrcStructOrUnionType directMemberStructOrUnionType
      = directMemberType.toIso(SrcStructOrUnionType.class);
    assert(directMemberStructOrUnionType != null);
    assert(resultIsLvalue == directMember.isLvalue());
    return directMemberStructOrUnionType.accessMember(
      directMemberType.expandSrcTypeQualifiers(),
      directMember.getLLVMValue(), memberName, resultIsLvalue, module,
      builder, allocaBuilder);
  }

  /**
   * Same as
   * {@link #accessMember(LLVMValue, String, boolean, LLVMModule, LLVMInstructionBuilder, LLVMInstructionBuilder)}
   * except the member is specified by index, which must be valid.
   */
  public abstract ValueAndType accessMember(
    EnumSet<SrcTypeQualifier> structOrUnionTypeQualifiers, LLVMValue expr,
    int member, boolean resultIsLvalue, LLVMModule module,
    LLVMInstructionBuilder builder, LLVMInstructionBuilder allocBuilder);

  /**
   * Compute the offset of a member.
   * 
   * @param memberName
   *          the name of the member
   * @param module
   *          the LLVM module into which any required declarations should be
   *          generated
   * @return an rvalue holding the member offset as a {@code size_t}.
   * @throws SrcRuntimeException
   *           if {@code member} does not name a member in this
   */
  public ValueAndType offsetof(String memberName, LLVMModule module) {
    final LLVMContext ctxt = module.getContext();
    SrcStructOrUnionType structOrUnionType = this;
    LLVMConstant offset = null;
    while (true) {
      final DirectMemberIndex directMemberIndex
        = structOrUnionType.accessDirectMemberIndex(memberName);
      final LLVMConstant add
        = structOrUnionType.offsetof(directMemberIndex.index, module);
      assert(add.typeOf() == SRC_SIZE_T_TYPE.getLLVMType(ctxt));
      if (offset == null)
        offset = add;
      else
        offset = LLVMConstantExpression.add(offset, add);
      if (!directMemberIndex.isAnonymous)
        break;
      final SrcType directMemberType
        = structOrUnionType.memberTypes[directMemberIndex.index];
      structOrUnionType
        = directMemberType.toIso(SrcStructOrUnionType.class);
      assert(structOrUnionType != null);
    }
    return new ValueAndType(offset, SRC_SIZE_T_TYPE, false);
  }

  /**
   * Same as {@link #offsetof(String)} except a direct member is specified
   * by index, which must be valid, and the result is an LLVMConstant,
   * whose type must be for {@link #SRC_SIZE_T_TYPE}.
   */
  public abstract LLVMConstant offsetof(int memberIdx, LLVMModule module);

  @Override
  public String toString(String nestedDecl, Set<SrcType> skipTypes) {
    StringBuilder res = new StringBuilder(keyword);
    res.append(" ");
    res.append(tag);
    if (!isIncompleteType() && !skipTypes.contains(this)) {
      skipTypes.add(this);
      res.append(" {");
      for (int i = 0; i < memberNames.length; ++i) {
        if (i > 0) res.append(" ");
        res.append(memberTypes[i].toString(memberNames[i], skipTypes));
        res.append(";");
      }
      res.append("}");
    }
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
    if (isIncompleteType())
      throw new IllegalStateException(
        "toCompatibilityString called on incomplete struct type");
    StringBuilder res = new StringBuilder(keyword);
    res.append(" ");
    res.append(tag);
    if (!structSet.contains(this)) {
      structSet.add(this);
      res.append(" {");
      for (int i = 0; i < memberNames.length; ++i) {
        if (i > 0) res.append(" ");
        res.append(memberTypes[i].toCompatibilityString(structSet, ctxt));
        res.append(" ");
        res.append(memberNames[i]);
        res.append(";");
      }
      res.append("}");
    }
    return res.toString();
  }
}
