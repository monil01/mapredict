package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SRC_SIZE_T_TYPE;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

import openacc.codegen.llvmBackend.SrcQualifiedType.SrcTypeQualifier;

import org.jllvm.LLVMArrayType;
import org.jllvm.LLVMBitCast;
import org.jllvm.LLVMConstant;
import org.jllvm.LLVMConstantStruct;
import org.jllvm.LLVMContext;
import org.jllvm.LLVMIdentifiedStructType;
import org.jllvm.LLVMInstructionBuilder;
import org.jllvm.LLVMIntegerType;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMPointerType;
import org.jllvm.LLVMStackAllocation;
import org.jllvm.LLVMStoreInstruction;
import org.jllvm.LLVMTargetData;
import org.jllvm.LLVMType;
import org.jllvm.LLVMUser;
import org.jllvm.LLVMValue;

/**
 * The LLVM backend's class for all union types from the C source.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class SrcUnionType extends SrcStructOrUnionType {
  /**
   * Get the LLVM type used to pad unions.
   */
  public static final LLVMIntegerType getPadLLVMType(LLVMContext context) {
    return LLVMIntegerType.get(context, 8);
  }

  public SrcUnionType(String tag, LLVMContext llvmContext) {
    super("union", tag, llvmContext);
  }

  @Override
  protected void setValidatedBody(String[] memberNames, SrcType[] memberTypes,
                                  LLVMTargetData llvmTargetData,
                                  LLVMContext context)
  {
    // Check NVL constraints.
    for (SrcType memberType : memberTypes)
      new SrcTypeInvolvesNoNVMCheck(memberType, "union member type refers"
                                                +" to NVM storage")
      .run();

    // Remove unnamed bit-fields. We keep direct members that are C11
    // anonymous structures/unions.
    final List<String> memberNamesPrunedList = new ArrayList<>();
    final List<SrcType> memberTypesPrunedList = new ArrayList<>();
    for (int i = 0; i < memberTypes.length; ++i) {
      if (memberNames[i].isEmpty()
          && memberTypes[i].iso(SrcBitFieldType.class))
        continue;
      memberNamesPrunedList.add(memberNames[i]);
      memberTypesPrunedList.add(memberTypes[i]);
    }
    final String[] memberNamesPruned
      = new String[memberNamesPrunedList.size()];
    final SrcType[] memberTypesPruned
      = new SrcType[memberTypesPrunedList.size()];
    memberNamesPrunedList.toArray(memberNamesPruned);
    memberTypesPrunedList.toArray(memberTypesPruned);

    // Compute the largest member and the largest alignment in order to compute
    // the size of the union. clang (clang-600.0.56) uses the largest member
    // as the first member of the generated LLVM type, but we use the first
    // member from the source type as the first member of the generated LLVM
    // type so it's easy to generate constant-expression initializers.
    // TODO: That is, until we support designated initializers. See related
    // todo in BuildLLVM's visitor for Initializer nodes.
    final LLVMType member0Type = memberTypesPruned[0].getLLVMType(context);
    final long member0Size = llvmTargetData.storeSizeOfType(member0Type);
    long maxMemberSize = member0Size;
    long maxMemberAlign
      = llvmTargetData.preferredAlignmentOfType(member0Type);
    for (int i = 1; i < memberTypesPruned.length; ++i) {
      final LLVMType memberType = memberTypesPruned[i].getLLVMType(context);
      final long memberSize = llvmTargetData.storeSizeOfType(memberType);
      final long memberAlign
        = llvmTargetData.preferredAlignmentOfType(memberType);
      if (maxMemberSize < memberSize)
        maxMemberSize = memberSize;
      if (maxMemberAlign < memberAlign)
        maxMemberAlign = memberAlign;
    }
    final long size = (maxMemberSize+maxMemberAlign-1)/maxMemberAlign
                      * maxMemberAlign;
    final long padding = size- member0Size;
    if (padding > 0)
      setValidatedBody(
        memberNamesPruned, memberTypesPruned,
        new LLVMType[]{member0Type,
                       LLVMArrayType.get(getPadLLVMType(context),
                                         padding)});
    else
      setValidatedBody(memberNamesPruned, memberTypesPruned,
                       new LLVMType[]{member0Type});
  }

  @Override
  public ValueAndType buildConstant(List<LLVMConstant> inits,
                                    LLVMModule module,
                                    LLVMInstructionBuilder builder)
  {
    final LLVMContext ctxt = module.getContext();
    assert(inits.size() == 1);
    final LLVMConstant[] initsPacked = new LLVMConstant[1];
    final LLVMConstant init = inits.get(0);
    final SrcBitFieldType bitFieldType
      = getMemberTypes()[0].toIso(SrcBitFieldType.class);
    if (bitFieldType != null) {
      initsPacked[0]
        = LLVMConstant.constNull(getMemberTypes()[0].getLLVMType(ctxt));
      initsPacked[0]
        = (LLVMConstant)bitFieldType.insertIntoStorageUnit(
            initsPacked[0], init, module, builder);
    }
    else
      initsPacked[0] = init;
    return new ValueAndType(
      LLVMConstantStruct.get(getLLVMType(ctxt), initsPacked), this, false);
  }

  @Override
  public ValueAndType accessMember(
    EnumSet<SrcTypeQualifier> structOrUnionTypeQualifiers,
    LLVMValue expr, int member, boolean resultIsLvalue, LLVMModule module,
    LLVMInstructionBuilder builder, LLVMInstructionBuilder allocaBuilder)
  {
    final LLVMContext ctxt = module.getContext();
    final SrcType memberType = getMemberTypes()[member];
    final LLVMType memberLLVMType = memberType.getLLVMType(ctxt);
    final LLVMPointerType memberPtrType = LLVMPointerType.get(memberLLVMType,
                                                              0);
    // LLVM folds a constant-expression bitcast into a gep if it accesses the
    // first element of an LLVM struct. It doesn't do any such thing for the
    // non-constant-expression bitcast.
    if (expr.typeOf() instanceof LLVMPointerType) {
      assert(resultIsLvalue);
      assert(((LLVMPointerType)expr.typeOf()).getElementType()
             instanceof LLVMIdentifiedStructType);
      return new ValueAndType(
        LLVMBitCast.create(builder, ".unionMember", expr, memberPtrType),
        SrcQualifiedType.get(memberType, structOrUnionTypeQualifiers), true);
    }
    assert(!resultIsLvalue);
    assert(expr.typeOf() instanceof LLVMIdentifiedStructType);
    // The only rvalue union in C that I'm aware of is the result of a
    // function, but that's never a constant expression. If it were possible
    // to have a constant-expression rvalue union in C, I would have to
    // either access its member with a non-constant expression or I would
    // have to find a substitute for generating the following alloca, such
    // as generating a global variable, if that would make sense in whatever
    // context it would be.
    assert(!(expr instanceof LLVMConstant));
    // This alloca is required because we cannot bitcast a union to its
    // member unless we have the union's address. clang (clang-600.0.56)
    // does something similar.
    final LLVMUser alloca
      = new LLVMStackAllocation(allocaBuilder, ".unionTmp",
                                getLLVMType(ctxt), null);
    new LLVMStoreInstruction(builder, expr, alloca);
    final LLVMUser cast = LLVMBitCast.create(builder, ".unionTmpMember",
                                             alloca, memberPtrType);
    return memberType.load(cast, module, builder);
  }

  @Override
  public LLVMConstant offsetof(int member, LLVMModule module) {
    final LLVMContext ctxt = module.getContext();
    return LLVMConstant.constNull(SRC_SIZE_T_TYPE.getLLVMType(ctxt));
  }

  @Override
  public String toCompatibilityString(Set<SrcStructOrUnionType> structSet,
                                      LLVMContext ctxt)
  {
    // The implementation in SrcStructOrUnion is probably fine, but we don't
    // expect to be calling this for unions, so we're not testing it.
    throw new IllegalStateException(
      "toCompatibilityString called for union");
  }
}
