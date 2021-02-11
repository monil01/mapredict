package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SRC_SIZE_T_TYPE;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;

import openacc.codegen.llvmBackend.SrcQualifiedType.SrcTypeQualifier;

import org.jllvm.LLVMBitCast;
import org.jllvm.LLVMConstant;
import org.jllvm.LLVMConstantExpression;
import org.jllvm.LLVMConstantInteger;
import org.jllvm.LLVMConstantStruct;
import org.jllvm.LLVMContext;
import org.jllvm.LLVMExtractValueInstruction;
import org.jllvm.LLVMGetElementPointerInstruction;
import org.jllvm.LLVMIdentifiedStructType;
import org.jllvm.LLVMInstructionBuilder;
import org.jllvm.LLVMIntegerType;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMPointerType;
import org.jllvm.LLVMTargetData;
import org.jllvm.LLVMType;
import org.jllvm.LLVMValue;

/**
 * The LLVM backend's class for all struct types from the C source.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class SrcStructType extends SrcStructOrUnionType {
  private long memberToLLVMMember[] = null;

  public SrcStructType(String tag, LLVMContext llvmContext) {
    super("struct", tag, llvmContext);
  }

  /**
   * {@code llvmTargetData} can be null because it is unused in this
   * implementation.
   */
  @Override
  protected void setValidatedBody(
    String[] memberNames, SrcType[] memberTypes,
    LLVMTargetData llvmTargetData, LLVMContext context)
  {
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

    // Compute LLVM member types, compute map from pruned members to LLVM
    // members, and update pruned member types based on bit-field packing,
    // which can affect storage unit types and offsets.
    final List<LLVMType> memberLLVMTypes = new ArrayList<>();
    memberToLLVMMember = new long[memberTypesPruned.length];
    SrcBitFieldType prevBitFieldType = null;
    for (int i = 0, prunedIdx = 0; i < memberNames.length; ++i) {
      final SrcType memberType = memberTypes[i];
      final SrcBitFieldType bitFieldType
        = memberType.toIso(SrcBitFieldType.class);
      if (bitFieldType != null) {
        if (bitFieldType.getWidth() == 0) {
          // ISO C99 sec. 6.7.2.1p11: ends packing.
          prevBitFieldType = null;
          // Unnamed bit-fields are inaccessible.
          assert(memberNames[i].isEmpty());
          continue;
        }
        else if (prevBitFieldType != null
                 && prevBitFieldType.getHighOrderOffset()
                    + prevBitFieldType.getWidth() + bitFieldType.getWidth()
                    <= prevBitFieldType.getLLVMWidth()
                 && (prevBitFieldType.getStorageUnitType().isSigned()
                     == bitFieldType.getStorageUnitType().isSigned()
                     || null != prevBitFieldType.getStorageUnitType()
                                .getSignednessToggle()))
        {
          // Pack into previous bit-field's storage unit.
          final SrcIntegerType storageUnitType
            = prevBitFieldType.getStorageUnitType().isSigned()
              == bitFieldType.getStorageUnitType().isSigned()
              ? prevBitFieldType.getStorageUnitType()
              : prevBitFieldType.getStorageUnitType().getSignednessToggle();
          prevBitFieldType = new SrcBitFieldType(
            storageUnitType, bitFieldType.getWidth(),
            prevBitFieldType.getHighOrderOffset()
            + prevBitFieldType.getWidth());
          // Unnamed bit-fields are inaccessible.
          if (memberNames[i].isEmpty())
            continue;
          memberTypesPruned[prunedIdx]
            = SrcQualifiedType.get(prevBitFieldType,
                                   memberType.expandSrcTypeQualifiers());
        }
        else {
          // Start new bit-field storage unit.
          prevBitFieldType = bitFieldType;
          // Unnamed bit-fields are inaccessible.
          if (memberNames[i].isEmpty())
            continue;
          memberLLVMTypes.add(bitFieldType.getLLVMType(context));
        }
      }
      else {
        prevBitFieldType = null;
        memberLLVMTypes.add(memberType.getLLVMType(context));
      }
      memberToLLVMMember[prunedIdx++] = memberLLVMTypes.size() - 1;
    }
    final LLVMType[] memberLLVMTypeArray = new LLVMType[memberLLVMTypes.size()];
    memberLLVMTypes.toArray(memberLLVMTypeArray);
    setValidatedBody(memberNamesPruned, memberTypesPruned,
                     memberLLVMTypeArray);
  }

  @Override
  public ValueAndType buildConstant(List<LLVMConstant> inits,
                                    LLVMModule module,
                                    LLVMInstructionBuilder builder)
  {
    final LLVMContext ctxt = module.getContext();
    assert(inits.size() <= getMemberTypes().length);
    final LLVMType[] llvmMemberTypes = getLLVMType(ctxt).getElementTypes();
    final LLVMConstant[] initsPacked = new LLVMConstant[llvmMemberTypes.length];
    int llvmMemberCount = 0;
    for (int i = 0; i < getMemberTypes().length; ++i) {
      final LLVMConstant init
        = i < inits.size()
          ? inits.get(i)
          : LLVMConstant.constNull(getMemberTypes()[i].getLLVMType(ctxt));
      final SrcBitFieldType bitFieldType
        = getMemberTypes()[i].toIso(SrcBitFieldType.class);
      if (bitFieldType != null) {
        if (bitFieldType.getHighOrderOffset() == 0)
          initsPacked[llvmMemberCount++]
            = LLVMConstant.constNull(getMemberTypes()[i].getLLVMType(ctxt));
        initsPacked[llvmMemberCount-1]
          = (LLVMConstant)bitFieldType.insertIntoStorageUnit(
              initsPacked[llvmMemberCount-1], init, module, builder);
      }
      else
        initsPacked[llvmMemberCount++] = init;
    }
    assert(llvmMemberCount == llvmMemberTypes.length);
    return new ValueAndType(
      LLVMConstantStruct.get(getLLVMType(ctxt), initsPacked), this, false);
  }

  @Override
  public ValueAndType accessMember(
    EnumSet<SrcTypeQualifier> structOrUnionTypeQualifiers,
    LLVMValue expr, int member, boolean resultIsLvalue, LLVMModule module,
    LLVMInstructionBuilder builder, LLVMInstructionBuilder allocaBuilder)
  {
    final LLVMIntegerType i32 = LLVMIntegerType.get(module.getContext(), 32);
    final LLVMConstant zero = LLVMConstant.constNull(i32);
    final SrcType memberType = getMemberTypes()[member];
    final SrcType resType;
    LLVMValue resValue;
    if (resultIsLvalue) {
      assert(expr.typeOf() instanceof LLVMPointerType);
      assert(((LLVMPointerType)expr.typeOf()).getElementType()
             instanceof LLVMIdentifiedStructType);
      resValue = LLVMGetElementPointerInstruction.create(
        builder, ".structMember", expr,
        zero, LLVMConstantInteger.get(i32, memberToLLVMMember[member], false));
      final EnumSet<SrcTypeQualifier> resQualsAdd
        = structOrUnionTypeQualifiers;
      // If the member has nvl_wp, then the struct must have nvl. We cannot
      // combine these two, the former is correct on the result, but LLVM is
      // aware of only the latter.
      if (memberType.hasEffectiveQualifier(SrcTypeQualifier.NVL_WP)) {
        assert(resQualsAdd.contains(SrcTypeQualifier.NVL));
        resQualsAdd.remove(SrcTypeQualifier.NVL);
        // Some version of LLVM after 3.2 requires an addrspacecast here
        // instead of bitcast, but LLVM 3.2 doesn't support addrspacecast.
        resValue = LLVMBitCast.create(
          builder, ".castToNvlWp", resValue,
          SrcPointerType.get(SrcQualifiedType.get(memberType, resQualsAdd))
          .getLLVMType(module.getContext()));
      }
      resType = SrcQualifiedType.get(memberType, resQualsAdd);
    }
    else {
      assert(expr.typeOf() instanceof LLVMIdentifiedStructType);
      resValue = LLVMExtractValueInstruction.create(
        builder, ".structMember", expr, memberToLLVMMember[member]);
      final SrcBitFieldType memberBitFieldType
        = memberType.toIso(SrcBitFieldType.class);
      if (memberBitFieldType != null)
        resValue = memberBitFieldType.extractFromStorageUnit(resValue, module,
                                                             builder);
      resType = memberType;
    }
    return new ValueAndType(resValue, resType, resultIsLvalue);
  }

  @Override
  public LLVMConstant offsetof(int member, LLVMModule module) {
    final LLVMContext ctxt = module.getContext();
    final LLVMIntegerType i32 = LLVMIntegerType.get(ctxt, 32);
    final LLVMConstant zero = LLVMConstant.constNull(i32);
    final LLVMConstant offset
      = LLVMConstantExpression.gep(
          LLVMConstant.constNull(SrcPointerType.get(this).getLLVMType(ctxt)),
          zero, LLVMConstantInteger.get(i32, member, false));
    return LLVMConstantExpression.ptrToInt(offset, SRC_SIZE_T_TYPE
                                                   .getLLVMType(ctxt));
  }
}
