package openacc.codegen.llvmBackend;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Stack;

import org.jllvm.LLVMBasicBlock;
import org.jllvm.LLVMBranchInstruction;
import org.jllvm.LLVMContext;
import org.jllvm.LLVMFunction;
import org.jllvm.LLVMInstructionBuilder;
import org.jllvm.LLVMMDNode;
import org.jllvm.LLVMMDString;
import org.jllvm.LLVMValue;

/**
 * The LLVM backend's class for specifying basic block sets and for generating
 * LLVM metadata to identify those sets appropriately for the BasicBlockSet
 * class we have added to LLVM.
 * 
 * <p>
 * For a description of the format of the associated LLVM metadata, see the
 * include/llvm/Transforms/Utils/BasicBlockSet.h we have added to LLVM.
 * </p>
 * 
 * <p>
 * This class stores basic block sets in a stack. Pushing and popping a set (
 * {@link #push} and {@link #pop}) indicates the start and completion of the
 * construction of the set and its basic blocks. Sets stored higher in the
 * stack and are nested within sets stored lower. Registering a basic block (
 * {@link #registerBasicBlock}) makes it a direct member of the topmost set on
 * the stack and thus an indirect member of all other sets on the stack. No
 * basic block should be registered more than once. When a set is popped from
 * the stack, its direct members must already be complete because metadata is
 * then attached to their terminator instructions.
 * </p>
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public class BasicBlockSetStack {
  private static class Set {
    public final LLVMFunction fn;
    public final LLVMMDNode md;
    public final String setListMDName;
    public final List<LLVMBasicBlock> bbs;
    public Set(LLVMFunction fn, LLVMMDNode md, String setListMDName) {
      this.fn = fn;
      this.md = md;
      this.setListMDName = setListMDName;
      this.bbs = new ArrayList<>();
    }
  }

  private final Stack<Set> stack = new Stack<>();
  private final LLVMContext llvmContext;

  /**
   * Create a new empty stack of basic block sets.
   * 
   * @param llvmContext
   *          the {@link LLVMContext} for all basic blocks in the sets in
   *          this stack
   */
  public BasicBlockSetStack(LLVMContext llvmContext) {
    this.llvmContext = llvmContext;
  }

  /**
   * Push a new basic block set, register a new basic block to which the
   * current basic block unconditionally branches as the start of this set,
   * and position the current builder in the new basic block.
   * 
   * @param setMDName
   *          the metadata string from the identifier metadata nodes for sets
   *          of this kind
   * @param setListMDName
   *          the metadata string from the list metadata nodes for sets of
   *          this kind. Also used as the metadata kind when such a list
   *          metadata node is attached to a basic block's terminator
   *          instruction.
   * @param startBBName
   *          the name of the new basic block, which is the first basic block
   *          in the new set
   * @param llvmBuilder
   *          the current builder
   * @return the set's ID metadata node
   */
  public LLVMMDNode push(String setMDName, String setListMDName,
                         String startBBName,
                         LLVMInstructionBuilder llvmBuilder)
  {
    final LLVMFunction llvmFunction = llvmBuilder.getInsertBlock().getParent();
    final LLVMMDNode md = push(setMDName, setListMDName, llvmBuilder);
    final LLVMBasicBlock startBB = registerBasicBlock(startBBName,
                                                      llvmFunction);
    new LLVMBranchInstruction(llvmBuilder, startBB);
    llvmBuilder.positionBuilderAtEnd(startBB);
    return md;
  }

  /**
   * Push a new basic block set.
   * 
   * @param setMDName
   *          the metadata string from the identifier metadata nodes for sets
   *          of this kind
   * @param setListMDName
   *          the metadata string from the list metadata nodes for sets of
   *          this kind. Also used as the metadata kind when such a list
   *          metadata node is attached to a basic block's terminator
   *          instruction.
   * @param llvmBuilder
   *          the current builder
   */
  public LLVMMDNode push(String setMDName, String setListMDName,
                         LLVMInstructionBuilder llvmBuilder)
  {
    final LLVMFunction llvmFunction = llvmBuilder.getInsertBlock().getParent();
    final LLVMMDNode tmpMD = LLVMMDNode.getTemporary(llvmContext);
    final LLVMMDNode md
      = LLVMMDNode.get(llvmContext, LLVMMDString.get(llvmContext, setMDName),
                       tmpMD);
    md.replaceOperandWith(1, md);
    tmpMD.deleteTemporary();
    stack.push(new Set(llvmFunction, md, setListMDName));
    return md;
  }

  /**
   * Pop the most recently pushed basic block set, register a new basic
   * block to which the current basic block unconditionally branches as an end
   * of this set, and position the current builder in the new basic block.
   * 
   * @param nextBBName
   *          the name of the new basic block, which is not part of the set
   *          being popped
   * @param llvmBuilder
   *          the current builder
   */
  public void pop(String nextBBName, LLVMInstructionBuilder llvmBuilder) {
    // Create the next basic block, but don't register it in a set yet.
    final LLVMFunction llvmFunction = llvmBuilder.getInsertBlock().getParent();
    final LLVMBasicBlock nextBB = llvmFunction.appendBasicBlock(nextBBName,
                                                                llvmContext);
    new LLVMBranchInstruction(llvmBuilder, nextBB);
    llvmBuilder.positionBuilderAtEnd(nextBB);

    // Add metadata to basic blocks in this set, and pop the set.
    final Map<String, List<LLVMMDNode>> mdLists = new HashMap<>();
    for (Set set : stack) {
      List<LLVMMDNode> mdList = mdLists.get(set.setListMDName);
      if (mdList == null) {
        mdList = new ArrayList<>();
        mdLists.put(set.setListMDName, mdList);
      }
      mdList.add(set.md);
    }
    for (Map.Entry<String, List<LLVMMDNode>> entry : mdLists.entrySet()) {
      final String kindName = entry.getKey();
      final List<LLVMMDNode> mdList = entry.getValue();
      final long mdKindID = llvmContext.getMetadataKindID(kindName);
      final LLVMMDString mdKindString = LLVMMDString.get(llvmContext,
                                                         kindName);
      final LLVMValue[] mdOps = new LLVMValue[1 + mdList.size()];
      mdOps[0] = mdKindString;
      int i = 1;
      for (LLVMMDNode md : mdList)
        mdOps[i++] = md;
      final LLVMMDNode md = LLVMMDNode.get(llvmContext, mdOps);
      for (LLVMBasicBlock bb : stack.peek().bbs)
        bb.getLastInstruction().setMetadata(mdKindID, md);
    }
    stack.pop();

    // Add next basic block to enclosing set, if any.
    if (!stack.isEmpty())
      stack.peek().bbs.add(nextBB);
  }

  /**
   * Create a new basic block, and register it in the current basic block set,
   * if any.
   * 
   * @param name
   *          the name of the basic block
   * @param llvmFunction
   *          the current function, in which the basic block will be inserted.
   *          If that is different than the function enclosing the current
   *          basic block set, then we're in a fake function (see
   *          {@link BuildLLVM.Visitor.FakeFunctionForEval}), so don't
   *          register the basic block in the set.
   * @return the new basic block
   */
  public LLVMBasicBlock registerBasicBlock(String name,
                                           LLVMFunction llvmFunction)
  {
    final LLVMBasicBlock bb = createBasicBlockEarly(name, llvmFunction);
    registerBasicBlock(bb);
    return bb;
  }

  /**
   * Same as {@link #registerBasicBlock(String, LLVMFunction)} except do not
   * register the basic block yet. Before the basic block's set (which might
   * not yet have been pushed) is popped, the basic block must be registered
   * using {@link #registerBasicBlock(LLVMBasicBlock)}, or the basic block
   * must be destroyed without adding any instructions.
   */
  public LLVMBasicBlock createBasicBlockEarly(String name,
                                              LLVMFunction llvmFunction)
  {
    return llvmFunction.appendBasicBlock(name, llvmContext);
  }

  /**
   * Same as {@link #registerBasicBlock(String, LLVMFunction)} except use an
   * existing basic block that has never been registered in the set stack.
   */
  public void registerBasicBlock(LLVMBasicBlock bb) {
    if (!stack.isEmpty() && stack.peek().fn == bb.getParent())
      stack.peek().bbs.add(bb);
  }

  /**
   * Same as {@link #createBasicBlockEarly} except never register the basic
   * block. While {@link #createBasicBlockEarly} is functionally the same,
   * this method clarifies that the caller guarantees the basic block will
   * never contain instructions and will be thrown away.
   */
  public LLVMBasicBlock createDummyBasicBlock(String name,
                                              LLVMFunction llvmFunction)
  {
    return llvmFunction.appendBasicBlock(name, llvmContext);
  }
}
