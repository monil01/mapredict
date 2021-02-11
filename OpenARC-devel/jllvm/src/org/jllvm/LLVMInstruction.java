package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.LLVMOpcode;
import org.jllvm.LLVMBasicBlock;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

/**
 * This is class is concrete purely so it can wrap null.
 */
public class LLVMInstruction extends LLVMUser {
	public LLVMInstruction(SWIGTYPE_p_LLVMOpaqueValue val) {
		super(val);
	}

	public LLVMBasicBlock getParent() {
		return LLVMBasicBlock.getBasicBlock(Core.LLVMGetInstructionParent(instance));
	}
	
	public LLVMInstruction getNextInstruction() {
		return LLVMInstruction.getInstruction(Core.LLVMGetNextInstruction(instance));
	}
	
	public LLVMInstruction getPreviousInstruction() {
		return LLVMInstruction.getInstruction(Core.LLVMGetPreviousInstruction(instance));
	}
	
	public LLVMValue getOperand(long index) {
		return LLVMValue.getValue(Core.LLVMGetOperand(instance, index));
	}
	
	public LLVMOpcode getInstructionOpcode() {
		return Core.LLVMGetInstructionOpcode(instance);
	}
	
	public boolean hasMetadata() {
		return Core.LLVMHasMetadata(instance) != 0;
	}
	
	public void setMetadata(long KindID, LLVMMDNode md) {
		Core.LLVMSetMetadata(instance, KindID, md.getInstance());
	}
	
	public LLVMMDNode getMetadata(long KindID) {
		return (LLVMMDNode)LLVMValue.getValue(Core.LLVMGetMetadata(instance, KindID));
	}
	
	public static LLVMInstruction getInstruction(SWIGTYPE_p_LLVMOpaqueValue val) {
		return val == null ? new LLVMInstruction(null)
		                   : (LLVMInstruction)LLVMValue.getValue(val);
	}
	
	/*
	public LLVMInstruction matchInstruction(SWIGTYPE_p_LLVMOpaqueValue val) {
		assert(Core.LLVMIsAInstruction(val));
	}
	*/
	
	public void eraseFromParent() {
		Core.LLVMInstructionEraseFromParent(instance);
	}
}
