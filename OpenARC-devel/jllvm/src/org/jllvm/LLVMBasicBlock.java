package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueBasicBlock;

public class LLVMBasicBlock extends LLVMValue {
	public LLVMBasicBlock(SWIGTYPE_p_LLVMOpaqueValue val) {
		super(val);
		assert(Core.LLVMValueIsBasicBlock(val) != 0);
	}
	
	public LLVMBasicBlock(SWIGTYPE_p_LLVMOpaqueBasicBlock bb) {
		super(Core.LLVMBasicBlockAsValue(bb));
		assert(bb == null || Core.LLVMValueIsBasicBlock(instance) != 0);
	}
	
	public SWIGTYPE_p_LLVMOpaqueBasicBlock getBBInstance() {
		return Core.LLVMValueAsBasicBlock(instance);
	}
	
	public LLVMFunction getParent() {
		SWIGTYPE_p_LLVMOpaqueBasicBlock bb = getBBInstance();
		return LLVMFunction.getFunction(Core.LLVMGetBasicBlockParent(bb));
	}
	
	public LLVMBasicBlock getNextBasicBlock() {
		return getBasicBlock(Core.LLVMGetNextBasicBlock(getBBInstance()));
	}
	
	public LLVMBasicBlock getPreviousBasicBlock() {
		return getBasicBlock(Core.LLVMGetPreviousBasicBlock(getBBInstance()));
	}
	
	public LLVMBasicBlock insertBasicBlockBefore(String name,LLVMContext ctxt) {
		return new LLVMBasicBlock(Core.LLVMInsertBasicBlockInContext(ctxt.getInstance(),getBBInstance(),name));
	}
	
	public void moveBefore(LLVMBasicBlock MovePos) {
		Core.LLVMMoveBasicBlockBefore(getBBInstance(),MovePos.getBBInstance());
	}
	
	public void moveAfter(LLVMBasicBlock MovePos) {
		Core.LLVMMoveBasicBlockAfter(getBBInstance(),MovePos.getBBInstance());
	}
	
	public LLVMInstruction getFirstInstruction() {
		return LLVMInstruction.getInstruction(Core.LLVMGetFirstInstruction(getBBInstance()));
	}
	
	public LLVMInstruction getLastInstruction() {
		return LLVMInstruction.getInstruction(Core.LLVMGetLastInstruction(getBBInstance()));
	}
	
	public void delete() {
		Core.LLVMDeleteBasicBlock(getBBInstance());
	}
	
	public static LLVMBasicBlock getBasicBlock(SWIGTYPE_p_LLVMOpaqueValue val) {
		return val == null ? new LLVMBasicBlock((SWIGTYPE_p_LLVMOpaqueValue)null)
		                   : (LLVMBasicBlock)LLVMValue.getValue(val);
	}

	public static LLVMBasicBlock getBasicBlock(SWIGTYPE_p_LLVMOpaqueBasicBlock bb) {
		return bb == null ? new LLVMBasicBlock((SWIGTYPE_p_LLVMOpaqueBasicBlock)null)
		                  : (LLVMBasicBlock)LLVMValue.getValue(Core.LLVMBasicBlockAsValue(bb));
	}
}
