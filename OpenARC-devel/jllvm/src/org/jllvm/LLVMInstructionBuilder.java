package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueBuilder;

public class LLVMInstructionBuilder {
	private SWIGTYPE_p_LLVMOpaqueBuilder instance;
	
	// Avoid using global context: see LLVMContext documentation.
	//public LLVMInstructionBuilder() {
	//	instance = Core.LLVMCreateBuilder();
	//}
	
	public LLVMInstructionBuilder(LLVMContext ctxt) {
		instance = Core.LLVMCreateBuilderInContext(ctxt.getInstance());
	}
	
	public void positionBuilder(LLVMBasicBlock block,LLVMInstruction instr) {
		Core.LLVMPositionBuilder(instance,block.getBBInstance(),instr.getInstance());
	}
	
	public void positionBuilderBefore(LLVMInstruction instr) {
		Core.LLVMPositionBuilderBefore(instance,instr.getInstance());
	}
	
	public void positionBuilderAtEnd(LLVMBasicBlock block) {
		Core.LLVMPositionBuilderAtEnd(instance,block.getBBInstance());
	}
	
	public LLVMBasicBlock getInsertBlock() {
		return LLVMBasicBlock.getBasicBlock(Core.LLVMGetInsertBlock(instance));
	}
	
	public void clearInsertionPosition() {
		Core.LLVMClearInsertionPosition(instance);
	}
	
	public void insertIntoBuilder(LLVMInstruction instr) {
		Core.LLVMInsertIntoBuilder(instance,instr.getInstance());
	}
	
	public SWIGTYPE_p_LLVMOpaqueBuilder getInstance() {
		return instance;
	}
	
	protected void finalize() {
		Core.LLVMDisposeBuilder(instance);
	}
}
