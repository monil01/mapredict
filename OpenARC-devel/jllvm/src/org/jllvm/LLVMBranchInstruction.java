package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;
import org.jllvm.LLVMTerminatorInstruction;

public class LLVMBranchInstruction extends LLVMTerminatorInstruction {
	public LLVMBranchInstruction(LLVMInstructionBuilder builder,LLVMBasicBlock destination) {
		this(Core.LLVMBuildBr(builder.getInstance(),destination.getBBInstance()));
	}
	
	public LLVMBranchInstruction(LLVMInstructionBuilder builder,LLVMValue condition,LLVMBasicBlock thenBlock,LLVMBasicBlock elseBlock) {
		this(Core.LLVMBuildCondBr(builder.getInstance(),condition.getInstance(),thenBlock.getBBInstance(),elseBlock.getBBInstance()));
	}
	
	public LLVMBranchInstruction(SWIGTYPE_p_LLVMOpaqueValue val) {
		super(val);
	}
}
