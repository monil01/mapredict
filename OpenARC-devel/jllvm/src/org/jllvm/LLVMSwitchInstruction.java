package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;
import org.jllvm.LLVMTerminatorInstruction;

public class LLVMSwitchInstruction extends LLVMTerminatorInstruction {
	public LLVMSwitchInstruction(LLVMInstructionBuilder builder,LLVMValue value,LLVMBasicBlock block,long numCases) {
		this(Core.LLVMBuildSwitch(builder.getInstance(),value.getInstance(),block.getBBInstance(),numCases));
		assert(numCases >= 0);
	}
	
	public LLVMSwitchInstruction(SWIGTYPE_p_LLVMOpaqueValue val) {
		super(val);
	}
	
	public void addCase(LLVMValue onValue,LLVMBasicBlock destination) {
		Core.LLVMAddCase(getInstance(),onValue.getInstance(),destination.getBBInstance());
	}
}
