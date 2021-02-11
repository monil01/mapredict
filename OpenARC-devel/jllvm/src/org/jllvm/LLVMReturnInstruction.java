package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMReturnInstruction extends LLVMTerminatorInstruction {
	private static SWIGTYPE_p_LLVMOpaqueValue buildInstruction(LLVMInstructionBuilder builder,LLVMValue val) {
		if(val != null) {
			return Core.LLVMBuildRet(builder.getInstance(),val.getInstance());
		}
		else {
			return Core.LLVMBuildRetVoid(builder.getInstance());
		}
	}

	/* Can accept null as its second parameter. */	
	public LLVMReturnInstruction(LLVMInstructionBuilder builder,LLVMValue val) {
		this(buildInstruction(builder,val));
	}
	
	public LLVMReturnInstruction(SWIGTYPE_p_LLVMOpaqueValue val) {
		super(val);
	}
	
	public boolean isVoid() {
		return Core.LLVMGetNumOperands(instance) == 0;
	}
}
