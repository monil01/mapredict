package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMUnreachableInstruction extends LLVMTerminatorInstruction {
	public LLVMUnreachableInstruction(LLVMInstructionBuilder builder) {
		this(Core.LLVMBuildUnreachable(builder.getInstance()));
	}
	public LLVMUnreachableInstruction(SWIGTYPE_p_LLVMOpaqueValue val) {
		super(val);
	}
}
