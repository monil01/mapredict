package org.jllvm;

import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public abstract class LLVMAllocationInstruction extends LLVMInstruction {
	public LLVMAllocationInstruction(SWIGTYPE_p_LLVMOpaqueValue c) {
		super(c);
	}
}
