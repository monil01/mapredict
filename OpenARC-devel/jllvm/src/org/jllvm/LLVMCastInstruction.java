package org.jllvm;

import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public abstract class LLVMCastInstruction extends LLVMInstruction {
	public LLVMCastInstruction(SWIGTYPE_p_LLVMOpaqueValue c) {
		super(c);
	}
}
