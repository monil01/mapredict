package org.jllvm;

import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public abstract class LLVMArithmeticInstruction extends LLVMInstruction {
	public LLVMArithmeticInstruction(SWIGTYPE_p_LLVMOpaqueValue val) {
		super(val);
	}
}
