package org.jllvm;

import org.jllvm.LLVMInstruction;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

/* This class exists solely to mirror the C++ bindings and have a superclass for terminator instructions. */
public abstract class LLVMTerminatorInstruction extends LLVMInstruction {
	public LLVMTerminatorInstruction(SWIGTYPE_p_LLVMOpaqueValue val) {
		super(val);
	}
}
