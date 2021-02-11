package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMVariableArgumentInstruction extends LLVMInstruction {
	public LLVMVariableArgumentInstruction(LLVMInstructionBuilder builder,String name,LLVMValue valist,LLVMType vatype) {
		this(Core.LLVMBuildVAArg(builder.getInstance(),valist.getInstance(),vatype.getInstance(),name));
	}
	public LLVMVariableArgumentInstruction(SWIGTYPE_p_LLVMOpaqueValue val) {
		super(val);
	}
}
