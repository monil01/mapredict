package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMExtractValueInstruction extends LLVMInstruction {
	public static LLVMUser create(LLVMInstructionBuilder builder,String name,LLVMValue aggr,long index) {
		return (LLVMUser)LLVMValue.getValue(Core.LLVMBuildExtractValue(builder.getInstance(),aggr.getInstance(),index,name));
	}
	public LLVMExtractValueInstruction(SWIGTYPE_p_LLVMOpaqueValue c) {
		super(c);
	}
}
