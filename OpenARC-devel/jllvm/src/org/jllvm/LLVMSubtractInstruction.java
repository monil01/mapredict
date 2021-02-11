package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMSubtractInstruction extends LLVMArithmeticInstruction {
	private static SWIGTYPE_p_LLVMOpaqueValue buildInstruction(LLVMInstructionBuilder builder,LLVMValue lhs,LLVMValue rhs,boolean fp,String name) {
		if(fp)
			return Core.LLVMBuildFSub(builder.getInstance(),lhs.getInstance(),rhs.getInstance(),name);
		else
			return Core.LLVMBuildSub(builder.getInstance(),lhs.getInstance(),rhs.getInstance(),name);
	}
	
	public static LLVMUser create(LLVMInstructionBuilder builder,String name,LLVMValue lhs,LLVMValue rhs,boolean fp) {
		return (LLVMUser)LLVMValue.getValue(buildInstruction(builder,lhs,rhs,fp,name));
	}
	
	public LLVMSubtractInstruction(SWIGTYPE_p_LLVMOpaqueValue c) {
		super(c);
	}
}
