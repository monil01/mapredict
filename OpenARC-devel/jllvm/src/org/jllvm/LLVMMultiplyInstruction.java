package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMMultiplyInstruction extends LLVMArithmeticInstruction {
	private static SWIGTYPE_p_LLVMOpaqueValue buildInstruction(LLVMInstructionBuilder builder,LLVMValue lhs,LLVMValue rhs,boolean fp,String name) {
		if(fp)
			return Core.LLVMBuildFMul(builder.getInstance(),lhs.getInstance(),rhs.getInstance(),name);
		else
			return Core.LLVMBuildMul(builder.getInstance(),lhs.getInstance(),rhs.getInstance(),name);
	}
	
	public static LLVMUser create(LLVMInstructionBuilder builder,String name,LLVMValue lhs,LLVMValue rhs,boolean fp) {
		return (LLVMUser)LLVMValue.getValue(buildInstruction(builder,lhs,rhs,fp,name));
	}
	
	public LLVMMultiplyInstruction(SWIGTYPE_p_LLVMOpaqueValue c) {
		super(c);
	}
}
