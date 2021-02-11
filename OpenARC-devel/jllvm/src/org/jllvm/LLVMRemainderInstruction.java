package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMRemainderInstruction extends LLVMArithmeticInstruction {
	public enum RemainderType { FLOAT, SIGNEDINT, UNSIGNEDINT }
	
	private static SWIGTYPE_p_LLVMOpaqueValue buildInstruction(LLVMInstructionBuilder builder,LLVMValue lhs,LLVMValue rhs,RemainderType kind,String name) {
		switch(kind) {
			case FLOAT:
				return Core.LLVMBuildFRem(builder.getInstance(),lhs.getInstance(),rhs.getInstance(),name);
			case SIGNEDINT:
				return Core.LLVMBuildSRem(builder.getInstance(),lhs.getInstance(),rhs.getInstance(),name);
			case UNSIGNEDINT:
				return Core.LLVMBuildURem(builder.getInstance(),lhs.getInstance(),rhs.getInstance(),name);
		}
		//This should never run.
		return null;
	}
	
	public static LLVMUser create(LLVMInstructionBuilder builder,String name,LLVMValue lhs,LLVMValue rhs,RemainderType kind) {
		return (LLVMUser)LLVMValue.getValue(buildInstruction(builder,lhs,rhs,kind,name));
	}
	
	public LLVMRemainderInstruction(SWIGTYPE_p_LLVMOpaqueValue c) {
		super(c);
	}
}
