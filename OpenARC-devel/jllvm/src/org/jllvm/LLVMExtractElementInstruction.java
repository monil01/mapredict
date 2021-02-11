package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMExtractElementInstruction extends LLVMInstruction {
	private static SWIGTYPE_p_LLVMOpaqueValue buildInstruction(LLVMInstructionBuilder builder,LLVMValue vector,LLVMValue index,String name) {
		assert(vector.typeOf() instanceof LLVMVectorType && index.typeOf() instanceof LLVMIntegerType);
		return Core.LLVMBuildExtractElement(builder.getInstance(),vector.getInstance(),index.getInstance(),name);
	}
	public static LLVMUser create(LLVMInstructionBuilder builder,String name,LLVMValue vector,LLVMValue index) {
		return (LLVMUser)LLVMValue.getValue(buildInstruction(builder,vector,index,name));
	}
	public LLVMExtractElementInstruction(SWIGTYPE_p_LLVMOpaqueValue c) {
		super(c);
	}
}
