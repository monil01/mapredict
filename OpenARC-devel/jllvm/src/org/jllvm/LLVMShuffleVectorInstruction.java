package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMShuffleVectorInstruction extends LLVMInstruction {
	private static SWIGTYPE_p_LLVMOpaqueValue buildInstruction(LLVMInstructionBuilder builder,LLVMValue vec1,LLVMValue vec2,LLVMValue mask,String name) {
		assert(((LLVMVectorType)vec1.typeOf()).getElementType() == ((LLVMVectorType)vec2.typeOf()).getElementType()
		       && mask.typeOf() instanceof LLVMIntegerType
		       && ((LLVMIntegerType)mask.typeOf()).getWidth() == 32);
		return Core.LLVMBuildShuffleVector(builder.getInstance(),vec1.getInstance(),vec2.getInstance(),mask.getInstance(),name);
	}
	public LLVMUser create(LLVMInstructionBuilder builder,String name,LLVMValue vec1,LLVMValue vec2,LLVMValue mask) {
		return (LLVMUser)LLVMValue.getValue(buildInstruction(builder,vec1,vec2,mask,name));
	}
	public LLVMShuffleVectorInstruction(SWIGTYPE_p_LLVMOpaqueValue val) {
		super(val);
	}
}
