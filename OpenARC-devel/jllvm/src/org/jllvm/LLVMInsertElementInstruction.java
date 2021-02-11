package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMInsertElementInstruction extends LLVMInstruction {
	private static SWIGTYPE_p_LLVMOpaqueValue buildInstruction(LLVMInstructionBuilder builder,LLVMValue vector,LLVMValue element,LLVMValue index,String name) {
		assert(vector.typeOf() instanceof LLVMVectorType && index.typeOf() instanceof LLVMIntegerType && element.typeOf() == ((LLVMVectorType)vector.typeOf()).getElementType());
		return Core.LLVMBuildInsertElement(builder.getInstance(),vector.getInstance(),element.getInstance(),index.getInstance(),name);
	}
	public static LLVMUser create(LLVMInstructionBuilder builder,LLVMValue vector,LLVMValue element,LLVMValue index,String name) {
		return (LLVMUser)LLVMValue.getValue(buildInstruction(builder,vector,element,index,name));
	}
	public LLVMInsertElementInstruction(SWIGTYPE_p_LLVMOpaqueValue c) {
		super(c);
	}
}
