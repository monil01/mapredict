package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;
import org.jllvm.bindings.SWIGTYPE_p_p_LLVMOpaqueValue;

public class LLVMGetElementPointerInstruction extends LLVMInstruction {
	private static SWIGTYPE_p_LLVMOpaqueValue buildInstruction(LLVMInstructionBuilder builder,LLVMValue pointer,LLVMValue indices[],String name) {
		int num_indices = indices.length;
		SWIGTYPE_p_p_LLVMOpaqueValue values = Core.new_LLVMValueRefArray(num_indices);
		for(int i=0;i<indices.length;i++)
			Core.LLVMValueRefArray_setitem(values,i,indices[i].getInstance());
		SWIGTYPE_p_LLVMOpaqueValue result = Core.LLVMBuildGEP(builder.getInstance(),pointer.getInstance(),values,num_indices,name);
		Core.delete_LLVMValueRefArray(values);
		return result;
	}
	public static LLVMUser create(LLVMInstructionBuilder builder,String name,LLVMValue pointer,LLVMValue... indices) {
		return (LLVMUser)LLVMValue.getValue(buildInstruction(builder,pointer,indices,name));
	}
	public LLVMGetElementPointerInstruction(SWIGTYPE_p_LLVMOpaqueValue c) {
		super(c);
	}
}
