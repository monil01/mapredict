package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMSelectInstruction extends LLVMInstruction {
	private static SWIGTYPE_p_LLVMOpaqueValue buildInstruction(LLVMInstructionBuilder builder,LLVMValue condition,LLVMValue then,LLVMValue otherwise,String name) {
		assert((condition.typeOf() instanceof LLVMIntegerType && ((LLVMIntegerType)condition.typeOf()).getWidth() == 1)
		       || (condition.typeOf() instanceof LLVMVectorType
		           && ((LLVMVectorType)condition.typeOf()).getElementType() instanceof LLVMIntegerType
		           && ((LLVMIntegerType)((LLVMVectorType)condition.typeOf()).getElementType()).getWidth() == 1));
		assert(then.typeOf() == otherwise.typeOf());
		return Core.LLVMBuildSelect(builder.getInstance(),condition.getInstance(),then.getInstance(),otherwise.getInstance(),name);
	}
	public static LLVMUser create(LLVMInstructionBuilder builder,String name,LLVMValue condition,LLVMValue then,LLVMValue otherwise) {
		return (LLVMUser)LLVMValue.getValue(buildInstruction(builder,condition,then,otherwise,name));
	}
	public LLVMSelectInstruction(SWIGTYPE_p_LLVMOpaqueValue val) {
		super(val);
	}
}
