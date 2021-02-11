package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;
import org.jllvm.bindings.SWIGTYPE_p_p_LLVMOpaqueValue;
import org.jllvm.LLVMTerminatorInstruction;

public class LLVMInvokeInstruction extends LLVMTerminatorInstruction {
	private static SWIGTYPE_p_LLVMOpaqueValue buildInstruction(LLVMInstructionBuilder builder,LLVMFunction func,LLVMValue[] arguments,LLVMBasicBlock destination,LLVMBasicBlock unwind,String name) {
		SWIGTYPE_p_p_LLVMOpaqueValue argvalues = Core.new_LLVMValueRefArray(arguments.length);
		for(int i=0;i<arguments.length;i++)
			Core.LLVMValueRefArray_setitem(argvalues,i,arguments[i].getInstance());
		SWIGTYPE_p_LLVMOpaqueValue result = Core.LLVMBuildInvoke(builder.getInstance(),func.getInstance(),argvalues,(long)arguments.length,destination.getBBInstance(),unwind.getBBInstance(),name);
		Core.delete_LLVMValueRefArray(argvalues);
		return result;
	}
	public LLVMInvokeInstruction(LLVMInstructionBuilder builder,String name,LLVMFunction func,LLVMValue[] arguments,LLVMBasicBlock destination,LLVMBasicBlock unwind) {
		this(buildInstruction(builder,func,arguments,destination,unwind,name));
	}
	public LLVMInvokeInstruction(SWIGTYPE_p_LLVMOpaqueValue val) {
		super(val);
	}
}
