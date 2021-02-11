package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMLoadInstruction extends LLVMInstruction {
	private static SWIGTYPE_p_LLVMOpaqueValue buildInstruction(LLVMInstructionBuilder builder,LLVMValue pointer,String name) {
		assert(pointer.typeOf() instanceof LLVMPointerType);
		return Core.LLVMBuildLoad(builder.getInstance(),pointer.getInstance(),name);
	}
	public LLVMLoadInstruction(LLVMInstructionBuilder builder,String name,LLVMValue pointer) {
		this(buildInstruction(builder,pointer,name));
	}
	public LLVMLoadInstruction(SWIGTYPE_p_LLVMOpaqueValue c) {
		super(c);
	}
}
