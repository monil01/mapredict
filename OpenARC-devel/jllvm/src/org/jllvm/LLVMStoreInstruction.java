package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMStoreInstruction extends LLVMInstruction {
	public LLVMStoreInstruction(LLVMInstructionBuilder builder,LLVMValue value,LLVMValue pointer) {
		super(Core.LLVMBuildStore(builder.getInstance(),value.getInstance(),pointer.getInstance()));
		assert(pointer.typeOf() instanceof LLVMPointerType);
	}
	public LLVMStoreInstruction(SWIGTYPE_p_LLVMOpaqueValue val) {
		super(val);
	}
}
