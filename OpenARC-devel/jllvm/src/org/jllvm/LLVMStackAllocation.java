package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMStackAllocation extends LLVMAllocationInstruction {
	private static SWIGTYPE_p_LLVMOpaqueValue buildInstruction(LLVMInstructionBuilder builder,LLVMType type,LLVMValue number,String name) {
		LLVMPointerType ptrtype = LLVMPointerType.get(type,0);
		if(number != null) {
			assert(number.typeOf() instanceof LLVMIntegerType);
			return Core.LLVMBuildArrayAlloca(builder.getInstance(),ptrtype.getElementType().getInstance(),number.getInstance(),name);
		}
		return Core.LLVMBuildAlloca(builder.getInstance(),ptrtype.getElementType().getInstance(),name);
	}
	public LLVMStackAllocation(LLVMInstructionBuilder builder,String name,LLVMType type,LLVMValue number) {
		this(buildInstruction(builder, type, number, name));
	}
	public LLVMStackAllocation(SWIGTYPE_p_LLVMOpaqueValue c) {
		super(c);
	}
}
