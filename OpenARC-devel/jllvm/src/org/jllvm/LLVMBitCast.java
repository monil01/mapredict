package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMBitCast extends LLVMCastInstruction {
	public static LLVMUser create(LLVMInstructionBuilder builder,String name,LLVMValue val,LLVMType destType) {
		return (LLVMUser)LLVMValue.getValue(Core.LLVMBuildBitCast(builder.getInstance(),val.getInstance(),destType.getInstance(),name));
	}
	public LLVMBitCast(SWIGTYPE_p_LLVMOpaqueValue c) {
		super(c);
	}
}
