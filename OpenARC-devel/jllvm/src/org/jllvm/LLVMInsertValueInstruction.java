package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMInsertValueInstruction extends LLVMInstruction {
	public static LLVMUser create(LLVMInstructionBuilder builder,LLVMValue aggr,LLVMValue Value,long index,String name) {
		//assert(aggr.typeOf() instanceof LLVMaggrType && Value.typeOf() == ((LLVMaggrType)aggr.typeOf()).getValueType());
		return (LLVMUser)LLVMValue.getValue(Core.LLVMBuildInsertValue(builder.getInstance(),aggr.getInstance(),Value.getInstance(),index,name));
	}
	public LLVMInsertValueInstruction(SWIGTYPE_p_LLVMOpaqueValue c) {
		super(c);
	}
}
