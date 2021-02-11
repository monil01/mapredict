package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.LLVMIntPredicate;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMIntegerComparison extends LLVMComparisonInstruction {
	public static LLVMUser create(LLVMInstructionBuilder builder,String name,LLVMIntPredicate Op,LLVMValue lhs,LLVMValue rhs) {
		return (LLVMUser)LLVMValue.getValue(Core.LLVMBuildICmp(builder.getInstance(),Op,lhs.getInstance(),rhs.getInstance(),name));
	}
	public LLVMIntegerComparison(SWIGTYPE_p_LLVMOpaqueValue c) {
		super(c);
	}
}
