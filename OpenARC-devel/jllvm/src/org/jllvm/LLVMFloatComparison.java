package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.LLVMRealPredicate;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMFloatComparison extends LLVMComparisonInstruction {
	public static LLVMUser create(LLVMInstructionBuilder builder,String name,LLVMRealPredicate Op,LLVMValue lhs,LLVMValue rhs) {
		return (LLVMUser)LLVMValue.getValue(Core.LLVMBuildFCmp(builder.getInstance(),Op,lhs.getInstance(),rhs.getInstance(),name));
	}
	public LLVMFloatComparison(SWIGTYPE_p_LLVMOpaqueValue c) {
		super(c);
	}
}
