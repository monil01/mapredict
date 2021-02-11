package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;
import org.jllvm.LLVMValue;

/* Core.h doesn't specify anything about the type User, so there's pretty much nothing here. */
public abstract class LLVMUser extends LLVMValue {
	public LLVMUser(SWIGTYPE_p_LLVMOpaqueValue val) {
		super(Core.LLVMIsAUser(val));
	}
	public void setOperand(long Index,LLVMValue Val) {
		Core.LLVMSetOperand(instance,Index,Val.getInstance());
	}
	public LLVMValue getOperand(long Index) {
		return LLVMValue.getValue(Core.LLVMGetOperand(instance,Index));
	}
}
