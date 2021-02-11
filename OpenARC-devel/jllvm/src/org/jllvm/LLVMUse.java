package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueUse;

public class LLVMUse {
	private SWIGTYPE_p_LLVMOpaqueUse instance;
	
	public SWIGTYPE_p_LLVMOpaqueUse getInstance() {
		return instance;
	}
	
	public LLVMUse(SWIGTYPE_p_LLVMOpaqueUse c) {
		instance = c;
	}
	
	public LLVMUse getNextUse() {
		return new LLVMUse(Core.LLVMGetNextUse(instance));
	}
	
	public LLVMValue getUser() {
		return LLVMValue.getValue(Core.LLVMGetUser(instance));
	}
	
	public LLVMValue getUsedValue() {
		return LLVMValue.getValue(Core.LLVMGetUsedValue(instance));
	}
}
