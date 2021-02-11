package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMUndefinedValue extends LLVMConstant {
	public static LLVMUndefinedValue get(LLVMType t) {
		return (LLVMUndefinedValue)LLVMValue.getValue(Core.LLVMGetUndef(t.getInstance()));
	}
	public LLVMUndefinedValue(SWIGTYPE_p_LLVMOpaqueValue val) {
		super(val);
	}
}
