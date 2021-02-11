package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;
import org.jllvm.bindings.LLVMAttribute;

public class LLVMArgument extends LLVMValue {
	public LLVMArgument(SWIGTYPE_p_LLVMOpaqueValue val) {
		super(val);
	}
	
	public LLVMFunction getParent() {
		return LLVMFunction.getFunction(Core.LLVMGetParamParent(instance));
	}
	
	public LLVMArgument getNextParameter() {
		return getArgument(Core.LLVMGetNextParam(instance));
	}
	
	public LLVMArgument getPreviousParameter() {
		return getArgument(Core.LLVMGetPreviousParam(instance));
	}
	
	public void addAttribute(LLVMAttribute attr) {
		Core.LLVMAddAttribute(instance,attr);
	}
	
	public void removeAttribute(LLVMAttribute attr) {
		Core.LLVMRemoveAttribute(instance,attr);
	}
	
	public void setParameterAlignment(long alignment) {
		Core.LLVMSetParamAlignment(instance,alignment);
	}
	
	public static LLVMArgument getArgument(SWIGTYPE_p_LLVMOpaqueValue val) {
		return val == null ? new LLVMArgument(null)
		                   : (LLVMArgument)LLVMValue.getValue(val);
	}
}
