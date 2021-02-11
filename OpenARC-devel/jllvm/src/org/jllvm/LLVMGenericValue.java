package org.jllvm;

import java.math.BigInteger;

import org.jllvm.bindings.ExecutionEngine;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueGenericValue;
import org.jllvm.bindings.SWIGTYPE_p_void;

public class LLVMGenericValue {
	protected SWIGTYPE_p_LLVMOpaqueGenericValue instance;
	
	public LLVMGenericValue(SWIGTYPE_p_LLVMOpaqueGenericValue val) {
		instance = val;
	}
	
	protected void finalize() {
		ExecutionEngine.LLVMDisposeGenericValue(instance);
	}
	
	public SWIGTYPE_p_LLVMOpaqueGenericValue getInstance() {
		return instance;
	}
	
	public BigInteger toInt(boolean IsSigned) {
		return ExecutionEngine.LLVMGenericValueToInt(instance,IsSigned ? 1 : 0);
	}
	
	public SWIGTYPE_p_void toPointer(LLVMType Ty) {
		return ExecutionEngine.LLVMGenericValueToPointer(instance);
	}
	
	public double toFloat(LLVMType Ty) {
		return ExecutionEngine.LLVMGenericValueToFloat(Ty.getInstance(),instance);
	}
}
