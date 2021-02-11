package org.jllvm;

import org.jllvm.bindings.ExecutionEngine;

public class LLVMGenericReal extends LLVMGenericValue {
	final LLVMRealType type;
	
	public LLVMGenericReal(LLVMRealType t,double n) {
		super(ExecutionEngine.LLVMCreateGenericValueOfFloat(t.getInstance(),n));
		type = t;
	}
	
	public double toReal() {
		return ExecutionEngine.LLVMGenericValueToFloat(type.getInstance(),instance);
	}
	
	public String toString() {
		return String.valueOf(toReal());
	}
}
