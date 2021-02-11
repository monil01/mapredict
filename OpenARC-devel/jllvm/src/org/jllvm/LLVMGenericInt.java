package org.jllvm;

import org.jllvm.bindings.ExecutionEngine;

public class LLVMGenericInt extends LLVMGenericValue {
	public LLVMGenericInt(LLVMType t,java.math.BigInteger n,boolean isSigned) {
		super(ExecutionEngine.LLVMCreateGenericValueOfInt(t.getInstance(),n,isSigned ? 1 : 0));
	}
	
	public java.math.BigInteger toInt(boolean isSigned) {
		return ExecutionEngine.LLVMGenericValueToInt(instance,isSigned ? 1 : 0);
	}
	
	public long intWidth() {
		return ExecutionEngine.LLVMGenericValueIntWidth(instance);
	}
	
	public String toString() {
		return toInt(false).toString();
	}
}
