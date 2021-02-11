package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;
import java.math.BigInteger;

public class LLVMConstantBoolean extends LLVMConstantInteger {
	public LLVMConstantBoolean(SWIGTYPE_p_LLVMOpaqueValue c) {
		super(c);
		assert(typeOf() instanceof LLVMIntegerType);
		assert(((LLVMIntegerType)typeOf()).getWidth() == 1);
	}
	
	// Avoid using global context: see LLVMContext documentation.
	//public static LLVMConstantBoolean get(boolean val) {
	//	return (LLVMConstantBoolean)LLVMValue.getValue(Core.LLVMConstInt(Core.LLVMInt1Type(),BigInteger.valueOf(val ? 1 : 0),1));
	//}
	
	public static LLVMConstantBoolean get(boolean val, LLVMContext ctxt) {
		return (LLVMConstantBoolean)LLVMValue.getValue(Core.LLVMConstInt(Core.LLVMInt1TypeInContext(ctxt.getInstance()),BigInteger.valueOf(val ? 1 : 0),1));
	}
}
