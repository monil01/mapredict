package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueType;
import org.jllvm.bindings.LLVMTypeKind;

public class LLVMFloatType extends LLVMRealType {
	// Avoid using global context: see LLVMContext documentation.
	//public static LLVMFloatType get() {
	//	return (LLVMFloatType)getType(Core.LLVMFloatType());
	//}
	
	public static LLVMFloatType get(LLVMContext context) {
		return (LLVMFloatType)getType(Core.LLVMFloatTypeInContext(context.getInstance()));
	}
	
	public LLVMFloatType(SWIGTYPE_p_LLVMOpaqueType t) {
		super(t);
		assert(Core.LLVMGetTypeKind(t) == LLVMTypeKind.LLVMFloatTypeKind);
	}
	
	@Override
	public long getPrimitiveSizeInBits() {
		return 32;
	}
	
	public String toString() {
		return "float";
	}
}
