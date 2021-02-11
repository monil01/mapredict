package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.LLVMTypeKind;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueType;

public class LLVMDoubleType extends LLVMRealType {
	// Avoid using global context: see LLVMContext documentation.
	//public static LLVMDoubleType get() {
	//	return (LLVMDoubleType)getType(Core.LLVMDoubleType());
	//}
	
	public static LLVMDoubleType get(LLVMContext context) {
		return (LLVMDoubleType)getType(Core.LLVMDoubleTypeInContext(context.getInstance()));
	}
	
	public LLVMDoubleType(SWIGTYPE_p_LLVMOpaqueType t) {
		super(t);
		assert(Core.LLVMGetTypeKind(t) == LLVMTypeKind.LLVMDoubleTypeKind);
	}
	
	@Override
	public long getPrimitiveSizeInBits() {
		return 64;
	}
	
	public String toString() {
		return "double";
	}
}
