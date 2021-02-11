package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueType;
import org.jllvm.bindings.LLVMTypeKind;

public class LLVMVoidType extends LLVMType {
	// Avoid using global context: see LLVMContext documentation.
	//public static LLVMVoidType get() {
	//	return (LLVMVoidType)getType(Core.LLVMVoidType());
	//}
	
	public static LLVMVoidType get(LLVMContext context) {
		return (LLVMVoidType)getType(Core.LLVMVoidTypeInContext(context.getInstance()));
	}
	
	public LLVMVoidType(SWIGTYPE_p_LLVMOpaqueType t) {
		super(t);
		assert(Core.LLVMGetTypeKind(t) == LLVMTypeKind.LLVMVoidTypeKind);
	}

	public String toString() {
		return "void";
	}
	
	public String toStringForIntrinsic() {
		throw new IllegalStateException("unexpected type for overloaded intrinsic");
	}
}
