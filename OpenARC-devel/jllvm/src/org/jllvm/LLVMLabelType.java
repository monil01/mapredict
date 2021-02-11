package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.LLVMTypeKind;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueType;

public class LLVMLabelType extends LLVMType {
	// Avoid using global context: see LLVMContext documentation.
	//public static LLVMLabelType get() {
	//	return (LLVMLabelType)getType(Core.LLVMLabelType());
	//}
	
	public static LLVMLabelType get(LLVMContext ctxt) {
		return (LLVMLabelType)getType(Core.LLVMLabelTypeInContext(ctxt.getInstance()));
	}
	
	public LLVMLabelType(SWIGTYPE_p_LLVMOpaqueType t) {
		super(t);
		assert(Core.LLVMGetTypeKind(t) == LLVMTypeKind.LLVMLabelTypeKind);
	}
	
	public String toString() {
		return "label";
	}
	
	public String toStringForIntrinsic() {
		throw new IllegalStateException("unexpected type for overloaded intrinsic");
	}
}
