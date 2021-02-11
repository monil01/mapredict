package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueType;
import org.jllvm.bindings.LLVMTypeKind;

/* Implements all methods for array types specified in Core.h */
public class LLVMArrayType extends LLVMSequenceType {
	public static LLVMArrayType get(LLVMType element_type,long num_elements) {
		return (LLVMArrayType)LLVMType.getType(Core.LLVMArrayType(element_type.getInstance(),num_elements));
	}
	
	public LLVMArrayType(SWIGTYPE_p_LLVMOpaqueType t) {
		super(t);
		assert(Core.LLVMGetTypeKind(t) == LLVMTypeKind.LLVMArrayTypeKind);
	}
	
	public long getLength() {
		return Core.LLVMGetArrayLength(instance);
	}
	
	public String toString() {
		return "[" + getLength() + " x " + getElementType() + "]";
	}
	
	public String toStringForIntrinsic() {
		throw new IllegalStateException("unexpected type for overloaded intrinsic");
	}
}
