package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueType;
import org.jllvm.bindings.LLVMTypeKind;

/* Implements all methods for pointer types specified in Core.h */
public class LLVMPointerType extends LLVMSequenceType {
	public static LLVMPointerType get(LLVMType element_type,long addressSpace) {
		return (LLVMPointerType)LLVMType.getType(Core.LLVMPointerType(element_type.getInstance(),addressSpace));
	}
	
	public LLVMPointerType(SWIGTYPE_p_LLVMOpaqueType t) {
		super(t);
		assert(Core.LLVMGetTypeKind(t) == LLVMTypeKind.LLVMPointerTypeKind);
	}
	
	public long getAddressSpace() {
		return Core.LLVMGetPointerAddressSpace(instance);
	}
	
	public String toString() {
		StringBuilder res = new StringBuilder(getElementType().toString());
		long addrspace = getAddressSpace();
		if (addrspace != 0) {
			res.append(" addrspace(");
			res.append(addrspace);
			res.append(")");
		}
		res.append("*");
		return res.toString();
	}
	
	public String toStringForIntrinsic() {
		StringBuilder res = new StringBuilder("p");
		res.append(getAddressSpace());
		res.append(getElementType());
		return res.toString();
	}
}
