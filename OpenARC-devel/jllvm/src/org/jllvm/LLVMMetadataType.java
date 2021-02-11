package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueType;
import org.jllvm.bindings.LLVMTypeKind;

public class LLVMMetadataType extends LLVMSequenceType {
	public static LLVMMetadataType get(LLVMContext context) {
		return (LLVMMetadataType)getType(Core.LLVMMetadataTypeInContext(context.getInstance()));
	}
	
	public LLVMMetadataType(SWIGTYPE_p_LLVMOpaqueType t) {
		super(t);
		assert(Core.LLVMGetTypeKind(t) == LLVMTypeKind.LLVMMetadataTypeKind);
	}
	
	public String toString() {
		return "<metadata>";
	}
	
	public String toStringForIntrinsic() {
		throw new IllegalStateException("unexpected type for overloaded intrinsic");
	}
}
