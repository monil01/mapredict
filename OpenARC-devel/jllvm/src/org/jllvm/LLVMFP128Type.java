package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.LLVMTypeKind;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueType;

public class LLVMFP128Type extends LLVMRealType {
	// Avoid using global context: see LLVMContext documentation.
	//public static LLVMFP128Type get() {
	//	return (LLVMFP128Type)getType(Core.LLVMFP128Type());
	//}
	
	public static LLVMFP128Type get(LLVMContext ctxt) {
		return (LLVMFP128Type)getType(Core.LLVMFP128TypeInContext(ctxt.getInstance()));
	}
	
	public LLVMFP128Type(SWIGTYPE_p_LLVMOpaqueType t) {
		super(t);
		assert(Core.LLVMGetTypeKind(t) == LLVMTypeKind.LLVMFP128TypeKind);
	}
	
	@Override
	public long getPrimitiveSizeInBits() {
		return 128;
	}
	
	public String toString() {
		return "fp128";
	}
}
