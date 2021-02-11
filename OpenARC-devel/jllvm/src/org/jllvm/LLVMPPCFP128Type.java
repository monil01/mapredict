package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueType;
import org.jllvm.bindings.LLVMTypeKind;

public class LLVMPPCFP128Type extends LLVMRealType {
	// Avoid using global context: see LLVMContext documentation.
	//public static LLVMPPCFP128Type get() {
	//	return (LLVMPPCFP128Type)getType(Core.LLVMPPCFP128Type());
	//}
	
	public static LLVMPPCFP128Type get(LLVMContext ctxt) {
		return (LLVMPPCFP128Type)getType(Core.LLVMPPCFP128TypeInContext(ctxt.getInstance()));
	}
	
	public LLVMPPCFP128Type(SWIGTYPE_p_LLVMOpaqueType t) {
		super(t);
		assert(Core.LLVMGetTypeKind(t) == LLVMTypeKind.LLVMPPC_FP128TypeKind);
	}
	
	@Override
	public long getPrimitiveSizeInBits() {
		return 128;
	}
	
	public String toString() {
		return "ppc_fp128";
	}
}
