package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.LLVMTypeKind;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueType;

public class LLVMX86FP80Type extends LLVMRealType {
	// Avoid using global context: see LLVMContext documentation.
	//public static LLVMX86FP80Type get() {
	//	return (LLVMX86FP80Type)getType(Core.LLVMX86FP80Type());
	//}
	
	public static LLVMX86FP80Type get(LLVMContext context) {
		return (LLVMX86FP80Type)getType(Core.LLVMX86FP80TypeInContext(context.getInstance()));
	}
	
	public LLVMX86FP80Type(SWIGTYPE_p_LLVMOpaqueType t) {
		super(t);
		assert(Core.LLVMGetTypeKind(t) == LLVMTypeKind.LLVMX86_FP80TypeKind);
	}
	
	@Override
	public long getPrimitiveSizeInBits() {
		return 80;
	}
	
	public String toString() {
		return "x86_fp80";
	}
}
