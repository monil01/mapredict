package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMConstantInlineASM extends LLVMValue {
	public static LLVMConstantInlineASM get(LLVMFunctionType type,String asm,String constraints,boolean hasSideEffects,boolean isAlignStack) {
		return (LLVMConstantInlineASM)LLVMValue.getValue(Core.LLVMConstInlineAsm(type.getInstance(),asm,constraints,hasSideEffects ? 1 : 0,isAlignStack ? 1 : 0));
	}
	public LLVMConstantInlineASM(SWIGTYPE_p_LLVMOpaqueValue val) {
		super(val);
	}
}
