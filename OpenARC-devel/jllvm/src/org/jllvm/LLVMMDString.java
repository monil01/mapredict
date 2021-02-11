package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;
import org.jllvm.bindings.SWIGTYPE_p_unsigned_int;

public class LLVMMDString extends LLVMValue {
	public static LLVMMDString get(LLVMContext context,String str) {
		return (LLVMMDString)LLVMValue.getValue(Core.LLVMMDStringInContext(context.getInstance(), str, str.length()));
	}
	
	public String getString() {
		// TODO: We need to somehow tell SWIG to use NewString not NewStringUTF
		// when building the result String in Core_wrap.c because there's no
		// guarantee of null termination. For now, the length, which is returned
		// via the second parameter here, is simply ignored.
		SWIGTYPE_p_unsigned_int len = Core.new_UnsignedIntArray(1);
		String res = Core.LLVMGetMDString(instance, len);
		Core.delete_UnsignedIntArray(len);
		return res;
	}
	
	public LLVMMDString(SWIGTYPE_p_LLVMOpaqueValue val) {
		super(val);
	}
	
	public static LLVMMDString getMDString(SWIGTYPE_p_LLVMOpaqueValue val) {
		return val == null ? new LLVMMDString(null)
		                   : (LLVMMDString)LLVMValue.getValue(val);
	}
}
