package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;
import org.jllvm.bindings.SWIGTYPE_p_p_LLVMOpaqueValue;

public class LLVMConstantArray extends LLVMConstantAggregate {
	/**
	 * As in the LLVM C++ API's ConstantArray::get, this might not return an
	 * {@link LLVMConstantArray}.
	 */
	public static LLVMConstant get(LLVMType elementType,LLVMConstant... elements) {
		SWIGTYPE_p_p_LLVMOpaqueValue params = Core.new_LLVMValueRefArray(elements.length);
		for(int i=0;i<elements.length;i++)
			Core.LLVMValueRefArray_setitem(params,i,elements[i].instance);
		SWIGTYPE_p_LLVMOpaqueValue array = Core.LLVMConstArray(elementType.getInstance(),params,elements.length);
		Core.delete_LLVMValueRefArray(params);
		assert(Core.LLVMIsConstant(array) != 0);
		return (LLVMConstant)LLVMValue.getValue(array);
	}
	public static LLVMConstantArray get(LLVMContext ctxt,String str,boolean nullTerminate) {
		return (LLVMConstantArray)LLVMValue.getValue(Core.LLVMConstStringInContext(ctxt.getInstance(),str,str.length(),nullTerminate ? 0 : 1));
	}
	public LLVMConstantArray(SWIGTYPE_p_LLVMOpaqueValue val) {
		super(val);
	}
}
