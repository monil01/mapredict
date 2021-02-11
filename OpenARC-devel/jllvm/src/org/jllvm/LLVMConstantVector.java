package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;
import org.jllvm.bindings.SWIGTYPE_p_p_LLVMOpaqueValue;

public class LLVMConstantVector extends LLVMConstant {
	public static LLVMConstantVector get(LLVMConstant[] elements) {
		SWIGTYPE_p_p_LLVMOpaqueValue params = Core.new_LLVMValueRefArray(elements.length);
		for(int i=0;i<elements.length;i++)
			Core.LLVMValueRefArray_setitem(params,i,elements[i].instance);
		LLVMConstantVector result = (LLVMConstantVector)LLVMValue.getValue(Core.LLVMConstVector(params,elements.length));
		Core.delete_LLVMValueRefArray(params);
		assert(Core.LLVMIsConstant(result.instance) != 0);
		return result;
	}
	
	public LLVMConstantExpression extractElement(LLVMConstantInteger index) {
		return (LLVMConstantExpression)LLVMValue.getValue(Core.LLVMConstExtractElement(instance,index.instance));
	}
	
	public LLVMConstantExpression insertElement(LLVMConstant element,LLVMConstantInteger index) {
		assert(element.typeOf().equals(((LLVMVectorType)typeOf()).getElementType()));
		return (LLVMConstantExpression)LLVMValue.getValue(Core.LLVMConstInsertElement(instance,element.instance,index.instance));
	}

	public LLVMConstantVector(SWIGTYPE_p_LLVMOpaqueValue val) {
		super(val);
	}
}
