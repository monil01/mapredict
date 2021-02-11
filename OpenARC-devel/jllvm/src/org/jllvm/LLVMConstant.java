package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;
import org.jllvm.bindings.SWIGTYPE_p_p_LLVMOpaqueValue;
import org.jllvm.LLVMUser;

/* Implements all the methods from Core.h for constants. */
public abstract class LLVMConstant extends LLVMUser {
	public boolean isNullValue() {
		return Core.LLVMIsNull(instance) != 0;
	}
	
	public static LLVMConstant allOnes(LLVMType type) {
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstAllOnes(type.getInstance()));
	}
	
	public static LLVMConstant constNull(LLVMType type) {
		SWIGTYPE_p_LLVMOpaqueValue val = Core.LLVMConstNull(type.getInstance());
		if(type instanceof LLVMPointerType)
			return (LLVMConstantPointer)LLVMValue.getValue(val);
		else
			return (LLVMConstant)LLVMValue.getValue(val);
	}
	
	public static LLVMConstant getUndef(LLVMType type) {
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMGetUndef(type.getInstance()));
	}
	
	public boolean isUndef() {
		return Core.LLVMIsUndef(instance) != 0;
	}
	
	public LLVMConstant negative() {
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstNeg(instance));
	}
	
	public LLVMConstant not() {
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstNot(instance));
	}
	
	public LLVMConstantExpression getElementPointer(LLVMConstant[] indices) {
		SWIGTYPE_p_p_LLVMOpaqueValue params = Core.new_LLVMValueRefArray(indices.length);
		for(int i=0;i<indices.length;i++)
			Core.LLVMValueRefArray_setitem(params,i,indices[i].instance);
		LLVMConstantExpression result = (LLVMConstantExpression)LLVMValue.getValue(Core.LLVMConstGEP(instance,params,indices.length));
		Core.delete_LLVMValueRefArray(params);
		return result;
	}
	
	public LLVMConstant bitCast(LLVMType targetType) {
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstBitCast(instance,targetType.getInstance()));
	}
	
	public LLVMConstant(SWIGTYPE_p_LLVMOpaqueValue val) {
		super(val);
		assert(val == null || Core.LLVMIsConstant(val) != 0);
	}
}
