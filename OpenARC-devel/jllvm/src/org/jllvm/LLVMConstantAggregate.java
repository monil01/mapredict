package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;
import org.jllvm.bindings.SWIGTYPE_p_unsigned_int;

public abstract class LLVMConstantAggregate extends LLVMConstant {
	public LLVMConstant extractValue(long[] indices) {
		SWIGTYPE_p_unsigned_int params = Core.new_UnsignedIntArray(indices.length);
		for(int i=0;i<indices.length;i++)
			Core.UnsignedIntArray_setitem(params,i,indices[i]);
		LLVMConstant result = (LLVMConstant)LLVMValue.getValue(Core.LLVMConstExtractValue(instance,params,indices.length));
		Core.delete_UnsignedIntArray(params);
		return result;
	}
	
	public LLVMConstant insertValue(LLVMConstant value,long[] indices) {
		SWIGTYPE_p_unsigned_int params = Core.new_UnsignedIntArray(indices.length);
		for(int i=0;i<indices.length;i++)
			Core.UnsignedIntArray_setitem(params,i,indices[i]);
		LLVMConstant result = (LLVMConstant)LLVMValue.getValue(Core.LLVMConstInsertValue(instance,value.instance,params,indices.length));
		Core.delete_UnsignedIntArray(params);
		return result;
	}
	public LLVMConstantAggregate(SWIGTYPE_p_LLVMOpaqueValue val) {
		super(val);
	}
}
