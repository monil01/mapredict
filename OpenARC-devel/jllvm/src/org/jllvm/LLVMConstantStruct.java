package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;
import org.jllvm.bindings.SWIGTYPE_p_p_LLVMOpaqueValue;

public class LLVMConstantStruct extends LLVMConstantAggregate {
	// Avoid using global context: see LLVMContext documentation.
	//public static LLVMConstantStruct get(LLVMConstant[] elements,boolean packed) {
	//	SWIGTYPE_p_p_LLVMOpaqueValue params = Core.new_LLVMValueRefArray(elements.length);
	//	for(int i=0;i<elements.length;i++)
	//		Core.LLVMValueRefArray_setitem(params,i,elements[i].instance);
	//	SWIGTYPE_p_LLVMOpaqueValue struct = Core.LLVMConstStruct(params,elements.length,packed ? 1 : 0);
	//	Core.delete_LLVMValueRefArray(params);
	//	assert(Core.LLVMIsConstant(struct) != 0);
	//	return (LLVMConstantStruct)LLVMValue.getValue(struct);
	//}
	public static LLVMConstantStruct get(LLVMIdentifiedStructType ty, LLVMConstant[] elements) {
		SWIGTYPE_p_p_LLVMOpaqueValue params = Core.new_LLVMValueRefArray(elements.length);
		for(int i=0;i<elements.length;i++)
			Core.LLVMValueRefArray_setitem(params,i,elements[i].instance);
		SWIGTYPE_p_LLVMOpaqueValue struct = Core.LLVMConstNamedStruct(ty.getInstance(),params,elements.length);
		Core.delete_LLVMValueRefArray(params);
		// LLVMIsAConstantStruct is unreliable, so we check constant-ness and
		// struct-ness separately. See comment in LLVMValue.getValue.
		assert(Core.LLVMIsConstant(struct) != 0);
		assert(LLVMType.getType(Core.LLVMTypeOf(struct)) instanceof LLVMStructType);
		return (LLVMConstantStruct)LLVMValue.getValue(struct);
	}
	public static LLVMConstantStruct get(LLVMConstant[] elements,LLVMContext ctxt,boolean packed) {
		SWIGTYPE_p_p_LLVMOpaqueValue params = Core.new_LLVMValueRefArray(elements.length);
		for(int i=0;i<elements.length;i++)
			Core.LLVMValueRefArray_setitem(params,i,elements[i].instance);
		SWIGTYPE_p_LLVMOpaqueValue struct = Core.LLVMConstStructInContext(ctxt.getInstance(),params,elements.length,packed ? 1 : 0);
		Core.delete_LLVMValueRefArray(params);
		assert(Core.LLVMIsConstant(struct) != 0);
		return (LLVMConstantStruct)LLVMValue.getValue(struct);
	}
	public LLVMConstantStruct(SWIGTYPE_p_LLVMOpaqueValue val) {
		super(val);
	}
}
