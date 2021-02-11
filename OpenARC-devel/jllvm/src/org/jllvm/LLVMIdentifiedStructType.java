package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueType;
import org.jllvm.bindings.SWIGTYPE_p_p_LLVMOpaqueType;
import org.jllvm.bindings.LLVMTypeKind;

import org.jllvm.LLVMStructType;

/* Implements operations on identified struct types from Core.h */
public class LLVMIdentifiedStructType extends LLVMStructType {
	//Creates an identified struct, which can later become recursive
	public LLVMIdentifiedStructType(LLVMContext context, String name) {
		this(Core.LLVMStructCreateNamed(context.getInstance(),name));
	}
	
	public LLVMIdentifiedStructType(LLVMContext context) {
		this(Core.LLVMStructCreateNamed(context.getInstance(),""));
	}
	
	// Avoid using global context: see LLVMContext documentation.
	//public LLVMIdentifiedStructType(String name) {
	//	this(Core.LLVMStructCreateNamed(LLVMContext.getGlobalContext().getInstance(),name));
	//}
	
	public LLVMIdentifiedStructType(SWIGTYPE_p_LLVMOpaqueType t) {
		super(t);
		assert(t == null || Core.LLVMGetTypeKind(t) == LLVMTypeKind.LLVMStructTypeKind);
	}
	
	public String getName() {
		return Core.LLVMGetStructName(instance);
	}
	
	public void setBody(boolean packed,LLVMType... elementTypes) {
		SWIGTYPE_p_p_LLVMOpaqueType elements = Core.new_LLVMTypeRefArray(elementTypes.length);
		for(int i=0;i<elementTypes.length;i++)
			Core.LLVMTypeRefArray_setitem(elements,i,elementTypes[i].getInstance());
		Core.LLVMStructSetBody(instance,elements,elementTypes.length,packed ? 1 : 0);
		Core.delete_LLVMTypeRefArray(elements);
	}
	
	public boolean isOpaque() {
		return (Core.LLVMIsOpaqueStruct(instance) > 0);
	}
	
	public static LLVMIdentifiedStructType getIdentifiedStructType(SWIGTYPE_p_LLVMOpaqueType tr) {
		return tr == null ? new LLVMIdentifiedStructType((SWIGTYPE_p_LLVMOpaqueType)null)
		                  : (LLVMIdentifiedStructType)LLVMType.getType(tr);
	}
	
	public String toString() {
		return getName();
	}
}
