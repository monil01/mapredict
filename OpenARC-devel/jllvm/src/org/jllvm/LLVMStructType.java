package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueType;
import org.jllvm.bindings.SWIGTYPE_p_p_LLVMOpaqueType;
import org.jllvm.bindings.LLVMTypeKind;

import org.jllvm.LLVMType;

/* Implements all operations on struct types from Core.h */
public class LLVMStructType extends LLVMType {
	// Avoid using global context: see LLVMContext documentation.
	//public static LLVMStructType get(LLVMType[] elements,boolean packed) {
	//	SWIGTYPE_p_p_LLVMOpaqueType elmnts = Core.new_LLVMTypeRefArray(elements.length);
	//	for(int i=0;i<elements.length;i++)
	//		Core.LLVMTypeRefArray_setitem(elmnts,i,elements[i].getInstance());
	//	SWIGTYPE_p_LLVMOpaqueType tr = Core.LLVMStructType(elmnts,elements.length,packed ? 0 : 1);
	//	Core.delete_LLVMTypeRefArray(elmnts);
	//	return (LLVMStructType)LLVMType.getType(tr);
	//}
	
	public static LLVMStructType get(LLVMContext context,boolean packed,LLVMType... elements) {
		SWIGTYPE_p_p_LLVMOpaqueType elmnts = Core.new_LLVMTypeRefArray(elements.length);
		for(int i=0;i<elements.length;i++)
			Core.LLVMTypeRefArray_setitem(elmnts,i,elements[i].getInstance());
		SWIGTYPE_p_LLVMOpaqueType tr = Core.LLVMStructTypeInContext(context.getInstance(),elmnts,elements.length,packed ? 0 : 1);
		Core.delete_LLVMTypeRefArray(elmnts);
		return (LLVMStructType)LLVMType.getType(tr);
	}
	
	public LLVMStructType(SWIGTYPE_p_LLVMOpaqueType t) {
		super(t);
		assert(t == null || Core.LLVMGetTypeKind(t) == LLVMTypeKind.LLVMStructTypeKind);
	}
	
	public long countElementTypes() {
		return Core.LLVMCountStructElementTypes(instance);
	}
	
	public LLVMType[] getElementTypes() {
		int num_elements = (int)countElementTypes();
		SWIGTYPE_p_p_LLVMOpaqueType elements = Core.new_LLVMTypeRefArray(num_elements);
		Core.LLVMGetStructElementTypes(instance,elements);
		LLVMType[] result = new LLVMType[num_elements];
		for(int i=0;i<result.length;i++)
			result[i] = LLVMType.getType(Core.LLVMTypeRefArray_getitem(elements,i));
		Core.delete_LLVMTypeRefArray(elements);
		return result;
	}
	
	public boolean isPacked() {
		return Core.LLVMIsPackedStruct(instance) > 0;
	}
	
	public String toString() {
		StringBuilder res = new StringBuilder();
		boolean isPacked = isPacked();
		if (isPacked)
			res.append("<");
		res.append("{");
		LLVMType[] elements = getElementTypes();
		for (int i=0; i<elements.length; i++) {
			if (i > 0)
				res.append(", ");
			res.append(elements[i]);
		}
		res.append("}");
		if (isPacked)
			res.append(">");
		return res.toString();
	}
	
	public String toStringForIntrinsic() {
		throw new IllegalStateException("unexpected type for overloaded intrinsic");
	}
}
