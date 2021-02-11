package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;
import org.jllvm.bindings.SWIGTYPE_p_p_LLVMOpaqueValue;

public class LLVMMDNode extends LLVMValue {
	public static LLVMMDNode get(LLVMContext context,LLVMValue... vals) {
		SWIGTYPE_p_p_LLVMOpaqueValue vals_ = Core.new_LLVMValueRefArray(vals.length);
		for(int i=0;i<vals.length;i++)
			Core.LLVMValueRefArray_setitem(vals_,i,vals[i].getInstance());
		SWIGTYPE_p_LLVMOpaqueValue res = Core.LLVMMDNodeInContext(context.getInstance(), vals_, vals.length);
		Core.delete_LLVMValueRefArray(vals_);
		return (LLVMMDNode)LLVMValue.getValue(res);
	}
	
	public static LLVMMDNode getTemporary(LLVMContext context,LLVMValue... vals) {
		SWIGTYPE_p_p_LLVMOpaqueValue vals_ = Core.new_LLVMValueRefArray(vals.length);
		for(int i=0;i<vals.length;i++)
			Core.LLVMValueRefArray_setitem(vals_,i,vals[i].getInstance());
		SWIGTYPE_p_LLVMOpaqueValue res = Core.LLVMMDNodeTemporaryInContext(context.getInstance(), vals_, vals.length);
		Core.delete_LLVMValueRefArray(vals_);
		return (LLVMMDNode)LLVMValue.getValue(res);
	}
	
	/**
	 * Must be called only on an {@link LLVMMDNode} created by
	 * {@link #getTemporary}.
	 */
	public void deleteTemporary() {
		Core.LLVMDeleteMDNodeTemporary(instance);
	}
	
	public LLVMMDNode(SWIGTYPE_p_LLVMOpaqueValue val) {
		super(val);
	}
	
	public long getNumOperands() {
		return Core.LLVMGetMDNodeNumOperands(instance);
	}
	
	public LLVMValue[] getOperands() {
		int n = (int)getNumOperands();
		SWIGTYPE_p_p_LLVMOpaqueValue ops = Core.new_LLVMValueRefArray(n);
		Core.LLVMGetMDNodeOperands(instance,ops);
		LLVMValue[] res = new LLVMValue[n];
		for(int i=0;i<n;i++)
			res[n] = LLVMValue.getValue(Core.LLVMValueRefArray_getitem(ops,i));
		Core.delete_LLVMValueRefArray(ops);
		return res;
	}
	
	public void replaceOperandWith(long I, LLVMValue Op) {
		Core.LLVMReplaceMDNodeOperandWith(instance, I, Op.getInstance());
	}
	
	public static LLVMMDNode getMDNode(SWIGTYPE_p_LLVMOpaqueValue val) {
		return val == null ? new LLVMMDNode(null)
		                   : (LLVMMDNode)LLVMValue.getValue(val);
	}
}
