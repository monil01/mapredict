package org.jllvm;

import org.jllvm.LLVMPointerType;
import org.jllvm.bindings.Core;
import org.jllvm.bindings.CoreJNI;
import org.jllvm.bindings.LLVMAttribute;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;
import org.jllvm.bindings.SWIGTYPE_p_p_LLVMOpaqueValue;
import org.jllvm.bindings.SWIGTYPE_p_p_LLVMOpaqueBasicBlock;

public class LLVMFunction extends LLVMGlobalValue {
	public LLVMFunction(LLVMModule mod,String name,LLVMFunctionType funcType) {
		this(Core.LLVMAddFunction(mod != null ? mod.getInstance() : null,name,funcType.getInstance()));
	}
	
	public LLVMFunction(SWIGTYPE_p_LLVMOpaqueValue val) {
		super(val);
		assert(val == null || Core.LLVMIsAFunction(val) != null);
	}
	
	public LLVMFunction getNextFunction() {
		return getFunction(Core.LLVMGetNextFunction(instance));
	}
	
	public LLVMFunction getPreviousFunction() {
		return getFunction(Core.LLVMGetPreviousFunction(instance));
	}
	
	public long getIntrinsicID() {
		return Core.LLVMGetIntrinsicID(instance);
	}
	
	public long getCallingConvention() {
		return Core.LLVMGetFunctionCallConv(instance);
	}
	
	public void setCallingConvention(long callconv) {
		assert(callconv >= 0);
		Core.LLVMSetFunctionCallConv(instance,callconv);
	}
	
	public String getGC() {
		return Core.LLVMGetGC(instance);
	}
	
	public void setGC(String name) {
		Core.LLVMSetGC(instance,name);
	}
	
	public long countParameters() {
		return Core.LLVMCountParams(instance);
	}
	
	public LLVMArgument[] getParameters() {
		int num_parameters = (int)countParameters();
		SWIGTYPE_p_p_LLVMOpaqueValue params = Core.new_LLVMValueRefArray(num_parameters);
		Core.LLVMGetParams(instance,params);
		LLVMArgument[] result = new LLVMArgument[num_parameters];
		for(int i=0;i<num_parameters;i++)
			result[i] = LLVMArgument.getArgument(Core.LLVMValueRefArray_getitem(params,i));
		Core.delete_LLVMValueRefArray(params);
		return result;
	}
	
	public LLVMArgument getParameter(long i) {
		return LLVMArgument.getArgument(Core.LLVMGetParam(instance,i));
	}
	
	public LLVMArgument getFirstParameter() {
		return LLVMArgument.getArgument(Core.LLVMGetFirstParam(instance));
	}
	
	public LLVMArgument getLastParameter() {
		return LLVMArgument.getArgument(Core.LLVMGetLastParam(instance));
	}
	
	public long countBasicBlocks() {
		return Core.LLVMCountBasicBlocks(instance);
	}
	
	public LLVMBasicBlock[] getBasicBlocks() {
		int num_blocks = (int)countBasicBlocks();
		LLVMBasicBlock[] blocks = new LLVMBasicBlock[num_blocks];
		SWIGTYPE_p_p_LLVMOpaqueBasicBlock bbs = Core.new_LLVMBasicBlockRefArray(num_blocks);
		Core.LLVMGetBasicBlocks(instance,bbs);
		for(int i=0;i<num_blocks;i++)
			blocks[i] = LLVMBasicBlock.getBasicBlock(Core.LLVMBasicBlockRefArray_getitem(bbs,i));
		Core.delete_LLVMBasicBlockRefArray(bbs);
		return blocks;
	}
	
	public LLVMBasicBlock getFirstBasicBlock() {
		return LLVMBasicBlock.getBasicBlock(Core.LLVMGetFirstBasicBlock(instance));
	}
	
	public LLVMBasicBlock getLastBasicBlock() {
		return LLVMBasicBlock.getBasicBlock(Core.LLVMGetLastBasicBlock(instance));
	}
	
	public LLVMBasicBlock getEntryBasicBlock() {
		return LLVMBasicBlock.getBasicBlock(Core.LLVMGetEntryBasicBlock(instance));
	}
	
	// Avoid using global context: see LLVMContext documentation.
	//public LLVMBasicBlock appendBasicBlock(String name) {
	//	return new LLVMBasicBlock(Core.LLVMAppendBasicBlock(instance,name));
	//}
	
	public LLVMBasicBlock appendBasicBlock(String name, LLVMContext ctxt) {
		return new LLVMBasicBlock(Core.LLVMAppendBasicBlockInContext(ctxt.getInstance(),instance,name));
	}
	
	public static LLVMFunction getFunction(SWIGTYPE_p_LLVMOpaqueValue f) {
		return f == null ? new LLVMFunction(null)
		                 : (LLVMFunction)LLVMValue.getValue(f);
	}
	
	public LLVMFunctionType getFunctionType() {
		LLVMPointerType pointerType = (LLVMPointerType)typeOf();
		return (LLVMFunctionType)pointerType.getElementType();
	}
	
	public void addAttribute(LLVMAttribute attr) {
		Core.LLVMAddFunctionAttr(instance, attr);
	}
	
	public boolean hasAttribute(LLVMAttribute attr) {
		return 0 != (CoreJNI.LLVMGetFunctionAttr(instance.getCPtr())
		             & attr.swigValue());
	}
	
	public void removeAttribute(LLVMAttribute attr) {
		Core.LLVMRemoveFunctionAttr(instance, attr);
	}
	
	public void delete() {
		Core.LLVMDeleteFunction(instance);
	}
}
