package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;
import org.jllvm.bindings.SWIGTYPE_p_p_LLVMOpaqueValue;
import org.jllvm.bindings.LLVMAttribute;
import org.jllvm.LLVMInstruction;

public class LLVMCallInstruction extends LLVMInstruction {
	public long getCallingConvention() {
		return Core.LLVMGetInstructionCallConv(instance);
	}
	
	public void setCallingConvention(long CC) {
		Core.LLVMSetInstructionCallConv(instance,CC);
	}

	public void addAttribute(long index,LLVMAttribute attr) {
		assert(index >= 0);
		Core.LLVMAddInstrAttribute(instance,index,attr);
	}
	
	public void removeAttribute(long index,LLVMAttribute attr) {
		assert(index >= 0);
		Core.LLVMRemoveInstrAttribute(instance,index,attr);
	}
	
	public void setParameterAlignment(long index,long alignment) {
		assert(index >= 0 && alignment >= 0);
		Core.LLVMSetInstrParamAlignment(instance,index,alignment);
	}
	
	public boolean isTailCall() {
		return Core.LLVMIsTailCall(instance) > 0;
	}
	
	public void setTailCall(boolean tailCall) {
		Core.LLVMSetTailCall(instance,tailCall ? 1 : 0);
	}
	
	private static SWIGTYPE_p_LLVMOpaqueValue buildInstruction(LLVMInstructionBuilder builder,LLVMValue func,LLVMValue[] arguments,String name) {
		assert(func.typeOf() instanceof LLVMPointerType);
		assert(((LLVMPointerType)func.typeOf()).getElementType() instanceof LLVMFunctionType);
		LLVMFunctionType fnType = (LLVMFunctionType)((LLVMPointerType)func.typeOf()).getElementType();
		assert(arguments.length == fnType.countParamTypes()
		       || (fnType.isVarArg() && fnType.countParamTypes() < arguments.length));
		SWIGTYPE_p_p_LLVMOpaqueValue args = Core.new_LLVMValueRefArray(arguments.length);
		for(int i=0;i<arguments.length;i++) {
			assert(fnType.countParamTypes() <= i || fnType.getParamTypes()[i] == arguments[i].typeOf());
			Core.LLVMValueRefArray_setitem(args,i,arguments[i].getInstance());
		}
		SWIGTYPE_p_LLVMOpaqueValue result = Core.LLVMBuildCall(builder.getInstance(),func.getInstance(),args,arguments.length,name);
		Core.delete_LLVMValueRefArray(args);
		return result;
	}
	
	/**
	 * @param name
	 *          the return value name, or the empty string to specify no return
	 *          value name. Having no return value name is required in the case
	 *          of a void return type.
	 */
	public LLVMCallInstruction(LLVMInstructionBuilder builder,String name,LLVMValue func,LLVMValue... arguments) {
		this(buildInstruction(builder,func,arguments,name));
	}
	
	public LLVMCallInstruction(SWIGTYPE_p_LLVMOpaqueValue c) {
		super(c);
	}
}
