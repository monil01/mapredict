package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMConstantReal extends LLVMConstant {
	public static LLVMConstantReal get(LLVMType realType,double N) {
		return (LLVMConstantReal)LLVMValue.getValue(Core.LLVMConstReal(realType.getInstance(),N));
	}
	
	public static LLVMConstantReal get(LLVMType realType,String text) {
		return (LLVMConstantReal)LLVMValue.getValue(Core.LLVMConstRealOfString(realType.getInstance(),text));
	}
	
	public LLVMConstantExpression truncate(LLVMRealType targetType) {
		return (LLVMConstantExpression)LLVMValue.getValue(Core.LLVMConstFPTrunc(instance,targetType.getInstance()));
	}
	
	public LLVMConstantExpression extend(LLVMRealType targetType) {
		return (LLVMConstantExpression)LLVMValue.getValue(Core.LLVMConstFPExt(instance,targetType.getInstance()));
	}
	
	public LLVMConstantExpression realToInteger(LLVMIntegerType targetType,boolean signed) {
		if(signed)
			return (LLVMConstantExpression)LLVMValue.getValue(Core.LLVMConstFPToSI(instance,targetType.getInstance()));
		else
			return (LLVMConstantExpression)LLVMValue.getValue(Core.LLVMConstFPToUI(instance,targetType.getInstance()));
	}
	
	public LLVMConstantReal(SWIGTYPE_p_LLVMOpaqueValue c) {
		super(c);
	}
}
