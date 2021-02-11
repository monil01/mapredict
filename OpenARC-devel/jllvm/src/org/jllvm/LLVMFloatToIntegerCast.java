package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMFloatToIntegerCast extends LLVMCastInstruction {
	public enum FPToIntCastType {SIGNED,UNSIGNED};
	
	protected FPToIntCastType castType;
	
	public FPToIntCastType getCastType() {
		return castType;
	}
	
	private static SWIGTYPE_p_LLVMOpaqueValue buildInstruction(LLVMInstructionBuilder builder,LLVMValue val,LLVMIntegerType destType,String name,FPToIntCastType type) {
		switch(type) {
			case SIGNED:
				return Core.LLVMBuildFPToSI(builder.getInstance(),val.getInstance(),destType.getInstance(),name);
			case UNSIGNED:
				return Core.LLVMBuildFPToUI(builder.getInstance(),val.getInstance(),destType.getInstance(),name);
		}
		throw new IllegalStateException();
	}
	
	public static LLVMUser create(LLVMInstructionBuilder builder,String name,LLVMValue val,LLVMIntegerType destType,FPToIntCastType type) {
		return (LLVMUser)LLVMValue.getValue(buildInstruction(builder,val,destType,name,type));
	}
	
	public LLVMFloatToIntegerCast(SWIGTYPE_p_LLVMOpaqueValue c, FPToIntCastType type) {
		super(c);
		castType = type;
	}
}
