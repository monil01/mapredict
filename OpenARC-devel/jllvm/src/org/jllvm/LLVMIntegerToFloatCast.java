package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMIntegerToFloatCast extends LLVMCastInstruction {
	public enum IntCastType {SIGNED,UNSIGNED};
	
	protected IntCastType castType;
	
	public IntCastType getCastType() {
		return castType;
	}
	
	private static SWIGTYPE_p_LLVMOpaqueValue buildInstruction(LLVMInstructionBuilder builder,LLVMValue val,LLVMRealType destType,String name,IntCastType type) {
		switch(type) {
			case SIGNED:
				return Core.LLVMBuildSIToFP(builder.getInstance(),val.getInstance(),destType.getInstance(),name);
			case UNSIGNED:
				return Core.LLVMBuildUIToFP(builder.getInstance(),val.getInstance(),destType.getInstance(),name);
		}
		throw new IllegalStateException();
	}
	
	public static LLVMUser create(LLVMInstructionBuilder builder,String name,LLVMValue val,LLVMRealType destType,IntCastType type) {
		return (LLVMUser)LLVMValue.getValue(buildInstruction(builder,val,destType,name,type));
	}
	
	public LLVMIntegerToFloatCast(SWIGTYPE_p_LLVMOpaqueValue c, IntCastType type) {
		super(c);
		castType = type;
	}
}
