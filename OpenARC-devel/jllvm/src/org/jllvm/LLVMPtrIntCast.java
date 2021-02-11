package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMPtrIntCast extends LLVMCastInstruction {
	public enum PtrIntCastType {PTR_TO_INT,INT_TO_PTR};
	
	protected PtrIntCastType castType;
	
	public PtrIntCastType getCastType() {
		return castType;
	}
	
	private static SWIGTYPE_p_LLVMOpaqueValue buildInstruction(LLVMInstructionBuilder builder,LLVMValue val,LLVMType destType,String name,PtrIntCastType type) {
		switch(type) {
			case PTR_TO_INT: {
				assert(destType instanceof LLVMIntegerType);
				return Core.LLVMBuildPtrToInt(builder.getInstance(),val.getInstance(),destType.getInstance(),name);
			}
			case INT_TO_PTR: {
				assert(destType instanceof LLVMPointerType);
				return Core.LLVMBuildIntToPtr(builder.getInstance(),val.getInstance(),destType.getInstance(),name);
			}
		}
		throw new IllegalStateException();
	}
	
	public static LLVMUser create(LLVMInstructionBuilder builder,String name,LLVMValue val,LLVMType destType,PtrIntCastType type) {
		return (LLVMUser)LLVMValue.getValue(buildInstruction(builder,val,destType,name,type));
	}
	
	public LLVMPtrIntCast(SWIGTYPE_p_LLVMOpaqueValue c, PtrIntCastType type) {
		super(c);
		castType = type;
	}
}
