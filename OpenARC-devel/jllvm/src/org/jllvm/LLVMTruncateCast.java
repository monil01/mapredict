package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMTruncateCast extends LLVMCastInstruction {
	public enum TruncateType { FLOAT,INTEGER };
	
	protected TruncateType instructionType;
	
	public TruncateType getInstructionType() {
		return instructionType;
	}
	
	private static SWIGTYPE_p_LLVMOpaqueValue buildInstruction(TruncateType type,LLVMInstructionBuilder builder,LLVMValue val,LLVMType destType,String name) {
		switch(type) {
			case FLOAT: {
				assert(val.typeOf() instanceof LLVMRealType);
				assert(destType instanceof LLVMRealType);
				return Core.LLVMBuildFPTrunc(builder.getInstance(),val.getInstance(),destType.getInstance(),name);
			}
			case INTEGER: {
				assert(val.typeOf() instanceof LLVMIntegerType);
				assert(destType instanceof LLVMIntegerType);
				return Core.LLVMBuildTrunc(builder.getInstance(),val.getInstance(),destType.getInstance(),name);
			}
		}
		throw new IllegalStateException();
	}
	
	public static LLVMUser create(LLVMInstructionBuilder builder,String name,LLVMValue val,LLVMType destType,TruncateType type) {
		return (LLVMUser)LLVMValue.getValue(buildInstruction(type,builder,val,destType,name));
	}
	
	public LLVMTruncateCast(SWIGTYPE_p_LLVMOpaqueValue c, TruncateType type) {
		super(c);
		instructionType = type;
	}
}
