package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMExtendCast extends LLVMCastInstruction {
	public enum ExtendType { ZERO,SIGN,FLOAT };
	
	protected ExtendType instructionType;
	
	public ExtendType getInstructionType() {
		return instructionType;
	}
	
	private static SWIGTYPE_p_LLVMOpaqueValue buildInstruction(ExtendType type,LLVMInstructionBuilder builder,LLVMValue val,LLVMType destType,String name) {
		assert((destType instanceof LLVMIntegerType && val.typeOf() instanceof LLVMIntegerType) ||
		       (destType instanceof LLVMRealType && val.typeOf() instanceof LLVMRealType));
		if(destType instanceof LLVMIntegerType && val.typeOf() instanceof LLVMIntegerType)
			assert(((LLVMIntegerType)destType).getWidth() > ((LLVMIntegerType)val.typeOf()).getWidth());
		switch(type) {
			case ZERO:
				return Core.LLVMBuildZExt(builder.getInstance(),val.getInstance(),destType.getInstance(),name);
			case SIGN:
				return Core.LLVMBuildSExt(builder.getInstance(),val.getInstance(),destType.getInstance(),name);
			case FLOAT:
				return Core.LLVMBuildFPExt(builder.getInstance(),val.getInstance(),destType.getInstance(),name);
		}
		throw new IllegalStateException();
	}
	
	public static LLVMUser create(LLVMInstructionBuilder builder,String name,LLVMValue val,LLVMType destType,ExtendType type) {
		return (LLVMUser)LLVMValue.getValue(buildInstruction(type, builder, val, destType, name));
	}
	
	public LLVMExtendCast(SWIGTYPE_p_LLVMOpaqueValue c, ExtendType type) {
		super(c);
		instructionType = type;
	}
}
