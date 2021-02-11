package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMShiftInstruction extends LLVMInstruction {
	public enum ShiftType {SHL,LOGICAL_SHR,ARITHMETIC_SHR}
	
	protected ShiftType shiftType;
	
	public ShiftType getShiftType() {
		return shiftType;
	}
	
	private static SWIGTYPE_p_LLVMOpaqueValue buildInstruction(ShiftType type,LLVMInstructionBuilder builder,LLVMValue lhs,LLVMValue rhs,String name) {
		switch(type) {
			case SHL:
				return Core.LLVMBuildShl(builder.getInstance(),lhs.getInstance(),rhs.getInstance(),name);
			case LOGICAL_SHR:
				return Core.LLVMBuildLShr(builder.getInstance(),lhs.getInstance(),rhs.getInstance(),name);
			case ARITHMETIC_SHR:
				return Core.LLVMBuildAShr(builder.getInstance(),lhs.getInstance(),rhs.getInstance(),name);
		}
		throw new IllegalStateException();
	}
	
	public static LLVMUser create(LLVMInstructionBuilder builder,String name,ShiftType type,LLVMValue lhs,LLVMValue rhs) {
		return (LLVMUser)LLVMValue.getValue(buildInstruction(type, builder, lhs, rhs, name));
	}
	
	public LLVMShiftInstruction(SWIGTYPE_p_LLVMOpaqueValue val, ShiftType type) {
		super(val);
		shiftType = type;
	}
}
