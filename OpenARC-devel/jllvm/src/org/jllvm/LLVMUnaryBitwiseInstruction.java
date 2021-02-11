package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMUnaryBitwiseInstruction extends LLVMInstruction {
	public enum UnaryBitwiseInstructionType {NOT,NEGATIVE};
	
	protected UnaryBitwiseInstructionType instructionType;
	
	public UnaryBitwiseInstructionType getInstructionType() {
		return instructionType;
	}
	
	private static SWIGTYPE_p_LLVMOpaqueValue buildInstruction(UnaryBitwiseInstructionType type,LLVMInstructionBuilder builder,LLVMValue val,String name) {
		switch(type) {
			case NOT:
				return Core.LLVMBuildNot(builder.getInstance(),val.getInstance(),name);
			case NEGATIVE:
				return Core.LLVMBuildNeg(builder.getInstance(),val.getInstance(),name);
		}
		throw new IllegalStateException();
	}
	
	/**
	 * Create an instruction that behaves like the specified unary bitwise
	 * operation. In other words, this is similar to a pseudo-op.
	 * 
	 * <p>
	 * Though the object is cached like all {@link LLVMInstruction} objects,
	 * jllvm never retrieves it from the cache. Instead, when LLVM returns the
	 * actual instruction created for the object, jllvm creates or returns from
	 * the cache a jllvm wrapper of the type of the actual instruction instead.
	 * For {@link UnaryBitwiseInstructionType#NOT}, it's an
	 * {@link LLVMXorInstruction}. For
	 * {@link UnaryBitwiseInstructionType#NEGATIVE}, it's an
	 * {@link LLVMSubtractInstruction}.
	 * </p>
	 */
	public LLVMUnaryBitwiseInstruction(String name,UnaryBitwiseInstructionType type,LLVMInstructionBuilder builder,LLVMValue val) {
		super(buildInstruction(type,builder,val,name));
		instructionType = type;
	}
} 
