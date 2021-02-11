package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMHeapAllocation extends LLVMAllocationInstruction {
	private static SWIGTYPE_p_LLVMOpaqueValue buildInstruction(LLVMInstructionBuilder builder,LLVMType type,LLVMValue number,String name) {
		if(number != null) {
			assert(number.typeOf() instanceof LLVMIntegerType);
			assert(((LLVMIntegerType)number.typeOf()).getWidth() == 32);
			return Core.LLVMBuildArrayMalloc(builder.getInstance(),type.getInstance(),number.getInstance(),name);
		}
		else 
			return Core.LLVMBuildMalloc(builder.getInstance(),type.getInstance(),name);
	}
	/**
	 * Create multiple instructions including a malloc call.
	 * 
	 * <p>
	 * Though the object is cached like all {@link LLVMInstruction} objects,
	 * jllvm never retrieves it from the cache. Instead, when LLVM returns one of
	 * the instructions created for the object, jllvm creates or returns from the
	 * cache a jllvm wrapper for that individual instruction instead.
	 * </p>
	 */
	public LLVMHeapAllocation(LLVMInstructionBuilder builder,String name,LLVMType type,LLVMValue number) {
		super(buildInstruction(builder, type, number, name));
	}
}
