package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMFreeInstruction extends LLVMInstruction {
	private static SWIGTYPE_p_LLVMOpaqueValue buildInstruction(LLVMInstructionBuilder builder,LLVMValue pointerValue) {
		assert(pointerValue.typeOf() instanceof LLVMPointerType);
		return Core.LLVMBuildFree(builder.getInstance(),pointerValue.getInstance());
	}
	/**
	 * Create multiple instructions including a free call.
	 * 
	 * <p>
	 * Though the object is cached like all {@link LLVMInstruction} objects,
	 * jllvm never retrieves it from the cache. Instead, when LLVM returns one of
	 * the instructions created for the object, jllvm creates or returns from the
	 * cache a jllvm wrapper for that individual instruction instead.
	 * </p>
	 */
	public LLVMFreeInstruction(LLVMInstructionBuilder builder,LLVMValue pointerValue) {
		super(buildInstruction(builder,pointerValue));
	}
}
