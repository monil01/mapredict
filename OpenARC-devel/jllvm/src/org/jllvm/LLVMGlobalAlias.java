package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

public class LLVMGlobalAlias extends LLVMGlobalValue {
	public LLVMGlobalAlias(LLVMModule parent,LLVMType type,LLVMConstant aliasee,String name) {
		this(Core.LLVMAddAlias(parent != null ? parent.getInstance() : null,type.getInstance(),aliasee.getInstance(),name));
	}
	public LLVMGlobalAlias(SWIGTYPE_p_LLVMOpaqueValue val) {
		super(val);
	}
}
