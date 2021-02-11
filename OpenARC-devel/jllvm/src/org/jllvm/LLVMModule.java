package org.jllvm;

import org.jllvm.bindings.Analysis;
import org.jllvm.bindings.BitWriter;
import org.jllvm.bindings.Core;
import org.jllvm.bindings.LLVMLinkerMode;
import org.jllvm.bindings.LLVMVerifierFailureAction;
import org.jllvm.bindings.Linker;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueModule;
import org.jllvm.bindings.SWIGTYPE_p_p_char;

/**
 * Fully translated all functions for Modules from Core.java/Core.h back to a
 * Module class. Also fully translated all functions for Modules from
 * BitWriter.java/Bitwriter.h to this Module class.
 * 
 * <p>
 * All jllvm objects of type {@link LLVMModule} are stored in a cache (in the
 * associated {@link LLVMContext} object). The approach is the same as the
 * approach for the {@link LLVMValue} hierarchy except there are no subtypes
 * in this case. See the {@link LLVMValue} documentation for a full
 * explanation. {@link #getModule} here corresponds to
 * {@link LLVMValue#getValue} there for cache lookup.
 * </p>
 */
public class LLVMModule {
	/**
	 * Same as {@link SWIGTYPE_p_LLVMOpaqueModule} except instances compare equal
	 * when and only when the underlying LLVM object addresses are equal.
	 * {@link #hashCode} is adjusted accordingly.
	 */
	public static class Hashable_LLVMOpaqueModule
		extends SWIGTYPE_p_LLVMOpaqueModule
	{
		/**
		 * Construct using the LLVM object address contained in an existing
		 * {@link SWIGTYPE_p_LLVMOpaqueModule}.
		 * 
		 * @param o
		 *          contains the LLVM object address to store
		 */
		public Hashable_LLVMOpaqueModule(SWIGTYPE_p_LLVMOpaqueModule o) {
			super(getCPtr(o), false);
		}

		@Override
		public int hashCode() {
			long cPtr = getCPtr(this);
			// This is the hashCode documented for Long.hashCode at
			// http://docs.oracle.com/javase/7/docs/api/java/lang/Long.html.
			// We reproduce the formula here instead of calling Long.hashCode so
			// we don't have to waste time constructing a Long object.
			return (int) (cPtr ^ (cPtr >>> 32));
		}
	
		@Override
		public boolean equals(java.lang.Object obj) {
			if (!(obj instanceof Hashable_LLVMOpaqueModule)) {
				return false;
			}
			return getCPtr(this) == getCPtr((Hashable_LLVMOpaqueModule) obj);
		}
	}

	private Hashable_LLVMOpaqueModule instance;
	
	public SWIGTYPE_p_LLVMOpaqueModule getInstance() {
		return instance;
	}
	
	public String getDataLayout() {
		return Core.LLVMGetDataLayout(instance);
	}
	
	public void setDataLayout(String triple) {
		Core.LLVMSetDataLayout(instance,triple);
	}
	
	public String getTargetTriple() {
		return Core.LLVMGetTarget(instance);
	} 
	
	public void setTargetTriple(String triple) {
		Core.LLVMSetTarget(instance,triple);
	}
	
	public void dump() {
		Core.LLVMDumpModule(instance);
	}
	
	public LLVMGlobalVariable addGlobal(LLVMType type,String name) {
		return (LLVMGlobalVariable)LLVMValue.getValue(Core.LLVMAddGlobal(instance,type.getInstance(),name));
	}
	
	public LLVMGlobalVariable getNamedGlobal(String name) {
		return LLVMGlobalVariable.getGlobalVariable(Core.LLVMGetNamedGlobal(instance,name));
	}
	
	public LLVMGlobalVariable getFirstGlobal() {
		return LLVMGlobalVariable.getGlobalVariable(Core.LLVMGetFirstGlobal(instance));
	}
	
	public LLVMGlobalVariable getLastGlobal() {
		return LLVMGlobalVariable.getGlobalVariable(Core.LLVMGetLastGlobal(instance));
	}
	
	public LLVMFunction getNamedFunction(String name) {
		return LLVMFunction.getFunction(Core.LLVMGetNamedFunction(instance,name));
	}
	
	public LLVMFunction getFirstFunction() {
		return LLVMFunction.getFunction(Core.LLVMGetFirstFunction(instance));
	}
	
	public LLVMFunction getLastFunction() {
		return LLVMFunction.getFunction(Core.LLVMGetLastFunction(instance));
	}
	
	public LLVMContext getContext() {
		return LLVMContext.getContext(Core.LLVMGetModuleContext(instance));
	}
	
	/**
	 * Finds an identified struct by name. Even though this is a method of an
	 * {@link LLVMModule}, identified structs are actually stored in the
	 * associated {@link LLVMContext}.
	 */
	public LLVMIdentifiedStructType getTypeByName(String name) {
		return LLVMIdentifiedStructType.getIdentifiedStructType(Core.LLVMGetTypeByName(instance,name));
	}
	
	public boolean writeBitcodeToFile(String path) {
		int result = BitWriter.LLVMWriteBitcodeToFile(instance,path);
		return (result == 0);
	}
	
	public boolean writeBitcodeToFD(int fd,boolean shouldClose,boolean unbuffered) {
		int result = BitWriter.LLVMWriteBitcodeToFD(instance,fd,shouldClose ? 1 : 0,unbuffered ? 1 : 0);
		return (result == 0);
	}
	
	public boolean writeBitcodeToFileHandle(int handle) {
		int result = BitWriter.LLVMWriteBitcodeToFileHandle(instance,handle);
		return (result == 0);
	}
	
	public String linkModule(LLVMModule Src,LLVMLinkerMode Mode) {
		SWIGTYPE_p_p_char str = Linker.new_StringArray(1);
		String res = null;
		if (0 != Linker.LLVMLinkModules(instance,Src.instance,Mode,str))
			res = Linker.StringArray_getitem(str,0);
		Linker.delete_StringArray(str);
		return res;
	}
	
	public LLVMModule(String name,LLVMContext context) {
		instance = new Hashable_LLVMOpaqueModule(Core.LLVMModuleCreateWithNameInContext(name,context.getInstance()));
		getContext().getModuleCache().put(instance,this);
	}
	
	/**
	 * According to the LLVM documentation for Verifier::verifyModule in
	 * Verifier.h, "This should only be used for debugging, because it plays
	 * games with PassManagers and stuff."
	 */
	public String verify(LLVMVerifierFailureAction action) {
		SWIGTYPE_p_p_char str = Analysis.new_StringArray(1);
		String res = null;
		if (0 != Analysis.LLVMVerifyModule(instance,action,str))
			res = Analysis.StringArray_getitem(str,0);
		Analysis.delete_StringArray(str);
		return res;
	}
	
	/**
	 * Get a cached {@link LLVMModule} object.
	 * 
	 * <p>
	 * Callers of this method appear to assume that all LLVM module objects with
	 * which jllvm might interact are created using the jllvm API so that they
	 * are cached by jllvm, and callers appear never to pass null or a wrapped
	 * null as {@code m}.  Thus, this method asserts that the return value is
	 * never null.
	 * </p>
	 * 
	 * @param m
	 *          contains the address of the LLVM object for which a cached
	 *          {@link LLVMModule} object should be sought
	 * @return the cached {@link LLVMModule} object
	 */
	public static LLVMModule getModule(SWIGTYPE_p_LLVMOpaqueModule m) {
		assert(m != null);
		final LLVMContext ctxt = LLVMContext.getContext(Core.LLVMGetModuleContext(m));
		final LLVMModule result = ctxt.getModuleCache().get(new Hashable_LLVMOpaqueModule(m));
		assert(result != null);
		return result;
	}
	
	/**
	 * It's fine to dispose of this before but not after its {@link LLVMContext}.
	 * If this has an {@link LLVMExecutionEngine}, do not dispose of this at all
	 * as that will do so.
	 */
	public void dispose() {
		Core.LLVMDisposeModule(instance);
	}
}
