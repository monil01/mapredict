package org.jllvm;

import org.jllvm.bindings.Support;

public class LLVMSupport {
	/**
	 * @param Filename the library to load
	 * @return true iff it failed to load
	 */
	public static boolean loadLibraryPermanently(String Filename) {
		return 0 != Support.LLVMLoadLibraryPermanently(Filename);
	}
	public static void unloadLibraries() {
		Support.LLVMUnloadLibraries();
	}
}
