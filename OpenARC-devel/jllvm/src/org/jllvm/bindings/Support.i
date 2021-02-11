%module Support
%{
#include <llvm-c/Support.h>
%}

%import "Core.i"
/*===-- llvm-c/Support.h - Support C Interface --------------------*- C -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file defines the C interface to the LLVM support library.             *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_SUPPORT_H
#define LLVM_C_SUPPORT_H

#include "llvm-c/Core.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * This function permanently loads the dynamic library at the given path.
 * It is safe to call this function multiple times for the same library.
 *
 * @see sys::DynamicLibrary::LoadLibraryPermanently()
  */
LLVMBool LLVMLoadLibraryPermanently(const char* Filename);

/**
 * This function is a hack to help support the OpenARC+LLVM test suite.
 * Specifically, we've found that Java will automatically unload dynamically
 * loaded libraries in a different thread than the thread that loaded the
 * libraries (whether loading was performed by LLVMLoadLibraryPermanently
 * above or by Java's System.loadLibrary). This behavior can cause trouble
 * with thread-local data (__thread) because initializers/constructors
 * (__attribute__(constructor)) might then operate on different data than
 * destructors/finalizers (__attribute__(destructor)). This function can be
 * used in the test suite to unload libraries explicitly in the same thread
 * so that Java will not automatically unload them in a different thread
 * later.
 */
void LLVMUnloadLibraries(void);

#ifdef __cplusplus
}
#endif

#endif
