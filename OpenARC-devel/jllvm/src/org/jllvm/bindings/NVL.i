%module NVL
%{
#include <stdbool.h>
#include <llvm-c/Transforms/NVL.h>
%}

%import "Core.i"
/*===-- NVL.h - NVL Transformation Library C Interface ----------*- C++ -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header declares the C interface to libLLVMNVL.a, which                *|
|* implements various transformations of the LLVM IR for the sake of NVL.     *|
|*                                                                            *|
|* Many exotic languages can interoperate with C code but have a harder time  *|
|* with C++ due to name mangling. So in addition to C, this interface enables *|
|* tools written in such languages.                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_TRANSFORMS_NVL_H
#define LLVM_C_TRANSFORMS_NVL_H

#include "llvm-c/Core.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup LLVMCTransformsNVL NVL transformations
 * @ingroup LLVMCNVL
 *
 * @{
 */

/** See llvm::createNVLAddTxsPass function. */
void LLVMAddNVLAddTxsPass(LLVMPassManagerRef PM);

/** See llvm::createNVLHoistTxAddPass function. */
void LLVMAddNVLHoistTxAddsPass(LLVMPassManagerRef PM, bool aggressive);

/** See llvm::createNVLAddSafetyPass function. */
void LLVMAddNVLAddSafetyPass(LLVMPassManagerRef PM);

/** See llvm::createNVLAddRefCountingPass function. */
void LLVMAddNVLAddRefCountingPass(LLVMPassManagerRef PM);

/** See llvm::createNVLAddPersistsPass function. */
void LLVMAddNVLAddPersistsPass(LLVMPassManagerRef PM);

/** See llvm::createNVLLowerPointersPass function. */
void LLVMAddNVLLowerPointersPass(LLVMPassManagerRef PM);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif /* defined(__cplusplus) */

#endif
