package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcCharType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedIntType;
import static openacc.codegen.llvmBackend.SrcPrimitiveNonNumericType.SrcVoidType;

import org.jllvm.LLVMContext;
import org.jllvm.LLVMModule;

/**
 * The LLVM backend's table of built-in types.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class BuiltinTypeTable {
  /** Should only be accessed via {@link #getVaList}. */
  private SrcStructType vaList = null;
  /** Should only be accessed via {@link #getNVLHeapT}. */
  private SrcType nvlHeapT = null;

  /**
   * Look up a built-in type.
   * 
   * <p>
   * Do not call this method for a type that is not actually required by the C
   * source code or unnecessary warnings might be generated. It is safe (no
   * warnings or errors) to call this method to test if a symbol used in the
   * C source code is a built-in type.
   * </p>
   * 
   * @param name
   *          the name of the type
   * @param llvmModule
   *          the LLVM module into which any required declarations should be
   *          generated
   * @param warningsAsErrors
   *          whether to report warnings as errors
   * @return the type, or null if there is no type by that name
   */
  public SrcType lookup(String name, LLVMModule llvmModule,
                        boolean warningsAsErrors)
  {
    switch (name) {
    case "__builtin_va_list":
      return getVaList(llvmModule, warningsAsErrors);
    case "__builtin_nvl_heap":
      return getNVLHeapT(llvmModule, warningsAsErrors);
    default:
      return null;
    }
  }

  /**
   * Get the {@code va_list} type for the target platform.
   * 
   * <p>
   * This method generates a warning if it doesn't recognize the target
   * triple, but we don't want to warn about unsupported features if they're
   * not used, so only call this method when {@code va_list} is actually
   * used.
   * </p>
   * 
   * <p>
   * TODO: This needs to be extended for other platforms. To find out what
   * to do for any new platform, just invoke clang there and see what type
   * it generates for {@code va_list}. In case it helps, here's what clang's
   * API has:
   * </p>
   * 
   * <ul>
   * <li>
   * <a href=
   * "http://clang.llvm.org/doxygen/classclang_1_1TargetInfo.html#aa6330d7debefbbde9e0e724705596280"
   * >getBuiltinVaListKind</a></li>
   * <li>
   * <a href=
   * "http://clang.llvm.org/doxygen/include_2clang_2Basic_2TargetInfo_8h_source.html#l00133"
   * >BuiltinVaListKind</a></li>
   * </ul>
   * 
   * @param llvmModule
   *          the LLVM module into which any required declarations should be
   *          generated
   * @param warningsAsErrors
   *          whether to report warnings as errors
   * @return the type
   */
  public SrcStructType getVaList(LLVMModule llvmModule,
                                 boolean warningsAsErrors)
  {
    if (vaList != null)
      return vaList;

    final String tag = ".va_list";

    // http://llvm.org/docs/LangRef.html#target-triple
    final LLVMContext ctxt = llvmModule.getContext();
    final String[] targetTriple = llvmModule.getTargetTriple().split("-", 4);
                                final String arch;
    @SuppressWarnings("unused") final String vendor;
    @SuppressWarnings("unused") final String os;
    @SuppressWarnings("unused") final String env;
    {
      int i = 0;
      arch   = i < targetTriple.length ? targetTriple[i++] : "";
      vendor = i < targetTriple.length ? targetTriple[i++] : "";
      os     = i < targetTriple.length ? targetTriple[i++] : "";
      env    = i < targetTriple.length ? targetTriple[i++] : "";
    }

    vaList = new SrcStructType(tag, ctxt);
    if (arch.equals("x86_64"))
      // fig 3.34 of http://www.x86-64.org/documentation/abi.pdf
      vaList.setBody(
        new String[]{"gp_offset", "fp_offset", "overflow_arg_area",
                     "reg_save_area"},
        new SrcType[]{SrcUnsignedIntType, SrcUnsignedIntType,
                      SrcPointerType.get(SrcVoidType),
                      SrcPointerType.get(SrcVoidType)},
        null, ctxt);
    else {
      // http://llvm.org/docs/LangRef.html#int-varargs
      vaList.setBody(
        new String[]{"v"},
        new SrcType[]{SrcPointerType.get(SrcCharType)},
        null, ctxt);
      BuildLLVM.warn(warningsAsErrors,
                     "unknown target triple \""+llvmModule.getTargetTriple()
                     +"\": assuming default va_list type");
    }
    return vaList;
  }

  /**
   * Get the {@code nvl_heap_t} type.
   * 
   * @param llvmModule
   *          the LLVM module into which any required declarations should be
   *          generated
   * @param warningsAsErrors
   *          whether to report warnings as errors
   * @return the type
   */
  public SrcType getNVLHeapT(LLVMModule llvmModule,
                             boolean warningsAsErrors)
  {
    if (nvlHeapT != null)
      return nvlHeapT;
    return nvlHeapT = new SrcStructType("__builtin_nvl_heap",
                                        llvmModule.getContext());
  }
}
