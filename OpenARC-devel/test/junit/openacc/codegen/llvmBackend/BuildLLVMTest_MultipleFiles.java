package openacc.codegen.llvmBackend;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.File;

import org.jllvm.LLVMExecutionEngine;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMTargetData;
import org.jllvm.bindings.LLVMLinkerMode;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Checks the ability to build correct LLVM IR for multiple C files.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class BuildLLVMTest_MultipleFiles extends BuildLLVMTest {
  @BeforeClass public static void setup() {
    System.loadLibrary("jllvm");
  }

  @Test public void linkingGlobals() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final File file1 = writeTmpFile("1.c",
      "int foo();",
      "extern int i;",
      "int main() {",
      "  return foo() + i;",
      "}");
    final File file2 = writeTmpFile("2.c",
      "int i = 6;",
      "int foo() { return 5; }");
    final BuildLLVM result = buildLLVM("", "", true, file1, file2);
    final LLVMModule mod = result.getLLVMModules()[0];
    final String linkError
      = mod.linkModule(result.getLLVMModules()[1],
                       LLVMLinkerMode.LLVMLinkerDestroySource);
    if (linkError != null)
      fail(linkError);
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    final int status = exec.runFunctionAsMain(mod.getNamedFunction("main"),
                                              new String[]{}, new String[]{});
    assertEquals("main exit code", 11, status);

    exec.dispose();
  }

  /**
   * Cetus used to corrupt relative includes, which broke at least some SPEC
   * CPU 2006 v1.0.1 benchmarks, but this is a quicker check that relative
   * includes work properly.
   */
  @Test public void relativeIncludes() throws Exception {
    LLVMTargetData.initializeNativeTarget();

    // Headers must be included relative to test.c not the current working
    // directory. However, Cetus used to pre-preprocess test.c, write the
    // new version to the working directory, and then runs the C
    // preprocessor, which thus interpreted includes relative to the working
    // directory. These days, when not using -emitLLVM, Cetus's
    // pre-preprocesser adjusts the relative includes to endure the change
    // of directories. When using -emitLLVM, Cetus skips its
    // pre-preprocessor altogether.

    final File dir = mkTmpDir("");
    final File dir_dir = mkDirInTmpDir(dir, dir.getAbsolutePath());
    final File dir_include = mkDirInTmpDir(dir, "include");
    final File dir_include_openarcrt = mkDirInTmpDir(dir_include,
                                                     "openarcrt");
    writeFileInTmpDir(dir, "stdbool.h", "#define true 2.25");
    final File dir_abs_h = writeFileInTmpDir(dir, "abs.h",
                                             "int abs() { return 4000; }");
    writeFileInTmpDir(dir_dir, "abs.h", "int abs() { return 9000; }");
    writeFileInTmpDir(dir, "rel.h", "int rel() { return 1; }");
    writeFileInTmpDir(dir_include_openarcrt, "openacc.h",
                      "enum { acc_device_none = 20 };");
    assertTrue("openarcrt/openacc.h must exist",
               new File("openarcrt", "openacc.h").exists());

    final File test_c = writeFileInTmpDir(dir, "test.c",
      // Goes directly to include path, but relative include exists. Must not
      // adjust or will find the latter one.
      "#include <stdbool.h>",

      // Absolute include, but absolute name also exists relative to the
      // original directory. Must not try to adjust or will find the latter
      // one (which must exist or else this test is just exercising the
      // existence check (exercised below) not the absolute path check).
      "#include \"" + dir_abs_h.getAbsolutePath() + "\"",

      // Include from later in include path because one does not exist relative
      // to the original directory. Must not adjust or will specify the
      // non-existent file.
      "#include \"stddef.h\"",

      // Same as last one except it also exists relative to the working
      // directory (as guaranteed by assertion above). Cetus's
      // pre-preprocessor still breaks this case because it processes the
      // include relative to the working directory. For -emitLLVM, we don't
      // use Cetus's pre-preprocessor, so the preprocessor walks the include
      // path for us.
      "#include \"openarcrt/openacc.h\"",

      // Relative include. Must adjust or either won't find or will find wrong
      // one if one exists relative to working directory or relative to a
      // directory later in the include path.
      "#include \"rel.h\"",

      "int main() {",
      "  return true*50000 + abs() + (300+(int)NULL)",
      "         + acc_device_none + rel();",
      "}");

    final BuildLLVM result = buildLLVM(
      "", "",
      new String[]{"-addIncludePath="+dir_include.getAbsolutePath()},
      true, test_c);
    final LLVMModule mod = result.getLLVMModules()[0];
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    final int status = exec.runFunctionAsMain(mod.getNamedFunction("main"),
                                              new String[]{}, new String[]{});
    assertEquals("main exit code", 54321, status);

    exec.dispose();
  }
}
