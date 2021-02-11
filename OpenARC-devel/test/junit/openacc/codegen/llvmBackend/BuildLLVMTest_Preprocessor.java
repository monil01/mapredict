package openacc.codegen.llvmBackend;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.io.File;

import org.jllvm.LLVMExecutionEngine;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMTargetData;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Checks that preprocessing behaves correctly.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class BuildLLVMTest_Preprocessor extends BuildLLVMTest {
  @BeforeClass public static void setup() {
    System.loadLibrary("jllvm");
  }
  @Test public void openaccMacro() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final String[] srcLines = new String[]{
      "int main() {",
      "  #ifdef _OPENACC",
      "    return 1;",
      "  #else",
      "    return 0;",
      "  #endif",
      "}",
    };
    final File file = writeTmpFile(".c", srcLines);
    final BuildLLVM buildLLVM = buildLLVM("", "", true, file);
    final LLVMModule mod = buildLLVM.getLLVMModules()[0];
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    checkIntFn(exec, mod, "main", 0);
    exec.dispose();
    String outdirOpt = "-outdir=" + tmpFolder.newFolder(".cetus_output");
    {
      final String[] cmdLine = new String[]{
        outdirOpt, file.getAbsolutePath()};
      final Driver driver = new Driver();
      driver.run(cmdLine);
      assertTrue("preprocessor must set _OPENACC",
                 Driver.getOptionValue("preprocessor").contains("_OPENACC"));
    }
    {
      final String[] cmdLine = new String[]{
        "-emitLLVM", outdirOpt, file.getAbsolutePath()};
      final Driver driver = new Driver();
      driver.run(cmdLine);
      assertFalse("preprocessor must not set _OPENACC",
                  Driver.getOptionValue("preprocessor").contains("_OPENACC"));
    }
  }
}
