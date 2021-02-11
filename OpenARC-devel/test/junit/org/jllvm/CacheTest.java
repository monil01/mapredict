package org.jllvm;

import openacc.test.JUnitTest;

import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Checks jllvm caching, whose bugs tend to be very racy and thus hard to
 * reveal reliably via OpenARC's LLVM backend. We can produce them more easily
 * by calling jllvm code directly.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class CacheTest extends JUnitTest {
  @BeforeClass public static void setup() {
    // Normally the BuildLLVM pass loads the jllvm native library. However,
    // these test methods use jllvm before running the BuildLLVM pass. In case
    // these test methods are run before any other test methods that run the
    // BuildLLVM pass, we must load the jllvm native library first.
    System.loadLibrary("jllvm");
  }

  /**
   * LLVM requires every thread to use a different LLVM context in order to be
   * thread-safe. At one time, jllvm had {@code finalize} methods that called
   * LLVM dispose functions that affected LLVM contexts, but the JVM calls
   * {@code finalize} methods in separate threads. This test frequently caught
   * some of the LLVM failures that resulted, such as:
   * 
   * <pre>
   * Assertion failed: (Ts.count(o) == 0 && "Object already in set!"), function addGarbage, file /Users/jdenny/installs/llvm/3.2/src/lib/VMCore/LeaksContext.h, line 50.
   * </pre>
   */
  @Test public void noConcurrency() {
    LLVMContext ctxt = new LLVMContext();
    for (int i = 0; i < 100; ++i) {
      final LLVMModule mod = new LLVMModule("mod", ctxt);
      for (int j = 0; j < 1000; ++j)
        mod.addGlobal(LLVMIntegerType.get(ctxt, 32), "var");
      // This call replaces the implicit finalize call that would have occurred
      // at some later time for mod.
      mod.dispose();
    }
  }
}