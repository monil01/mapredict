package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcCharType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcIntType;
import static openacc.codegen.llvmBackend.SrcPrimitiveNonNumericType.SrcVoidType;
import static org.jllvm.bindings.IPO.LLVMAddFunctionInliningPass;
import static org.jllvm.bindings.NVL.LLVMAddNVLAddRefCountingPass;
import static org.jllvm.bindings.NVL.LLVMAddNVLAddSafetyPass;
import static org.jllvm.bindings.NVL.LLVMAddNVLAddTxsPass;
import static org.jllvm.bindings.NVL.LLVMAddNVLHoistTxAddsPass;
import static org.jllvm.bindings.NVL.LLVMAddNVLLowerPointersPass;
import static org.jllvm.bindings.Scalar.LLVMAddAggressiveDCEPass;
import static org.jllvm.bindings.Scalar.LLVMAddLoopRotatePass;
import static org.jllvm.bindings.Scalar.LLVMAddPromoteMemoryToRegisterPass;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Random;

import openacc.exec.BuildConfig;

import org.hamcrest.CoreMatchers;
import org.jllvm.LLVMArrayType;
import org.jllvm.LLVMConstant;
import org.jllvm.LLVMContext;
import org.jllvm.LLVMExecutionEngine;
import org.jllvm.LLVMFunction;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMModulePassManager;
import org.jllvm.LLVMPointerType;
import org.jllvm.LLVMSupport;
import org.jllvm.LLVMTargetData;
import org.jllvm.LLVMVoidType;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Checks the ability to build correct LLVM IR for NVL.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class BuildLLVMTest_NVL extends BuildLLVMTest {
  private static String targetTriple = null;
  private static String targetDataLayout = null;
  private static String nvmTestdir = null;

  /**
   * If {@link #setup} sets {@link #havePmemobjLibs} to true, then all three
   * of the following are defined to various versions of the NVL runtime
   * libraries and their dependencies, and libraries from
   * {@link #nvlrtPmemobjTxsLibs} are loaded in this process.
   * 
   * Before attempting to link to any of these libraries, a test should call
   * {@link #assumePmemobjLibs}, which checks for these libraries and skips
   * the remainder of the test if they're not available. To try compiling
   * without linking first, a test can check {@link #havePmemobjLibs} but
   * should still call {@link #assumePmemobjLibs} after compilation to
   * report that linking was skipped.
   */
  private static boolean havePmemobjLibs  = false;
  private static String[] nvlrtPmemobjLibdirs = null;
  private static String[] nvlrtPmemobjNorefsLibs = null;
  private static String[] nvlrtPmemobjTxsLibs = null;

  /**
   * If {@link #setup} sets {@link #havePmemobjLibs} to true and determines
   * that NVL MPI functionality is also available, then tests can link to
   * MPI plus the NVL runtime libraries and dependencies in
   * {@link #nvlrtPmemobjTxsMpiLibs}. Otherwise,
   * {@link #nvlrtPmemobjTxsMpiLibs} remains null.
   * 
   * These NVL runtime libraries are never loaded into this process because
   * MPI applications are run in external processes launched with
   * {@code mpiexec}.
   * 
   * Before attempting to compile MPI code, tests should call
   * {@link #assumeConfig} to check for {@code mpi_includes}. Before
   * attempting to link to MPI libraries and these NVL runtime libraries,
   * tests should call {@link #assumeConfig} to check for {@code mpi_libdir}
   * and should call {@link #assumePmemobjLibs}. To attempt compiling
   * without linking, tests can check {@link BuildConfig} for
   * {@code mpi_libdir} and can check {@link #havePmemobjLibs} but afterward
   * still should call {@link #assumeConfig} for {@code mpi_libdir} and
   * should call {@link #assumePmemobjLibs}. Before attempting to call
   * {@code mpiexec}, tests should call {@link #assumeConfig} to check for
   * {@code mpi_exec}. For convenience, {@link #nvlOpenarcCC} and
   * {@link #nvlRunExe} encapsulate many of these steps.
   * 
   * Notice there's a discrepancy in the way availability of pmemobj and MPI
   * are handled. This discrepancy is because missing MPI headers prevent
   * application compilation while missing pmemobj headers prevent only
   * application linking because they prevent NVL runtime compilation.
   */
  private static String[] nvlrtPmemobjTxsMpiLibs = null;

  @BeforeClass public static void setup() {
    System.loadLibrary("jllvm");
    final BuildConfig buildConfig = BuildConfig.getBuildConfig();

    targetTriple = buildConfig.getProperty("llvmTargetTriple");
    targetDataLayout = buildConfig.getProperty("llvmTargetDataLayout");
    nvmTestdir = buildConfig.getProperty("nvm_testdir");

    final String nvlrtLibdir = new File(getTopDir(), "nvl/rt").toString();
    final String pmemLibdir = buildConfig.getProperty("pmem_libdir");
    if (!buildConfig.getProperty("pmem_includes").isEmpty()
        && !pmemLibdir.isEmpty())
    {
      nvlrtPmemobjLibdirs = new String[]{nvlrtLibdir, pmemLibdir};
      nvlrtPmemobjNorefsLibs = new String[]{
        pmemLibdir+"/libpmem.so",
        pmemLibdir+"/libpmemobj.so",
        nvlrtLibdir+"/libnvlrt-pmemobj-norefs.so"};
      nvlrtPmemobjTxsLibs = new String[]{
        pmemLibdir+"/libpmem.so",
        pmemLibdir+"/libpmemobj.so",
        nvlrtLibdir+"/libnvlrt-pmemobj-txs.so"};
      if (!buildConfig.getProperty("mpi_includes").isEmpty()
          && !buildConfig.getProperty("mpi_libdir").isEmpty())
        nvlrtPmemobjTxsMpiLibs = new String[]{
          pmemLibdir+"/libpmem.so",
          pmemLibdir+"/libpmemobj.so",
          nvlrtLibdir+"/libnvlrt-pmemobj-txs-mpi.so"};
      for (String lib : nvlrtPmemobjTxsLibs) {
        // We use this as a convenient way to call dlopen with RTLD_GLOBAL.
        // Alternatively, we could use LD_PRELOAD when calling java.
        // System.loadLibrary or System.load apparently specifies
        // RTLD_LOCAL, which does not make our library symbols available to
        // LLVM's execution engine.
        if (LLVMSupport.loadLibraryPermanently(lib))
          throw new UnsatisfiedLinkError("via LLVM, could not load " + lib);
      }
      havePmemobjLibs = true;
    }
  }
  @AfterClass public static void cleanup() {
    // The need for this is documented in llvm.git/include/llvm-c/Support.h.
    if (havePmemobjLibs)
      LLVMSupport.unloadLibraries();
  }
  private void assumePmemobjLibs() {
    // Report missing pmem_includes or pmem_libdir because pmemobj versions
    // of the NVL runtime won't be built if either is missing.
    assumeConfig("pmem_includes");
    assumeConfig("pmem_libdir");
  }

  private File mkHeapFileNoDelete(String name) throws IOException {
    File dir = nvmTestdir.isEmpty() ? mkTmpDir(".nvl-heaps", true)
                                    : new File(nvmTestdir);
    File file = new File(dir, name);
    return file;
  }
  private void cleanDir(File dir) throws IOException {
    assert(dir.isDirectory());
    // Play it safe given that all we need right now are shallow
    // directories.
    for (String file : dir.list()) {
      if (new File(dir, file).isDirectory())
        throw new IOException("attempt to recursively remove deep"
                              +" directory: "+dir);
    }
    for (String file : dir.list())
      new File(dir, file).delete();
  }
  // Removes the file so it can be recreated.
  private File mkHeapFile(String name) throws IOException {
    File file = mkHeapFileNoDelete(name);
    file.delete();
    return file;
  }
  // Creates/cleans the directory so files within can be recreated.
  private File mkHeapDir(String name) throws IOException {
    File dir = mkHeapFileNoDelete(name);
    if (dir.isDirectory())
      cleanDir(dir);
    else {
      dir.delete();
      dir.mkdir();
    }
    return dir;
  }

  private void addNVLPasses(LLVMModulePassManager pm,
                            boolean mem2regForRefCounting)
  {
    LLVMAddNVLAddTxsPass(pm.getInstance());
    LLVMAddFunctionInliningPass(pm.getInstance());
    // This -mem2reg is for the sake of -nvl-host-tx-adds, but it affects
    // -nvl-add-ref-counting. We could keep that -mem2reg and add a -reg2mem
    // afterward, but that has an unfortunate effect: each assignment to a
    // pointer variable might become a separate register (at -mem2reg) and
    // then a separate variable (at -reg2mem) that lives for the duration of
    // function instead of dying or being overwritten (and thus decrementing
    // a reference count) where it originally was overwritten. Some of our
    // tests that expect memory to be freed at a specific time would then
    // fail.
    if (mem2regForRefCounting)
      LLVMAddPromoteMemoryToRegisterPass(pm.getInstance());
    LLVMAddLoopRotatePass(pm.getInstance());
    LLVMAddNVLHoistTxAddsPass(pm.getInstance(), true);
    LLVMAddNVLAddSafetyPass(pm.getInstance());
    if (mem2regForRefCounting) {
      LLVMAddPromoteMemoryToRegisterPass(pm.getInstance());
      LLVMAddAggressiveDCEPass(pm.getInstance());
    }
    LLVMAddNVLAddRefCountingPass(pm.getInstance());
    LLVMAddNVLLowerPointersPass(pm.getInstance());
  }

  private void addNVLPasses(LLVMModulePassManager pm) {
    addNVLPasses(pm, true);
  }

  private void nvlOpenarcCCAndRun(
    String exeName, File workDir, String ccOpts,
    String[] srcLines, String[] stderrExpect, int exitValueExpect)
    throws IOException, InterruptedException
  {
    nvlOpenarcCCAndRun(exeName, workDir, ccOpts, true, srcLines,
                       stderrExpect, exitValueExpect);
  }
  private void nvlOpenarcCCAndRun(
    String exeName, File workDir, String ccOpts, boolean txsAndRefs,
    String[] srcLines, String[] stderrExpect, int exitValueExpect)
    throws IOException, InterruptedException
  {
    nvlOpenarcCCAndRun(exeName, workDir, ccOpts, txsAndRefs, -1, srcLines,
                       stderrExpect, false, exitValueExpect);
  }
  private void nvlOpenarcCCAndRun(
    String exeName, File workDir, String ccOpts, boolean txsAndRefs,
    int mpiRanks, String[] srcLines, String[] stderrExpect,
    boolean atLeastOneLine, int exitValueExpect)
    throws IOException, InterruptedException
  {
    final File exe = nvlOpenarcCC(exeName, workDir, ccOpts, txsAndRefs,
                                  mpiRanks != -1, srcLines);
    nvlRunExe(exe, workDir, "", -1, mpiRanks, stderrExpect, atLeastOneLine,
              exitValueExpect);
  }
  private File nvlOpenarcCC(String outFileName, File workDir, String ccOpts,
                            String[] srcLines)
    throws IOException, InterruptedException
  {
    return nvlOpenarcCC(outFileName, workDir, ccOpts, true, false,
                        srcLines);
  }
  private File nvlOpenarcCC(String outFileName, File workDir, String ccOpts,
                            boolean txsAndRefs, boolean mpi,
                            String[] srcLines)
    throws IOException, InterruptedException
  {
    final File srcFile = writeFileInTmpDir(workDir, outFileName+".c",
                                           srcLines);
    return nvlOpenarcCC(outFileName, workDir, ccOpts, txsAndRefs, mpi,
                        srcFile);
  }
  private File nvlOpenarcCC(String outFileName, File workDir, String ccOpts,
                            boolean txsAndRefs, boolean mpi, File srcFile)
    throws IOException, InterruptedException
  {
    String ccArgs = ccOpts;
    if (!ccArgs.isEmpty())
      ccArgs += " ";
    if (txsAndRefs)
      ccArgs += "-fnvl-add-txs ";
    else
      ccArgs += "-fno-nvl-add-ref-counting ";
    ccArgs += shesc(srcFile.getAbsolutePath());

    // If MPI is requested but we don't have headers, then we cannot
    // compile, so skip the rest of the test. Don't try to use
    // nvlrtPmemobjTxsMpiLibs, which is null in this case.
    if (mpi)
      ccArgs += " -I"+shesc(assumeConfig("mpi_includes"));

    // We have required headers. If we don't have required libraries, we can
    // compile but not link.
    boolean compileOnly = false;
    if (!havePmemobjLibs)
      compileOnly = true;
    else {
      for (String lib : (txsAndRefs ? (mpi ? nvlrtPmemobjTxsMpiLibs
                                           : nvlrtPmemobjTxsLibs)
                                    : nvlrtPmemobjNorefsLibs))
        ccArgs += " "+shesc(lib);
      ccArgs += " -lm";
    }
    if (mpi) {
      final String mpiLibdir = BuildConfig.getBuildConfig()
                               .getProperty("mpi_libdir");
      if (mpiLibdir.isEmpty())
        compileOnly = true;
      else
        ccArgs += " -L"+shesc(mpiLibdir)+" -lmpi";
    }
    if (compileOnly)
      ccArgs += " -c";
    final File exe = openarcCC(outFileName, workDir, ccArgs);

    // If we skipped linking, mark the test as partially skipped.
    assumePmemobjLibs();
    if (mpi)
      assumeConfig("mpi_libdir");
    return exe;
  }
  private void nvlRunExe(
    File exe, File workDir, String exeArgs, long killTimeNano,
    final int mpiRanks, final String[] stderrExpect,
    final boolean atLeastOneLine, final int exitValueExpect)
    throws IOException, InterruptedException
  {
    nvlRunExe(exe, workDir, exeArgs, new RunChecker(){
      @Override
      public void checkAfter(String exeName, InputStream outStream,
                             InputStream errStream, int exitValue)
        throws IOException
      {
        assertEquals(exeName+": exit value", exitValueExpect, exitValue);
        if (stderrExpect != null) {
          final BufferedReader err
            = new BufferedReader(new InputStreamReader(errStream));
          String line;
          List<String> stderr = new LinkedList<String>();
          while (null != (line = err.readLine()))
            stderr.add(line);
          checkOut("stderr", Arrays.asList(stderrExpect), stderr, false,
                   mpiRanks>1, atLeastOneLine);
        }
      }
    }, killTimeNano, mpiRanks);
  }
  private void nvlRunExe(File exe, File workDir, String exeArgs,
                         RunChecker runChecker, long killTimeNano,
                         int mpiRanks)
    throws IOException, InterruptedException
  {
    assumePmemobjLibs();
    StringBuilder exePre = new StringBuilder("LD_LIBRARY_PATH=");
    {
      boolean firstDir = true;
      for (String dir : nvlrtPmemobjLibdirs) {
        if (firstDir)  firstDir = false;
        else           exePre.append(":");
        exePre.append(shesc(dir));
      }
    }
    if (mpiRanks >= 0) {
      exePre.append(":");
      exePre.append(shesc(assumeConfig("mpi_libdir")));
      exePre.append(" ");
      exePre.append(shesc(assumeConfig("mpi_exec")));
      exePre.append(" -n ");
      exePre.append(mpiRanks);
    }
    runExe(exe, workDir, exePre.toString(), exeArgs, runChecker,
           killTimeNano);
  }
  private void checkOut(
    String what, List<String> expected, List<String> actual,
    boolean permitDups, boolean permitOutOfOrder, boolean atLeastOneLine)
  {
    assert(!permitDups || !permitOutOfOrder);
    assert(permitOutOfOrder || !atLeastOneLine);
    if (!permitOutOfOrder) {
      final ListIterator<String> expItr = expected.listIterator(),
                                 actItr = actual.listIterator();
      int i;
      String prev = null;
      for (i = 0; actItr.hasNext(); ++i) {
        String exp = expItr.hasNext() ? expItr.next() : null;
        final String act = actItr.next();
        if (permitDups && (exp == null || !act.matches(exp)) && prev != null
            && act.equals(prev))
        {
          if (exp != null)
            expItr.previous();
          expItr.previous();
          exp = expItr.next();
        }
        else if (exp == null) {
          actItr.previous();
          break;
        }
        assertTrue(what+" line "+(i+1)+" expected:<"+exp+"> but was:<"
                   +act+">",
                   act.matches(exp));
        prev = act;
      }
      assertEquals(what+" has "+i+" lines", !expItr.hasNext(),
                   !actItr.hasNext());
      return;
    }
    final StringBuilder err = new StringBuilder();
    final List<String> expectedDup = new LinkedList<>(expected);
    int i = 0;
  LOOP_ACTUAL:
    for (final ListIterator<String> actItr = actual.listIterator();
         actItr.hasNext(); ++i)
    {
      final String act = actItr.next();
      for (final ListIterator<String> expItr = expectedDup.listIterator();
           expItr.hasNext();)
      {
        if (act.matches(expItr.next())) {
          expItr.remove();
          continue LOOP_ACTUAL;
        }
      }
      err.append(what+" line "+(i+1)+" was:<"+act+"> but ");
      if (expectedDup.isEmpty())
        err.append("there are no remaining expected line patterns");
      else
        err.append("does not match any remaining expected line pattern:");
      break;
    }
    if (err.length() == 0 && !expectedDup.isEmpty()) {
      if (!atLeastOneLine)
        err.append("expected line pattern(s) not matched in "+what+":");
      else if (expected.size() == expectedDup.size())
        err.append("no expected line pattern was matched in "+what+":");
    }
    if (err.length() > 0) {
      for (final ListIterator<String> expItr = expectedDup.listIterator();
           expItr.hasNext();)
      {
        err.append("\n<"+expItr.next()+">");
        if (expItr.hasNext())
          err.append(",");
      }
      assertTrue(err.toString(), false);
    }
  }

  private String[] concatArrays(String[]... arrs) {
    int len = 0;
    for (String[] arr : arrs)
      len += arr.length;
    String[] res = new String[len];
    len = 0;
    for (String[] arr : arrs) {
      for (String str : arr)
        res[len++] = str;
    }
    return res;
  }

  private void txsCheck(String testName, int nops, String exeOpts,
                        LinkedList<String> expout,
                        boolean permitStdoutDups, File heapFile,
                        String... testSrcLines)
    throws Exception
  {
    final File workDir = mkTmpDir("");
    final File exe = nvlOpenarcCC(testName, workDir, "", testSrcLines);
    txsCheck(testName, nops, exeOpts, expout, permitStdoutDups,
             heapFile, workDir, exe, -1);
  }
  private void txsCheck(String testName, int nops, String exeOpts,
                        LinkedList<String> expout,
                        boolean permitStdoutDups, File heapFileOrDir,
                        File workDir, File exe, int mpiRanks)
    throws Exception
  {
    // Time and check a run without killing it.
    System.err.println("running without kill...");
    final long startTime = System.nanoTime();
    LinkedList<String> stdout = txsRun(exe, workDir, exeOpts, -1, mpiRanks,
                                       false);
    final long duration = System.nanoTime() - startTime;
    txsCheckStdout(expout, stdout, false, mpiRanks>1);
    System.err.println("successful run duration: " + (duration*1e-9) + "s");

    // Start over.
    System.err.println("removing heap(s) and restarting...");
    if (heapFileOrDir.isDirectory())
      cleanDir(heapFileOrDir);
    else
      heapFileOrDir.delete();

    // Run many times, killing at a random time each run, and recovering
    // and resuming in the next run.
    final Random rand = new Random();
    final long avgOpDuration = duration/nops;
    int nkills = 2*nops;
    boolean heapReady = false;
    boolean rootReady = false;
    long killTime = 0;
    for (int kill = 0;
         kill < nkills || (killTime < duration && !rootReady);
         ++kill)
    {
      // Kill it at some random time. The lower time bound is zero so that
      // we might kill during recovery. The upper time bound is eventually
      // much longer than the average duration of an operation so that
      // previously killed operations might be given time to complete.
      // However, if we always set the upper bound high enough for that,
      // then we'll rarely kill early, so we increase the upper bound with
      // each run.
      killTime
        // final upper     grow upper per run  randomize btwn 0 and upper
        = 4*avgOpDuration * (kill+1)/nkills    * rand.nextInt(10000)/10000;
      if (mpiRanks > 0) {
        final int killRank = rand.nextInt(mpiRanks);
        System.err.println("running with rank "+killRank+" kill time "
                           + (killTime*1e-9) + "s...");
        // All MPI processes normally die automatically when one dies, but
        // make sure by specifying a long killTime for mpiexec.
        stdout = txsRun(exe, workDir,
                        exeOpts+" -k"+killRank+"="+killTime,
                        killTime+duration, mpiRanks, true);
      }
      else {
        System.err.println("running with kill time "
                           + (killTime*1e-9) + "s...");
        stdout = txsRun(exe, workDir, exeOpts, killTime, mpiRanks, true);
      }
      final int heaps = mpiRanks >= 0 ? mpiRanks : 1;
      int heapReadies = 0;
      int rootReadies = 0;
      for (ListIterator<String> stdoutItr = stdout.listIterator();
           stdoutItr.hasNext();)
      {
        final String line = stdoutItr.next();
        if (line.endsWith("heapReady"))
          ++heapReadies;
        else if (line.endsWith("rootReady"))
          ++rootReadies;
      }
      // Heap creation can take a really long time relative to application
      // operations like linked list add and remove (this ratio is
      // especially high when we're targetting our ioScale hardware because
      // linked list add and remove are so fast; not so with HDD). Once
      // we've finally completed heap creation, reset the kill times so we
      // can carefully check killing application operations.
      if (!heapReady && heapReadies == heaps) {
        heapReady = true;
        kill = 0;
      }
      // Root creation is similar.
      if (!rootReady && rootReadies == heaps) {
        rootReady = true;
        kill = 0;
      }
    }

    // Finally, run it once more with a long kill time it to see if it can
    // still recover correctly and print the correct final result. If not,
    // the kill time prevents an infinite loop.
    final long longKillTime = duration*2;
    System.err.println("running with long kill time "
                       + (longKillTime*1e-9) + "s...");
    stdout = txsRun(exe, workDir, exeOpts, longKillTime, mpiRanks, false);
    txsCheckStdout(expout, stdout, permitStdoutDups, mpiRanks>1);
  }
  private LinkedList<String> txsRun(
    File exe, File workDir, String exeOpts, long killTimeNano, int mpiRanks,
    final boolean tolerateFailure)
    throws IOException, InterruptedException
  {
    final LinkedList<String> stdout = new LinkedList<>();
    nvlRunExe(exe, workDir, exeOpts, new RunChecker(){
      @Override
      public void checkAfter(String exeName, InputStream outStream,
                             InputStream errStream, int exitValue)
        throws IOException
      {
        if (!tolerateFailure)
          assertEquals(exeName+": exit value", 0, exitValue);
        final BufferedReader out
          = new BufferedReader(new InputStreamReader(outStream));
        String line;
        try {
          while (null != (line = out.readLine()))
            stdout.add(line);
        }
        catch (IOException e) {
          if (!tolerateFailure)
            throw e;
        }
      }
    },
    killTimeNano, mpiRanks);
    return stdout;
  }
  private void txsCheckStdout(List<String> expout, List<String> stdout,
                              boolean permitDups, boolean permitOutOfOrder)
  {
    checkOut("stdout", expout, stdout, permitDups, permitOutOfOrder, false);
  }

  // -----------------------------------------------------------------------
  // constraint: a type shall not be both nvl-qualified and nvl_wp-qualified

  @Test public void nvlAndNVLwpPtr() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "type has both nvl and nvl_wp type qualifiers"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "int nvl * nvl nvl_wp *p = 0;",
      });
  }

  @Test public void nvlWPAndNVLPtr() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "type has both nvl and nvl_wp type qualifiers"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "int nvl * nvl_wp nvl *p = 0;",
      });
  }

  @Test public void nvlTypedefNVLwpPtr() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "type has both nvl and nvl_wp type qualifiers"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "typedef int nvl * nvl_wp T;",
        "nvl T *p = 0;",
      });
  }

  @Test public void nvlWPTypedefNVLPtr() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "type has both nvl and nvl_wp type qualifiers"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "typedef int nvl * nvl T;",
        "T nvl_wp *p = 0;",
      });
  }

  @Test public void nvlArrayNVLwpPtr() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "type has both nvl and nvl_wp type qualifiers"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "typedef int nvl * nvl_wp T[5];",
        "nvl T *p = 0;",
      });
  }

  @Test public void nvlWPArrayNVLPtr() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "type has both nvl and nvl_wp type qualifiers"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "typedef int nvl * nvl T[];",
        "T nvl_wp *p = 0;",
      });
  }

  // -----------------------------------------------------------
  // constraint: non-pointer types shall not be nvl_wp-qualified

  @Test public void ptrToNVLwpInt() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "non-pointer type has nvl_wp type qualifier"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "int nvl_wp *p = 0;",
      });
  }

  @Test public void nvlWPIntMember() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "non-pointer type has nvl_wp type qualifier"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S {",
        "  nvl_wp int i;",
        "};",
      });
  }

  @Test public void nvlWPArrayOfInt() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "non-pointer type has nvl_wp type qualifier"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "typedef int a5i[5];",
        "a5i nvl_wp *p = 0;",
      });
  }

  @Test public void nvlWPArrayOfPtr() throws IOException {
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "typedef int nvl *a5pi[5];",
        "a5pi nvl_wp *p = 0;",
      });
  }

  // -------------------------------------------------------------------
  // constraint: target of NVM-stored ptr shall be explicitly NVM-stored

  @Test public void nvlPtrToNVM() throws IOException {
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { int nvl * nvl p; };",
        "int nvl * nvl * ppi = 0;",
        "typedef int a5i[5];",
        "a5i nvl * nvl * ppa = 0;",
      });
  }

  @Test public void nvlWPptrToNVM() throws IOException {
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { int nvl * nvl_wp p; };",
        "int nvl * nvl_wp * ppi = 0;",
        "typedef int a5i[5];",
        "a5i nvl * nvl_wp * ppa = 0;",
      });
  }

  @Test public void nvlPtrToNonNVM1() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { int * nvl p; };",
      });
  }

  @Test public void nvlWPptrToNonNVM1() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { int * nvl_wp p; };",
      });
  }

  @Test public void nvlPtrToNonNVM2() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "int * nvl * pp = 0;",
      });
  }

  @Test public void nvlWPptrToNonNVM2() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "int * nvl_wp * pp = 0;",
      });
  }

  @Test public void nvmPtrToNVMArray() throws IOException {
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S1 { int nvl (* nvl p)[3]; };",
        "struct S2 { int nvl (* nvl_wp p)[3][5]; };",
      });
  }

  @Test public void nvlPtrToNonNVMArray() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { int (* nvl p)[3]; };",
      });
  }

  @Test public void nvlWPptrToNonNVMArray() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { int (* nvl_wp p)[3]; };",
      });
  }

  @Test public void implicitNVMPtrToNVM() throws IOException {
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S nvl *ps1 = 0;",
        "struct S { int nvl *pi; int nvl * nvl_wp *ppi; };",
        "struct S nvl *ps2 = 0;",
      });
  }

  @Test public void implicitNVMPtrToNonNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { int *pi; };",
        "struct S nvl *ps = 0;",
      });
  }

  @Test public void implicitNVMPtrToNonNVMArray() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { int (*pa)[5]; };",
        "struct S nvl *ps = 0;",
      });
  }

  @Test public void implicitNVMArrayOfPtrToNVM() throws IOException {
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S nvl *ps1 = 0;",
        "struct S { int nvl *ap[5]; };",
        "struct S nvl *ps2 = 0;",
      });
  }

  @Test public void implicitNVMArrayOfPtrToNonNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { int *ap[5]; };",
        "struct S nvl *ps = 0;",
      });
  }

  @Test public void incompleteStructWithImplicitNVMPtrToNonNVM()
    throws IOException
  {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S nvl *p = 0;",
        "struct S { int *p; };",
      });
  }

  // -----------------------------------------------------------------------
  // Repeating one of the tests above, make sure nvl or nvl_wp isn't dropped
  // when combined with various combinations of C type qualifiers on a
  // pointer. They used to be.

  @Test public void nvlConstPtr() throws Exception {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { int * const nvl p; };",
      });
  }

  @Test public void nvlRestrictPtr() throws Exception {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { int * nvl restrict p; };",
      });
  }

  @Test public void nvlVolatilePtr() throws Exception {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { int * volatile nvl p; };",
      });
  }

  @Test public void nvlConstRestrictPtr() throws Exception {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { int * const restrict nvl p; };",
      });
  }

  @Test public void nvlConstVolatilePtr() throws Exception {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { int * const nvl volatile p; };",
      });
  }

  @Test public void nvlRestrictVolatilePtr() throws Exception {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { int * nvl restrict volatile p; };",
      });
  }

  @Test public void nvlConstRestrictVolatilePtr() throws Exception {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { int * const nvl restrict volatile p; };",
      });
  }

  @Test public void nvlWPConstPtr() throws Exception {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { int * const nvl_wp p; };",
      });
  }

  @Test public void nvlWPRestrictPtr() throws Exception {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { int * nvl_wp restrict p; };",
      });
  }

  @Test public void nvlWPVolatilePtr() throws Exception {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { int * volatile nvl_wp p; };",
      });
  }

  @Test public void nvlWPConstRestrictPtr() throws Exception {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { int * const restrict nvl_wp p; };",
      });
  }

  @Test public void nvlWPConstVolatilePtr() throws Exception {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { int * const nvl_wp volatile p; };",
      });
  }

  @Test public void nvlWPRestrictVolatilePtr() throws Exception {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { int * nvl_wp restrict volatile p; };",
      });
  }

  @Test public void nvlWPConstRestrictVolatilePtr() throws Exception {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { int * const nvl_wp restrict volatile p; };",
      });
  }

  // -------------------------------------------------------------
  // constraint: allocation of ptr to NVM shall have explicit init

  @Test public void uninitLocalPtrToNVL() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "pointer to NVM allocated without explicit initializer"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void foo() {",
        "  int nvl *p;",
        "}",
      });
  }

  @Test public void uninitLocalPtrToNVLwp() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "pointer to NVM allocated without explicit initializer"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void foo() {",
        "  int nvl * nvl_wp *p;",
        "}",
      });
  }

  @Test public void uninitLocalPtrToNVLArray() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "pointer to NVM allocated without explicit initializer"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void foo() {",
        "  int nvl (*p)[3];",
        "}",
      });
  }

  @Test public void uninitLocalPtrToNVLwpArray() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "pointer to NVM allocated without explicit initializer"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void foo() {",
        "  int nvl * nvl_wp (*p)[3];",
        "}",
      });
  }

  @Test public void uninitStaticLocalPtrToNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "pointer to NVM allocated without explicit initializer"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void foo() {",
        "  static int nvl *p;",
        "}",
      });
  }

  @Test public void uninitGlobalPtrToNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "pointer to NVM allocated without explicit initializer"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "int nvl * nvl_wp *p;",
        "int nvl * nvl_wp *p;",
      });
  }

  @Test public void initGlobalPtrToNVMAfterTentativeDef() throws IOException {
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "int nvl *p1;",
        "int nvl *p1 = 0;",
        "int nvl * nvl_wp *p2;",
        "int nvl * nvl_wp *p2 = 0;",
      });
  }

  @Test public void uninitLocalStructWithPtrToNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "pointer to NVM allocated without explicit initializer"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { int i; int nvl *p; };",
        "void foo() {",
        "  struct S s;",
        "}",
      });
  }

  @Test public void globalStructWithUninitPtrToNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "pointer to NVM allocated without explicit initializer"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { int i; int j; int nvl * nvl_wp *p; };",
        "struct S s = {0};",
      });
  }

  @Test public void localNestedStructWithUninitPtrToNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "pointer to NVM allocated without explicit initializer"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { int i; int j; int nvl *p; };",
        "void foo() {",
        "  struct {int i; struct S s;} s = {0, {0, 0}};",
        "}",
      });
  }

  @Test public void uninitGlobalArrayWithPtrToNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "pointer to NVM allocated without explicit initializer"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "int nvl * nvl_wp *a[5];",
      });
  }

  @Test public void localArrayWithUninitPtrToNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "pointer to NVM allocated without explicit initializer"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void foo() {",
        "  int nvl *a[3] = {0, 0};",
        "}",
      });
  }

  @Test public void localArrayWithInitPtrToNVM() throws IOException {
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void foo() {",
        "  int nvl *a1[] = {0};",
        "  int nvl * nvl_wp *a2[] = {0, 0};",
        "  int nvl *a3[3] = {0, 0, 0};",
        "}",
      });
  }

  @Test public void localExternPtrToNVM() throws IOException {
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "int nvl (*p1)[] = 0;",
        "int nvl * nvl_wp (*p2)[] = 0;",
        "void foo() {",
        "  extern int nvl (*p1)[3];",
        "  extern int nvl * nvl_wp (*p2)[3];",
        "}",
      });
  }

  // ----------------------------------------------------
  // constraint: jump shall not bypass init of ptr to NVM

  @Test public void gotoDoesNotBypassPtrToNVMInit() throws IOException {
    final List<String> src = new ArrayList<>();
    src.add("#include <nvl.h>");
    final String[] types = new String[]{"int nvl *", "int nvl * nvl_wp *"};
    for (int i = 0; i < types.length; ++i) {
      final String type = types[i];
      src.add(type+"pgb"+i+" = 0;"); // goto does not skip preceding global
      src.add("void foo() {");
      src.add("  "+type+"p1 = 0;"); // goto does not skip preceding outer scope local
      src.add("  {");
      src.add("    "+type+"p2 = 0;"); // goto leaving scope destroys preceding local
      src.add("    goto lab;");
      src.add("    "+type+"p3 = 0;"); // goto leaving scope skips following local
      src.add("  }");
      src.add("  {");
      src.add("    "+type+"p4 = 0;"); // goto skips separately scoped compound statement
      src.add("  }");
      src.add("  do {");
      src.add("    "+type+"p5 = 0;"); // goto skips separately scoped do loop
      src.add("  } while (0);");
      src.add("  while (1) {");
      src.add("    "+type+"p6 = 0;"); // goto skips separately scoped while loop
      src.add("  }");
      // goto skips separately scoped for loop init and body
      src.add("  for ("+type+"p7 = 0; p7 == 0; ++p7) {");
      src.add("    "+type+"p8 = 0;");
      src.add("  }");
      src.add("  static "+type+"p9 = 0;"); // static local init cannot be bypassed
      src.add("  lab:");
      src.add("  "+type+"p10 = 0;"); // goto does not skip following outer scope local
      src.add("}");
      src.add(""+type+"pga"+i+" = 0;"); // goto does not skip following global
    }
    buildLLVMSimple("", "", src.toArray(new String[0]));
  }

  @Test public void gotoBypassesExternPtrToNVMInit() throws IOException {
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "int nvl (*p1)[] = 0;",
        "int nvl * nvl_wp (*p2)[] = 0;",
        "void foo() {",
        "  goto lab;",
        // composite type of p[12] changes here, so local p[12] must be added,
        // but it's not initialized locally
        "  extern int nvl (*p1)[3];",
        "  extern int nvl * nvl_wp (*p2)[3];",
        "  lab: ;",
        "}",
      });
  }

  @Test public void gotoBypassesPtrToNVLInit() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "goto bypasses initialization of pointer to NVM: p"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void foo() {",
        "  goto lab;",
        "  int nvl *p = 0;",
        "  lab: ;",
        "}",
      });
  }

  @Test public void gotoBypassesPtrToNVLwpInit() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "goto bypasses initialization of pointer to NVM: p"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void foo() {",
        "  goto lab;",
        "  int nvl * nvl_wp *p = 0;",
        "  lab: ;",
        "}",
      });
  }

  @Test public void gotoDestroysAndBypassesPtrToNVLInit() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "goto bypasses initialization of pointer to NVM: p2"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void foo() {",
        "  int nvl *p1 = 0;",
        "  {",
        "    int nvl *p2 = 0;",
        "    lab: ;",
        "  }",
        "  {",
        "    int nvl *p3 = 0;",
        "    goto lab;",
        "  }",
        "}",
      });
  }

  @Test public void gotoDestroysAndBypassesPtrToNVLwpInit() throws IOException
  {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "goto bypasses initialization of pointer to NVM: p2"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void foo() {",
        "  int nvl * nvl_wp *p1 = 0;",
        "  {",
        "    int nvl * nvl_wp *p2 = 0;",
        "    lab: ;",
        "  }",
        "  {",
        "    int nvl * nvl_wp *p3 = 0;",
        "    goto lab;",
        "  }",
        "}",
      });
  }

  @Test public void caseBypassesPtrToNVMInit() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "switch case bypasses initialization of pointer to NVM: p"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void foo(int i) {",
        "  switch (i) {",
        "    int nvl *p = 0;",
        "  case 0: ;",
        "  }",
        "}",
      });
  }

  @Test public void defaultBypassesPtrToNVMInit() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "switch default bypasses initialization of pointer to NVM: p"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void foo(int i) {",
        "  switch (i) {",
        "    int nvl * nvl_wp *p = 0;",
        "  default: ;",
        "  }",
        "}",
      });
  }

  @Test public void nestedDefaultDoesNotBypassPtrToNVMInit()
    throws IOException
  {
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void foo(int i, int j, int k) {",
        "  switch (i) {",
        "    switch (j) {",
        "    default: ;",
        "      int nvl *p1 = 0;",
        "      int nvl * nvl_wp *p1wp = 0;",
        "    }",
        "  default: ;",
        "    int nvl *p2 = 0;",
        "    int nvl * nvl_wp *p2wp = 0;",
        "    switch (k) {",
        "    default: ;",
        "      int nvl *p3 = 0;",
        "      int nvl * nvl_wp *p3wp = 0;",
        "    }",
        "  }",
        "}",
      });
  }

  @Test public void nestedDefaultBypassesPtrToNVMInit() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "switch default bypasses initialization of pointer to NVM: p3"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void foo(int i, int j, int k) {",
        "  switch (i) {",
        "    switch (j) {",
        "    default: ;",
        "      int nvl *p1 = 0;",
        "    }",
        "  default: ;",
        "    int nvl *p2 = 0;",
        "    switch (k) {",
        "      int nvl *p3 = 0;",
        "    default: ;",
        "    }",
        "  }",
        "}",
      });
  }

  // -----------------------------------------------------------------------
  // constraint: objects with static or automatic storage duration shall not
  // be NVM-stored

  @Test public void nvlGlobal() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "variable \"i\" is NVM-stored"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "int nvl i;",
      });
  }

  @Test public void nvlWPglobal() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "variable \"i\" is NVM-stored"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "int nvl * nvl_wp i = 0;",
      });
  }

  @Test public void nvlGlobalNonTentative() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "variable \"x\" is NVM-stored"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "int nvl x = 0;",
      });
  }

  @Test public void nvlWPglobalNonTentative() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "variable \"x\" is NVM-stored"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "int nvl * nvl_wp x = 0;",
      });
  }

  @Test public void nvmLocal() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "variable \"y\" is NVM-stored"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void foo() {",
        "  int nvl y;",
        "}",
      });
  }

  @Test public void nvmArrayLocal() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "variable \"a\" is NVM-stored"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  int nvl * nvl_wp a[3];",
        "}",
      });
  }

  @Test public void nvmReturnType() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "function return type is NVM-stored"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "int nvl fn();",
      });
  }

  @Test public void nvmParamTypeOnFn() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "function parameter 1 type is NVM-stored"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "int fn(int nvl * nvl_wp);",
      });
  }

  @Test public void nvmParamTypeOnFnDef() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "function parameter 1 type is NVM-stored"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "int fn(int nvl i, int nvl j) {}",
      });
  }

  @Test public void nvmParamTypeOnFnTypedef() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "function parameter 2 type is NVM-stored"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "typedef int T(int x, int nvl * nvl_wp y, ...);",
      });
  }

  @Test public void nvmParamTypeOnFnPtr() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "function parameter 3 type is NVM-stored"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "int (*p)(int x, int y, nvl int z);",
      });
  }

  // --------------------------------------------------
  // constraint: function types shall not be NVM-stored

  // type qualifiers on function types are generally not permitted in C
  @Test public void nvlFunction() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "type qualifiers specified on function type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "typedef int T();",
        "T nvl fn;",
      });
  }

  @Test public void nvlWPfunction() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "type qualifiers specified on function type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "typedef int T();",
        "T nvl_wp fn;",
      });
  }

  // ----------------------------------------------
  // constraint: void types shall not be NVM-stored

  @Test public void ptrNVLVoid() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "void type is NVM-stored"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void nvl *p;",
      });
  }

  @Test public void ptrNVLwpVoid() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "non-pointer type has nvl_wp type qualifier"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void nvl_wp *p;",
      });
  }

  @Test public void ptrPtrNVMVoid() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "void type is NVM-stored"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void nvl **p;",
      });
  }

  // -----------------------------------------------------------------------
  // rule: every member of an NVM-stored object of struct type is implicitly
  // nvl-qualified if that member is not explicitly NVM-stored by the
  // struct type definition

  @Test public void addressOfImplicitNVLMember() throws Exception {
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct T { int i; int nvl * nvl_wp wppi; };",
        "void fn() {",
        "  struct T nvl *pt = 0;",
        // This used to fail a ValueAndType assertion because the struct's nvl
        // qualifier wasn't added to the recorded source type at the original
        // ValueAndType construction for the member access, so the LLVM type
        // and the recorded source type didn't match.
        "  int nvl *pi = &pt->i;",
        // In this case, make sure the backend doesn't combine the nvl and
        // nvl_wp qualifiers and thus fail an NVL type constraint. Also, at the
        // LLVM IR level, the member has the nvl_wp addrspace not the nvl
        // addrspace, so a bitcast after the getelementptr is required.
        "  int nvl * nvl_wp *pwppi = &pt->wppi;",
        "}",
      });
  }

  // -----------------------------------------------------------------------
  // constraint: if the type of any member of a struct type is NVM-stored,
  // then every use of that struct type shall be explicitly NVM-stored

  @Test public void nonNVMStructWithNVLmember() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "non-NVM-stored struct type has NVM-stored member"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S {",
        "  int nvl i;",
        "} s;",
      });
  }

  @Test public void nonNVMStructWithNVLwpMember() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "non-NVM-stored struct type has NVM-stored member"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S {",
        "  int nvl * nvl_wp p;",
        "} s = {0};",
      });
  }

  @Test public void nonNVMStructWithNVLArrayMember() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "non-NVM-stored struct type has NVM-stored member"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S {",
        "  int i;",
        "  int nvl arr[];",
        "} s;",
      });
  }

  @Test public void nonNVMStructWithNVLwpArrayMember() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "non-NVM-stored struct type has NVM-stored member"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S {",
        "  int i;",
        "  int nvl * nvl_wp arr[];",
        "} s;",
      });
  }

  @Test public void ptrToNonNVMStructWithNVMMember() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "non-NVM-stored struct type has NVM-stored member"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S {",
        "  int nvl i;",
        "};",
        "void foo(struct S *p);"
      });
  }

  @Test public void typedefNonNVMStructWithNVMMember() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "non-NVM-stored struct type has NVM-stored member"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "typedef struct S {",
        "  int nvl * nvl_wp i;",
        "} S;",
      });
  }

  /** 
   * {@link #recursiveNonNVMStructWithNVMMember} is the same except it
   * declares the pointer recursively.
   */
  @Test public void incompleteNonNVMStructWithNVMMember() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "non-NVM-stored struct type has NVM-stored member"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S *p;",
        "struct S {",
        "  int nvl i;",
        "};",
      });
  }

  /**
   * This used to cause an infinite recursion before we were careful to skip
   * validation of qualified types that had already been cached and thus
   * validated.
   */
  @Test public void recursiveNVMStructWithNVMMember() throws IOException {
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S {",
        "  struct S nvl *p;",
        "};",
      });
  }

  @Test public void nestedNonNVMStructWithNVMMember() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "non-NVM-stored struct type has NVM-stored member"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct C {",
        "  struct P {",
        "    struct S {",
        "      int nvl * nvl_wp i;",
        "    } s1, s2;",
        "  } p;",
        "};",
      });
  }

  @Test public void nestedNVMStructWithNVMMember() throws IOException {
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct C {",
        "  struct P {",
        "    struct S {",
        "      int nvl i;",
        "      int nvl * nvl_wp p;",
        "    } nvl s1, s2;",
        "  } nvl p;",
        "};",
        "struct C nvl *p = 0;",
      });
  }

  // -----------------------------------------------------------------------
  // constraint: if the type of any member of a struct type is NVM-stored,
  // then the type of every member shall be NVM-storable

  @Test public void membersNVLAndPtrToNonNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S {",
        "  int nvl i;",
        "  int *p;",
        "};",
      });
  }

  @Test public void membersNVLwpAndPtrToNonNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S {",
        "  int *p1;",
        "  int nvl * nvl_wp p2;",
        "};",
      });
  }

  @Test public void membersNVMAndFnPtr() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S {",
        "  int nvl a[5];",
        "  int (*p)();",
        "};",
      });
  }

  @Test public void membersNVMArrayAndPtrToNonNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S {",
        "  int *p;",
        "  int nvl * nvl_wp a[];",
        "};",
      });
  }

  /**
   * Same as {@link #incompleteNonNVMStructWithNVMMember} except declare the
   * pointer recursively.
   */
  @Test public void recursiveNonNVMStructWithNVMMember() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S {",
        "  struct S *p;",
        "  int nvl i;",
        "};",
      });
  }

  // -----------------------------------------------------------------------
  // constraint: if a struct type is NVM-stored anywhere in a translation
  // unit, it shall be complete by the end of the translation unit

  @Test public void neverCompleteNVMStruct() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored struct type is incomplete in translation unit"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S nvl *p = 0;",
      });
  }

  @Test public void neverCompleteLocalNVMStruct() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored struct type is incomplete in translation unit"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  struct S nvl *p = 0;",
        "}",
      });
  }

  // -------------------------------------------------------
  // constraint: uses of union types shall not be NVM-stored

  @Test public void nvlUnion() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo("union type is NVM-stored"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { union U nvl u; };",
      });
  }

  @Test public void nvlWpUnion() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "non-pointer type has nvl_wp type qualifier"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S { union U nvl_wp u; };",
      });
  }

  @Test public void ptrToNVMUnion() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo("union type is NVM-stored"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "union U nvl *p;",
      });
  }

  // ---------------------------------------------------------------------
  // constraint: union member types shall not reference NVM storage in any
  // way

  @Test public void unionHasNVL() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "union member type refers to NVM storage"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "union U { int nvl i; };",
      });
  }

  @Test public void unionHasNvlWp() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "union member type refers to NVM storage"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "union U { int nvl * nvl_wp p; };",
      });
  }

  @Test public void unionHasNvlArray() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "union member type refers to NVM storage"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "union U { int i; int nvl a[5]; int j; };",
      });
  }

  @Test public void unionHasNvlWpArray() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "union member type refers to NVM storage"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "union U { int i; int nvl * nvl_wp a[5]; int j; };",
      });
  }

  @Test public void unionHasPtrToNvl() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "union member type refers to NVM storage"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "union U { int nvl *i; };",
      });
  }

  @Test public void unionHasPtrToNvlWp() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "union member type refers to NVM storage"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "union U { int nvl * nvl_wp *i; };",
      });
  }

  @Test public void unionHasPtrToRecursiveStructThatHasNVM()
    throws IOException
  {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "union member type refers to NVM storage"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        // search for NVM storage would go into infinite recursion on first
        // member of union U and then first member of struct S if we didn't
        // protect against infinite recursion
        "struct S { struct S *t; int nvl *i; };",
        "union U { struct S *t; };",
      });
  }

  @Test public void unionHasPtrToFnWithParamPtrToNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "union member type refers to NVM storage"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "int (*fn)(int nvl *p);",
        "union U { int (*fn)(int nvl * nvl_wp *p); };",
      });
  }

  // -------------------------------------------------------------------------
  // constraint: given any two pointer types such that only one pointer type's
  // target type is nvl-qualified or only pointer type's target type is
  // nvl_wp-qualified, then those pointer types shall not be converted to one
  // another in any way

  @Test public void initPtrLosesTargetNvl() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "initialization changes NVM storage of pointer target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  int     *p1 = 0;",
        "  int nvl *p2 = p1;",
        "}",
      });
  }

  @Test public void initPtrLosesTargetNvlWp() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "initialization changes NVM storage of pointer target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  int nvl *        *p1 = 0;",
        "  int nvl * nvl_wp *p2 = p1;",
        "}",
      });
  }

  @Test public void initPtrGainsTargetNvl() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "initialization changes NVM storage of pointer target type"));
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  int nvl *p1 = 0;",
        "  int     *p2 = p1;",
        "}",
      });
  }

  @Test public void initPtrGainsTargetNvlWp() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "initialization changes NVM storage of pointer target type"));
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  int nvl * nvl_wp *p1 = 0;",
        "  int nvl *        *p2 = p1;",
        "}",
      });
  }

  @Test public void initPtrChangesTargetNvlWpToNvl() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "initialization changes NVM storage of pointer target type"));
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  int nvl * nvl_wp *p1 = 0;",
        "  int nvl * nvl    *p2 = p1;",
        "}",
      });
  }

  @Test public void initPtrChangesTargetNvlToNvlWp() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "initialization changes NVM storage of pointer target type"));
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  int nvl * nvl    *p1 = 0;",
        "  int nvl * nvl_wp *p2 = p1;",
        "}",
      });
  }

  @Test public void assignPtrLosesTargetNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "assignment operator changes NVM storage of pointer target type"));
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  int nvl *     *p1 = 0;",
        "  int nvl * nvl *p2 = 0;",
        "  p1 = p2;",
        "}",
      });
  }

  @Test public void assignPtrGainsTargetNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "assignment operator changes NVM storage of pointer target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  int nvl * nvl_wp *p1 = 0;",
        "  int nvl *        *p2 = 0;",
        "  p1 = p2;",
        "}",
      });
  }

  @Test public void assignPtrChangesTargetNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "assignment operator changes NVM storage of pointer target type"));
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  int nvl * nvl_wp *p1 = 0;",
        "  int nvl * nvl    *p2 = 0;",
        "  p1 = p2;",
        "}",
      });
  }

  @Test public void argPtrLosesTargetNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "argument 1 to \"fn1\" changes NVM storage of pointer target type"));
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "void fn1(int nvl **param);",
        "void fn2() {",
        "  int nvl * nvl_wp *arg = 0;",
        "  fn1(arg);",
        "}",
      });
  }

  @Test public void argPtrGainsTargetNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "argument 1 to \"fn1\" changes NVM storage of pointer target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void fn1(int nvl *param);",
        "void fn2() {",
        "  int *arg = 0;",
        "  fn1(arg);",
        "}",
      });
  }

  @Test public void returnPtrLosesTargetNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "return statement changes NVM storage of pointer target type"));
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "int *fn() {",
        "  int nvl *p = 0;",
        "  return p;",
        "}",
      });
  }

  @Test public void returnPtrChangesTargetNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "return statement changes NVM storage of pointer target type"));
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "int nvl * nvl_wp *fn() {",
        "  int nvl * nvl *p = 0;",
        "  return p;",
        "}",
      });
  }

  @Test public void castPtrChangesTargetNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "explicit cast changes NVM storage of pointer target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  int nvl * nvl_wp *p = 0;",
        "  (int nvl * nvl *)p;",
        "}",
      });
  }

  @Test public void castPtrGainsTargetNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "explicit cast changes NVM storage of pointer target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  int *p = 0;",
        "  (int nvl *)p;",
        "}",
      });
  }

  @Test public void ptrGainsArrayTargetNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "explicit cast changes NVM storage of pointer target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  int (*p)[] = 0;",
        "  (int nvl (*)[])p;",
        "}",
      });
  }

  @Test public void ptrLosesArrayTypedefTargetNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "explicit cast changes NVM storage of pointer target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "typedef int T[];",
        "void fn() {",
        "  T nvl * nvl_wp *p = 0;",
        "  (T nvl **)p;",
        "}",
      });
  }

  // -------------------------------------------------------------------------
  // constraint: given any two pointer types such that either target type
  // involves NVM storage in any way and unqualified versions of their target
  // types are not compatible, then those pointer types shall not be converted
  // to one another in any way

  public static final String INCOMPATIBLE_TARGET_MSG
    = "requires conversion between pointer types with target types that"
      +" involve NVM storage and that have incompatible unqualified versions";

  @Test public void ptrTargetsIntFloat() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "initialization "+INCOMPATIBLE_TARGET_MSG+": floating type is"
      +" incompatible with non-floating type: int"));
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  int   nvl *p1 = 0;",
        "  float nvl *p2 = p1;",
        "}",
      });
  }

  @Test public void ptrTargetsConstNonConst() throws IOException {
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  int const nvl *p1 = 0;",
        "  int       nvl *p2 = 0;",
        "  p1 = p2;",
        "}",
      });
  }

  @Test public void ptrPtrNvlTargetsConstNonConst() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "assignment operator "+INCOMPATIBLE_TARGET_MSG+": pointer types are"
      +" incompatible because their target types are incompatible: types"
      +" are incompatible because only one of them has each of the"
      +" following type qualifiers: const"));
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  int const nvl **p1 = 0;",
        "  int       nvl **p2 = 0;",
        "  p1 = p2;",
        "}",
      });
  }

  @Test public void ptrNvlWpPtrNvlTargetsConstNonConst() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "assignment operator "+INCOMPATIBLE_TARGET_MSG+": pointer types are"
      +" incompatible because their target types are incompatible: types"
      +" are incompatible because only one of them has each of the"
      +" following type qualifiers: const"));
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  int const nvl * nvl_wp *p1 = 0;",
        "  int       nvl * nvl_wp *p2 = 0;",
        "  p1 = p2;",
        "}",
      });
  }

  @Test public void ptrNvlPtrTargetsNvlWpNvl() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "assignment operator "+INCOMPATIBLE_TARGET_MSG+": pointer types are"
      +" incompatible because their target types are incompatible: types"
      +" are incompatible because only one of them has each of the"
      +" following type qualifiers: nvl, nvl_wp"));
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  int nvl * nvl_wp * nvl *p1 = 0;",
        "  int nvl * nvl    * nvl *p2 = 0;",
        "  p1 = p2;",
        "}",
      });
  }

  @Test public void ptrPtrTargetsNVMNonNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "assignment operator "+INCOMPATIBLE_TARGET_MSG+": pointer types are"
      +" incompatible because their target types are incompatible: types"
      +" are incompatible because only one of them has each of the"
      +" following type qualifiers: nvl"));
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  int     **p1 = 0;",
        "  int nvl **p2 = 0;",
        "  p1 = p2;",
        "}",
      });
  }

  @Test public void ptrPtrNvlPtrTargetsNvlWpNvl() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "assignment operator "+INCOMPATIBLE_TARGET_MSG+": pointer types are"
      +" incompatible because their target types are incompatible: pointer"
      +" types are incompatible because their target types are"
      +" incompatible: types are incompatible because only one of them has"
      +" each of the following type qualifiers: nvl, nvl_wp"));
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  int nvl * nvl_wp * nvl **p1 = 0;",
        "  int nvl * nvl    * nvl **p2 = 0;",
        "  p1 = p2;",
        "}",
      });
  }

  @Test public void ptrPtrConstNonConstCompleteStructsWithNvl()
    throws IOException
  {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "argument 1 to \"fn1\" "+INCOMPATIBLE_TARGET_MSG+": pointer types are"
      +" incompatible because their target types are incompatible: types"
      +" are incompatible because only one of them has each of the"
      +" following type qualifiers: const"));
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "struct S { int nvl *p; };",
        "void fn1(struct S **p);",
        "void fn2() {",
        "  struct S const **p = 0;",
        "  fn1(p);",
        "}",
      });
  }

  @Test public void ptrPtrConstNonConstCompleteStructsWithNvlWp()
    throws IOException
  {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "argument 1 to \"fn1\" "+INCOMPATIBLE_TARGET_MSG+": pointer types are"
      +" incompatible because their target types are incompatible: types"
      +" are incompatible because only one of them has each of the"
      +" following type qualifiers: const"));
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "struct S { int nvl * nvl_wp *p; };",
        "void fn1(struct S **p);",
        "void fn2() {",
        "  struct S const **p = 0;",
        "  fn1(p);",
        "}",
      });
  }

  @Test public void ptrPtrConstNonConstNeverCompleteStructs()
    throws IOException
  {
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "void fn1(struct S **p);",
        "void fn2() {",
        "  struct S const **p = 0;",
        "  fn1(p);",
        "}",
      });
  }

  @Test public void ptrPtrConstNonConstLaterCompleteStructsWithNvl()
    throws IOException
  {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "return statement "+INCOMPATIBLE_TARGET_MSG+": pointer types are"
      +" incompatible because their target types are incompatible: types"
      +" are incompatible because only one of them has each of the"
      +" following type qualifiers: const"));
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "struct S **fn() {",
        "  struct S const **p = 0;",
        "  return p;",
        "}",
        "struct S { int nvl *p; };",
      });
  }

  @Test public void ptrPtrConstNonConstLaterCompleteStructsWithNvlWp()
    throws IOException
  {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "return statement "+INCOMPATIBLE_TARGET_MSG+": pointer types are"
      +" incompatible because their target types are incompatible: types"
      +" are incompatible because only one of them has each of the"
      +" following type qualifiers: const"));
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "struct S **fn() {",
        "  struct S const **p = 0;",
        "  return p;",
        "}",
        "struct S { int nvl * nvl_wp *p; };",
      });
  }

  @Test public void ptrIncompatibleArray() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "explicit cast "+INCOMPATIBLE_TARGET_MSG+": array types are"
      +" incompatible because they have different sizes"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  int nvl (*p)[5] = 0;",
        "  (int nvl (*)[3])p;",
        "}",
      });
  }

  @Test public void ptrIncompatibleArrayTypedef() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "explicit cast "+INCOMPATIBLE_TARGET_MSG+": array types are"
      +" incompatible because they have different sizes"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "typedef int nvl *a5i[5];",
        "typedef int nvl *a4i[4];",
        "void fn() {",
        "  a5i nvl_wp *p = 0;",
        "  (a4i nvl_wp *)p;",
        "}",
      });
  }

  // -----------------------------------------------------------------------
  // constraint: any pointer type whose target type involves NVM storage in
  // any way shall not be converted to or from an integer except that such a
  // pointer can be converted to _Bool and a null pointer constant can be
  // converted to such a pointer

  @Test public void ptrConvertToBool() throws IOException {
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "int nvl *p1 = 0;",
        "int nvl * nvl_wp *p2 = 0;",
        "struct { int nvl *p; } nvl *p3 = 0;",
        "struct { int nvl * nvl_wp p; } nvl *p4 = 0;",
        "void fn() {",
        "  _Bool b1 = p1;",
        "  _Bool b2 = p2;",
        "  _Bool b3 = p3->p;",
        "  _Bool b4 = p4->p;",
        "}",
      });
  }

  @Test public void nullPointerConstantConvertToPtrToNVM() throws IOException {
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "#include <stdlib.h>",
        "int nvl * nvl    *p1 = 0;",
        "int nvl * nvl    *p2 = (void*)0;",
        "int nvl * nvl    *p3 = NULL;",
        "int nvl * nvl_wp *p4 = 0;",
        "int nvl * nvl_wp *p5 = (void*)0;",
        "int nvl * nvl_wp *p6 = NULL;",
        "void fn() {",
        "  *p1 = 0;",
        "  *p2 = (void*)0;",
        "  *p3 = 0;",
        "  *p4 = NULL;",
        "  *p5 = (void*)0;",
        "  *p6 = NULL;",
        "}",
      });
  }

  public static final String PTR_TO_INT_MSG
    = "requires conversion to non-_Bool integer type from pointer type"
      +" involving NVM storage";

  @Test public void ptrToNvlConvertToInt() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "initialization "+PTR_TO_INT_MSG));
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "int nvl *p = 0;",
        "void fn() {",
        "  int i = p;",
        "}",
      });
  }

  @Test public void ptrToNvlWpConvertToInt() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "initialization "+PTR_TO_INT_MSG));
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "int nvl * nvl_wp *p = 0;",
        "void fn() {",
        "  int i = p;",
        "}",
      });
  }

  @Test public void ptrToNVMConvertToChar() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "assignment operator "+PTR_TO_INT_MSG));
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "int nvl *p = 0;",
        "char c;",
        "void fn() {",
        "  c = p;",
        "}",
      });
  }

  @Test public void ptrToIncompleteStructWithNVMConvertToLong()
    throws IOException
  {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "argument 1 to \"fn1\" "+PTR_TO_INT_MSG));
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "struct S *p = 0;",
        "void fn1(long l);",
        "void fn2() {",
        "  fn1(p);",
        "}",
        "struct S { int nvl * nvl_wp *p; };",
      });
  }

  public static final String INT_TO_PTR_MSG
    = "requires conversion from integer type to pointer type involving NVM"
      +" storage";

  @Test public void intToPtrToNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "return statement "+INT_TO_PTR_MSG));
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "float nvl * nvl_wp *fn() {",
        "  return 5;",
        "}",
      });
  }

  @Test public void charToPtrToNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "explicit cast "+INT_TO_PTR_MSG));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "char i = 0;",
        "void fn() {",
        "  (float nvl *)i;",
        "}",
      });
  }

  @Test public void longLongConvertToPtrToIncompleteStructWithNVM()
    throws IOException
  {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "initialization "+INT_TO_PTR_MSG));
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "long long ll = 0;",
        "struct S1;",
        "void fn() {",
        "  struct S1 *p = ll;",
        "}",
        "struct S1 { struct S2 *p; };",
        "struct S2 { int i; float nvl *nvl_wp (*a)[]; };",
      });
  }

  /**
   * Exercise a few special cases of recursively searching a type for NVM
   * storage that led to infinite recursion when our implementation was not as
   * careful. Integer to pointer conversion is just an example that performs
   * such a search.
   */
  @Test public void inifiniteRecursion() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "initialization "+INT_TO_PTR_MSG));
    buildLLVMSimple(
      "", "", false,
      new String[]{
        "#include <nvl.h>",
        "struct T {",
           // pointers are one avenue to recursive type definitions
        "  struct T const *p1;",
           // return and param types are other avenues
        "  struct T const (*p2)(struct T const t);",
           // the type qualifiers above might hide the type from the
           // infinite recursion guard
           // to prove we made it past the recursive searches safely, add some
           // NVM storage (which will terminate the search, so it comes last)
           // so we'll get an error
        "  int nvl *p3;",
        "} *p = 5;",
      });
  }

  // -----------------------------------------------------------------------
  // constraint: the default argument promotions produce an invalid type for
  // any type that involves NVM storage in any way

  @Test public void variadicArg1HasNvl() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "invalid argument 2 to \"foo\": default argument promotions applied"
      +" to type that involves NVM storage"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "nvl int *p = 0;",
        "void foo(int i, ...);",
        "void bar() { foo(3, p, 5); }",
      });
  }

  @Test public void paramType1HasNvlCompatWithNoType() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "incompatible declarations of function \"foo\": function types are"
      +" incompatible because one has an unspecified parameter list while"
      +" the other's parameter 1 type is not compatible with itself after"
      +" default argument promotions: default argument promotions applied"
      +" to type that involves NVM storage"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void foo();",
        "void foo(nvl int *p, int i);",
      });
  }

  @Test public void variadicArg2HasNvlWp() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "invalid argument 3 to \"foo\": default argument promotions applied"
      +" to type that involves NVM storage"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "nvl int * nvl_wp *p = 0;",
        "void foo(int i, ...);",
        "void bar() { foo(3, 5, p); }",
      });
  }

  @Test public void paramType2HasNvlWpCompatWithNoType() throws IOException
  {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "incompatible declarations of function \"foo\": function types are"
      +" incompatible because one has an unspecified parameter list while"
      +" the other's parameter 2 type is not compatible with itself after"
      +" default argument promotions: default argument promotions applied"
      +" to type that involves NVM storage"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void foo(int i, nvl int * nvl_wp *p);",
        "void foo();",
      });
  }

  @Test public void variadicArg1HasIncompleteStructWithNvl()
    throws IOException
  {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "invalid argument 2 to \"foo\": default argument promotions applied"
      +" to type that involves NVM storage"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct T *p;",
        "void foo(int i, ...);",
        "void bar() { foo(3, p); }",
        "struct T {",
        "  nvl int *p;",
        "};",
      });
  }

  @Test public void paramType1HasIncompleteStructWithNVMCompatWithNoType()
    throws IOException
  {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "incompatible declarations of function \"foo\": function types are"
      +" incompatible because one has an unspecified parameter list while"
      +" the other's parameter 1 type is not compatible with itself after"
      +" default argument promotions: default argument promotions applied"
      +" to type that involves NVM storage"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct T;",
        "void foo();",
        "void foo(struct T *p);",
        "struct T {",
        "  nvl int * nvl_wp *p;",
        "};",
      });
  }

  // ---------
  // addrspace

  @Test public void addrspace() throws IOException {
    // First, make sure the address spaces encoded in OpenARC's LLVM backend
    // are sane before we use them directly when checking the output LLVM IR
    // below.
    assertEquals("OpenARC's LLVM addrspace default", 0,
                 SrcPointerType.LLVM_ADDRSPACE_DEFAULT);
    assertNotEquals("OpenARC's LLVM addrspace for nvl must not be the default",
                    SrcPointerType.LLVM_ADDRSPACE_DEFAULT,
                    SrcPointerType.LLVM_ADDRSPACE_NVL);
    assertNotEquals("OpenARC's LLVM addrspace for nvl_wp must not be the"
                    +" default",
                    SrcPointerType.LLVM_ADDRSPACE_DEFAULT,
                    SrcPointerType.LLVM_ADDRSPACE_NVL_WP);
    assertNotEquals("OpenARC's LLVM addrspace for nvl_wp must not be the same"
                    +" as for nvl",
                    SrcPointerType.LLVM_ADDRSPACE_NVL,
                    SrcPointerType.LLVM_ADDRSPACE_NVL_WP);

    final int nvlPtrSize = 128;
    final int nvlWpPtrSize = 256;
    final SimpleResult simpleResult = buildLLVMSimple(
      "",
      "p"+SrcPointerType.LLVM_ADDRSPACE_NVL
      +":"+nvlPtrSize+":"+nvlPtrSize+":"+nvlPtrSize
      +"-p"+SrcPointerType.LLVM_ADDRSPACE_NVL_WP
      +":"+nvlWpPtrSize+":"+nvlWpPtrSize+":"+nvlWpPtrSize,
      new String[]{
        "#include <nvl.h>",
        "#include <stddef.h>",
        "int nvl *nvlPtr = 0;",
        "int nvl * nvl_wp *nvlWpPtr = 0;",
        "size_t nvlPtrSize = sizeof nvlPtr;",
        "size_t nvlWpPtrSize = sizeof nvlWpPtr;",
        "int nvl *nvlPtrArr[5] = {0, 0, 0, 0, 0};",
        "int nvl * nvl_wp *nvlWpPtrArr[3] = {0, 0, 0};",
        "size_t nvlPtrArrSize = sizeof nvlPtrArr;",
        "size_t nvlWpPtrArrSize = sizeof nvlWpPtrArr;",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMPointerType nvlPtrType
      = LLVMPointerType.get(SrcIntType.getLLVMType(ctxt),
                            SrcPointerType.LLVM_ADDRSPACE_NVL);
    final LLVMPointerType nvlWpPtrType
      = LLVMPointerType.get(nvlPtrType,
                            SrcPointerType.LLVM_ADDRSPACE_NVL_WP);
    checkGlobalVar(mod, "nvlPtr", LLVMConstant.constNull(nvlPtrType));
    checkGlobalVar(mod, "nvlWpPtr", LLVMConstant.constNull(nvlWpPtrType));
    checkIntGlobalVar(mod, "nvlPtrSize", nvlPtrSize/8);
    checkIntGlobalVar(mod, "nvlWpPtrSize", nvlWpPtrSize/8);
    checkGlobalVar(mod, "nvlPtrArr",
                   LLVMConstant.constNull(LLVMArrayType.get(nvlPtrType, 5)));
    checkGlobalVar(mod, "nvlWpPtrArr",
                   LLVMConstant.constNull(LLVMArrayType.get(nvlWpPtrType, 3)));
    checkIntGlobalVar(mod, "nvlPtrArrSize", nvlPtrSize*5/8);
    checkIntGlobalVar(mod, "nvlWpPtrArrSize", nvlWpPtrSize*3/8);
  }

  // ----------
  // lower icmp

  @Test public void lower_icmp_v2nv() throws Exception {
    lower_icmp("nvl int");
  }
  @Test public void lower_icmp_v2nv2nv() throws Exception {
    lower_icmp("nvl int * nvl");
  }
  @Test public void lower_icmp_v2wp() throws Exception {
    lower_icmp("nvl int * nvl_wp");
  }
  private void lower_icmp(String targetTy) throws Exception {
    final String ty = targetTy + "*";
    LLVMTargetData.initializeNativeTarget();
    final String heapFile0 = mkHeapFile("0.nvl").getAbsolutePath();
    final String heapFile1 = mkHeapFile("1.nvl").getAbsolutePath();
    final File workDir = mkTmpDir("");
    final SimpleResult simpleResult = buildLLVMSimple(
      targetTriple, targetDataLayout,
      new String[]{
        "#include <nvl.h>",
        "#include <stddef.h>",
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "_Bool c_bool_c0  = ("+ty+")0;",
        "_Bool c_bool_c1  = ("+ty+")0 + 1;",
        "_Bool c_not_c0   = !("+ty+")0;",
        "_Bool c_not_c1   = !(("+ty+")0 + 1);",
        "_Bool c_cond_c0  = ("+ty+")0 ? 1 : 0;",
        "_Bool c_cond_c1  = (("+ty+")0+1) ? 1 : 0;",
        "_Bool c_eq_c0_c0 = ("+ty+")0 == ("+ty+")0;",
        "_Bool c_eq_c1_c0 = (("+ty+")0+1) == ("+ty+")0;",
        "_Bool c_eq_c1_c1 = (("+ty+")0+1) == (("+ty+")0+1);",
        "_Bool c_ne_c0_c0 = ("+ty+")0 != ("+ty+")0;",
        "_Bool c_ne_c2_c0 = (("+ty+")0+2) != (("+ty+")0+0);",
        "_Bool c_ne_c2_c1 = (("+ty+")0+2) != (("+ty+")0+1);",
        "_Bool get_c_ne_c2_c1() { return c_ne_c2_c1; }",
        "_Bool c_ne_c2_c2 = (("+ty+")0+2) != (("+ty+")0+2);",
        "_Bool c_lt_c0_c0 = ("+ty+")0 < ("+ty+")0;",
        "_Bool c_lt_c0_c1 = ("+ty+")0 < (("+ty+")0+1);",
        "_Bool c_lt_c1_c0 = (("+ty+")0+1) < (("+ty+")0+0);",
        "_Bool c_lt_c2_c1 = (("+ty+")0+2) < (("+ty+")0+1);",
        "_Bool get_c_lt_c2_c1() { return c_lt_c2_c1; }",
        "_Bool c_gt_c0_c0 = ("+ty+")0 > ("+ty+")0;",
        "_Bool c_gt_c0_c1 = ("+ty+")0 > (("+ty+")0+1);",
        "_Bool c_gt_c1_c0 = (("+ty+")0+1) > ("+ty+")0;",
        "_Bool c_le_c0_c0 = ("+ty+")0 <= ("+ty+")0;",
        "_Bool c_le_c0_c1 = ("+ty+")0 <= (("+ty+")0+1);",
        "_Bool c_le_c1_c0 = (("+ty+")0+1) <= ("+ty+")0;",
        "_Bool c_ge_c0_c0 = ("+ty+")0 >= ("+ty+")0;",
        "_Bool c_ge_c0_c1 = ("+ty+")0 >= (("+ty+")0+1);",
        "_Bool c_ge_c1_c0 = (("+ty+")0+1) >= ("+ty+")0;",

        "nvl_heap_t *heap0 = NULL;",
        "nvl_heap_t *heap1 = NULL;",
        ty+"ph0a0o0 = NULL;",
        ty+"ph0a0o1 = NULL;",
        ty+"ph0a1o0 = NULL;",
        ty+"ph1a0o0 = NULL;",
        "int setup() {",
        "  if (heap0 || heap1) return 1;",
        "  heap0 = nvl_create(\""+heapFile0+"\", 0, 0600);",
        "  heap1 = nvl_create(\""+heapFile1+"\", 0, 0600);",
        "  if (!heap0 || !heap1) return 2;",
        "  ph0a0o0 = nvl_alloc_nv(heap0, 5, "+targetTy+");",
        "  ph0a0o1 = ph0a0o0 + 1;",
        "  ph0a1o0 = nvl_alloc_nv(heap0, 1, "+targetTy+");",
        "  ph1a0o0 = nvl_alloc_nv(heap1, 1, "+targetTy+");",
        "  return 0;",
        "}",

        "#define A ph0a0o0",
        "_Bool bool_0()    {"+ty+"p = 0; return p;}",
        "_Bool bool_a()    {"+ty+"p = A; return p;}",
        "_Bool not_0()     {"+ty+"p = 0; return !p;}",
        "_Bool not_a()     {"+ty+"p = A; return !p;}",
        "_Bool cond_0()    {"+ty+"p = 0; if (p) return 1; return 0;}",
        "_Bool cond_a()    {"+ty+"p = A; if (p) return 1; return 0;}",
        "_Bool eq_c0_0()   {"+ty+"p = 0; return ("+ty+")0 == p;}",
        "_Bool eq_0_a()    {"+ty+"p0 = 0; "+ty+"p1 = A; return p0 == p1;}",
        "_Bool eq_a_a()    {"+ty+"p0 = A; "+ty+"p1 = p0;return p0 == p1;}",
        "_Bool ne_0_c0()   {"+ty+"p = 0; return p != ("+ty+")0;}",
        "_Bool ne_a_0()    {"+ty+"p0 = 0; "+ty+"p1 = A; return p1 != p0;}",
        "_Bool ne_a2_a2()  {"+ty+"p0=A+2; "+ty+"p1 = p0; return p0 != p1;}",
        "_Bool lt_0_0()    {"+ty+"p1 = 0; "+ty+"p2 = 0; return p1 < p2;}",
        "_Bool lt_ia1_a0() {"+ty+"p = A; return A+1 < p;}",
        "_Bool lt_a0_a1()  {"+ty+"p0 = A; "+ty+"p1 = A+1; return p0 < p1;}",
        "_Bool gt_c0_0()   {"+ty+"p = 0; return ("+ty+")0 > p;}",
        "_Bool gt_a2_a2()  {"+ty+"p0=A+2; "+ty+"p1=p0; return p0>p1;}",
        "_Bool gt_ia2_a1() {"+ty+"p = A+1; return A+2 > p;}",
        "_Bool gt_a1_a2()  {"+ty+"p0 = A+1; "+ty+"p1=p0+1; return p0>p1;}",
        "_Bool le_0_c0()   {"+ty+"p = 0; return p <= ("+ty+")0;}",
        "_Bool le_a0_ia1() {"+ty+"p = A; return p <= A+1;}",
        "_Bool le_ia1_a0() {"+ty+"p = A; return A+1 <= p;}",
        "_Bool ge_0_0()    {"+ty+"p0 = 0; "+ty+"p1 = 0; return p0 >= p1;}",
        "_Bool ge_a1_a0()  {"+ty+"p0 = A+1; "+ty+"p1 = A; return p0 >= p1;}",
        "_Bool ge_a2_a5()  {"+ty+"p0 = A+2; "+ty+"p1 = A+5; return p0 >= p1;}",

        // libnvlrt_pmemobj normally supplies this, but we want this test to
        // succeed on systems (OS X) that don't have that.
        "void nvlrt_report_heapAlloc() {",
        "  fprintf(stderr, \"unexpected nvlrt_report_heapAlloc call\");",
        "  exit(99);",
        "}",

        // icmp that always has constant second operand.
        "_Bool bool_n0() {return ph0a0o0; }",
        "_Bool not_n0()  {return !ph0a0o1; }",
        "_Bool cond_n0() {return ph0a0o0 && ph1a0o0; }",

        // Same obj.
        "_Bool eq_ph0a0o0_ph0a0o0() {return ph0a0o0 == ph0a0o0;}",
        "_Bool ne_ph0a0o0_ph0a0o0() {return ph0a0o0 != ph0a0o0;}",
        "_Bool lt_ph0a0o0_ph0a0o0() {return ph0a0o0 <  ph0a0o0;}",
        "_Bool gt_ph0a0o0_ph0a0o0() {return ph0a0o0 >  ph0a0o0;}",
        "_Bool le_ph0a0o0_ph0a0o0() {return ph0a0o0 <= ph0a0o0;}",
        "_Bool ge_ph0a0o0_ph0a0o0() {return ph0a0o0 >= ph0a0o0;}",

        // Same alloc, different obj.
        "_Bool eq_ph0a0o0_ph0a0o1() {return ph0a0o0 == ph0a0o1;}",
        "_Bool ne_ph0a0o0_ph0a0o1() {return ph0a0o0 != ph0a0o1;}",
        "_Bool lt_ph0a0o0_ph0a0o1() {return ph0a0o0 <  ph0a0o1;}",
        "_Bool gt_ph0a0o0_ph0a0o1() {return ph0a0o0 >  ph0a0o1;}",
        "_Bool le_ph0a0o0_ph0a0o1() {return ph0a0o0 <= ph0a0o1;}",
        "_Bool ge_ph0a0o0_ph0a0o1() {return ph0a0o0 >= ph0a0o1;}",

        // Same alloc, different obj, reversed.
        "_Bool eq_ph0a0o1_ph0a0o0() {return ph0a0o1 == ph0a0o0;}",
        "_Bool ne_ph0a0o1_ph0a0o0() {return ph0a0o1 != ph0a0o0;}",
        "_Bool lt_ph0a0o1_ph0a0o0() {return ph0a0o1 <  ph0a0o0;}",
        "_Bool gt_ph0a0o1_ph0a0o0() {return ph0a0o1 >  ph0a0o0;}",
        "_Bool le_ph0a0o1_ph0a0o0() {return ph0a0o1 <= ph0a0o0;}",
        "_Bool ge_ph0a0o1_ph0a0o0() {return ph0a0o1 >= ph0a0o0;}",

        // Same heap, different alloc.
        "_Bool eq_ph0a0o0_ph0a1o0() {return ph0a0o0 == ph0a1o0;}",
        "_Bool ne_ph0a0o0_ph0a1o0() {return ph0a0o0 != ph0a1o0;}",

        // Different heap.
        "_Bool eq_ph0a0o0_ph1a0o0() {return ph0a0o0 == ph1a0o0;}",
        "_Bool ne_ph0a0o0_ph1a0o0() {return ph0a0o0 != ph1a0o0;}",

        // As of pmem 0.3, reopening a pool file before closing it (which
        // terminating the process seems to achieve also) causes the open
        // to fail, so close the heap file used above before launching new
        // processes that open the heap file below.
        "int cleanup() {",
        "  if (!heap0 || !heap1) return 1;",
        "  nvl_close(heap0);",
        "  nvl_close(heap1);",
        "  return 0;",
        "}",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    final LLVMModulePassManager pm = new LLVMModulePassManager();
    pm.addTargetData(exec.getTargetData());
    addNVLPasses(pm);
    pm.run(mod);
    mod.dump();
    checkIntGlobalVar(mod, "c_bool_c0",  0);
    checkIntGlobalVar(mod, "c_bool_c1",  1);
    checkIntGlobalVar(mod, "c_not_c0",   1);
    checkIntGlobalVar(mod, "c_not_c1",   0);
    checkIntGlobalVar(mod, "c_cond_c0",  0);
    checkIntGlobalVar(mod, "c_cond_c1",  1);
    checkIntGlobalVar(mod, "c_eq_c0_c0", 1);
    checkIntGlobalVar(mod, "c_eq_c1_c0", 0);
    checkIntGlobalVar(mod, "c_eq_c1_c1", 1);
    checkIntGlobalVar(mod, "c_ne_c0_c0", 0);
    checkIntGlobalVar(mod, "c_ne_c2_c0", 1);
    // TODO: LLVM (3.2) doesn't fold the initializer for each commented case
    // here to a constant integer, so we cannot check its value here. If we
    // call a function to retrieve its value, we get "LLVM ERROR:
    // ConstantExpr not handled", apparently from ExecutionEngine.cpp (the
    // only file to contain that message).
    //checkIntGlobalVar(mod, "c_ne_c2_c1", 1);
    //checkIntFn(exec, mod, "get_c_ne_c2_c1", 1);
    checkIntGlobalVar(mod, "c_ne_c2_c2", 0);
    checkIntGlobalVar(mod, "c_lt_c0_c0", 0);
    checkIntGlobalVar(mod, "c_lt_c0_c1", 1);
    checkIntGlobalVar(mod, "c_lt_c1_c0", 0);
    //checkIntGlobalVar(mod, "c_lt_c2_c1", 0);
    //checkIntFn(exec, mod, "get_c_lt_c2_c1", 0);
    checkIntGlobalVar(mod, "c_gt_c0_c0", 0);
    checkIntGlobalVar(mod, "c_gt_c0_c1", 0);
    checkIntGlobalVar(mod, "c_gt_c1_c0", 1);
    checkIntGlobalVar(mod, "c_le_c0_c0", 1);
    checkIntGlobalVar(mod, "c_le_c0_c1", 1);
    checkIntGlobalVar(mod, "c_le_c1_c0", 0);
    checkIntGlobalVar(mod, "c_ge_c0_c0", 1);
    checkIntGlobalVar(mod, "c_ge_c0_c1", 0);
    checkIntGlobalVar(mod, "c_ge_c1_c0", 1);

    // The rest require at least nvlrt_inc_v.
    assumePmemobjLibs();
    checkIntFn(exec, mod, "setup",  0);

    checkIntFn(exec, mod, "bool_0",  0);
    checkIntFn(exec, mod, "bool_a",  1);
    checkIntFn(exec, mod, "not_0",   1);
    checkIntFn(exec, mod, "not_a",   0);
    checkIntFn(exec, mod, "cond_0",  0);
    checkIntFn(exec, mod, "cond_a",  1);
    checkIntFn(exec, mod, "eq_c0_0", 1);
    checkIntFn(exec, mod, "eq_0_a",  0);
    checkIntFn(exec, mod, "eq_a_a",  1);
    checkIntFn(exec, mod, "ne_0_c0", 0);
    checkIntFn(exec, mod, "ne_a_0",  1);
    checkIntFn(exec, mod, "ne_a2_a2",  0);
    checkIntFn(exec, mod, "lt_0_0",  0);
    checkIntFn(exec, mod, "lt_ia1_a0", 0);
    checkIntFn(exec, mod, "lt_a0_a1",  1);
    checkIntFn(exec, mod, "gt_c0_0", 0);
    checkIntFn(exec, mod, "gt_a2_a2",  0);
    checkIntFn(exec, mod, "gt_ia2_a1", 1);
    checkIntFn(exec, mod, "gt_a1_a2",  0);
    checkIntFn(exec, mod, "le_0_c0", 1);
    checkIntFn(exec, mod, "le_a0_ia1", 1);
    checkIntFn(exec, mod, "le_ia1_a0", 0);
    checkIntFn(exec, mod, "ge_0_0",  1);
    checkIntFn(exec, mod, "ge_a1_a0",  1);
    checkIntFn(exec, mod, "ge_a2_a5",  0);

    // icmp that always has constant second operand.
    checkIntFn(exec, mod, "bool_n0", 1);
    checkIntFn(exec, mod, "not_n0", 0);
    checkIntFn(exec, mod, "cond_n0", 1);

    // Same obj.
    checkIntFn(exec, mod, "eq_ph0a0o0_ph0a0o0", 1);
    checkIntFn(exec, mod, "ne_ph0a0o0_ph0a0o0", 0);
    checkIntFn(exec, mod, "lt_ph0a0o0_ph0a0o0", 0);
    checkIntFn(exec, mod, "gt_ph0a0o0_ph0a0o0", 0);
    checkIntFn(exec, mod, "le_ph0a0o0_ph0a0o0", 1);
    checkIntFn(exec, mod, "ge_ph0a0o0_ph0a0o0", 1);

    // Same alloc, different obj.
    checkIntFn(exec, mod, "eq_ph0a0o0_ph0a0o1", 0);
    checkIntFn(exec, mod, "ne_ph0a0o0_ph0a0o1", 1);
    checkIntFn(exec, mod, "lt_ph0a0o0_ph0a0o1", 1);
    checkIntFn(exec, mod, "gt_ph0a0o0_ph0a0o1", 0);
    checkIntFn(exec, mod, "le_ph0a0o0_ph0a0o1", 1);
    checkIntFn(exec, mod, "ge_ph0a0o0_ph0a0o1", 0);

    // Same alloc, different obj, reversed.
    checkIntFn(exec, mod, "eq_ph0a0o1_ph0a0o0", 0);
    checkIntFn(exec, mod, "ne_ph0a0o1_ph0a0o0", 1);
    checkIntFn(exec, mod, "lt_ph0a0o1_ph0a0o0", 0);
    checkIntFn(exec, mod, "gt_ph0a0o1_ph0a0o0", 1);
    checkIntFn(exec, mod, "le_ph0a0o1_ph0a0o0", 0);
    checkIntFn(exec, mod, "ge_ph0a0o1_ph0a0o0", 1);

    // Same heap, different alloc.
    checkIntFn(exec, mod, "eq_ph0a0o0_ph0a1o0", 0);
    checkIntFn(exec, mod, "ne_ph0a0o0_ph0a1o0", 1);

   // Different heap.
    checkIntFn(exec, mod, "eq_ph0a0o0_ph1a0o0", 0);
    checkIntFn(exec, mod, "ne_ph0a0o0_ph1a0o0", 1);

    checkIntFn(exec, mod, "cleanup",  0);

    exec.dispose();

    // The rest trap, so we need to run them in separate processes, or we'll
    // halt the test suite.

    String[] srcPre = new String[]{
      "#include <nvl.h>",
      "#include <signal.h>",
      "#include <stddef.h>",
      "#include <stdio.h>",
      "#include <stdlib.h>",
      "struct sigaction sigill_act;",
      "void sigill_handler(int signum, siginfo_t *info, void*) {",
      "  fprintf(stderr, \"received SIGILL\\n\");",
      "  exit(1);",
      "}",
      "int main() {",
      "  sigill_act.sa_flags = SA_SIGINFO;",
      "  sigill_act.sa_sigaction = sigill_handler;",
      "  sigaction(SIGILL, &sigill_act, 0);",
      "  nvl_heap_t *heap0 = nvl_open(\""+heapFile0+"\");",
      "  nvl_heap_t *heap1 = nvl_open(\""+heapFile1+"\");",
      "  if (!heap0 || !heap1) return 1;",
      "  "+ty+"ph0a0o0 = nvl_alloc_nv(heap0, 1, "+targetTy+");",
      "  "+ty+"ph0a1o0 = nvl_alloc_nv(heap0, 1, "+targetTy+");",
      "  "+ty+"ph1a0o0 = nvl_alloc_nv(heap1, 1, "+targetTy+");",
      "  if (!ph0a0o0 || !ph0a1o0 || !ph1a0o0) return 1;",
      "  fprintf(stderr, \"before instruction\\n\");"};
    String[] srcPost = new String[]{
      "  fprintf(stderr, \"after instruction\\n\");",
      "  return 0;",
      "}"};
    String[] stderrNoSafety = new String[]{
      "before instruction",
      "after instruction",
    };
    String[] stderrSafety = new String[]{
      "before instruction",
      "nvlrt-pmemobj: error: illegal operation on pointers to different NVM"
      +" allocations",
      "received SIGILL",
    };

    // code: different heaps.
    String[] lt_ph0a0o0_ph1a0o0
      = concatArrays(srcPre, new String[]{"  ph0a0o0 < ph1a0o0;"}, srcPost);
    String[] gt_ph0a0o0_ph1a0o0
      = concatArrays(srcPre, new String[]{"  ph0a0o0 > ph1a0o0;"}, srcPost);
    String[] le_ph0a0o0_ph1a0o0
      = concatArrays(srcPre, new String[]{"  ph0a0o0 <= ph1a0o0;"}, srcPost);
    String[] ge_ph0a0o0_ph1a0o0
      = concatArrays(srcPre, new String[]{"  ph0a0o0 >= ph1a0o0;"}, srcPost);

    // code: different allocations.
    String[] lt_ph0a0o0_ph0a1o0
      = concatArrays(srcPre, new String[]{"  ph0a0o0 < ph0a1o0;"}, srcPost);
    String[] gt_ph0a0o0_ph0a1o0
      = concatArrays(srcPre, new String[]{"  ph0a0o0 > ph0a1o0;"}, srcPost);
    String[] le_ph0a0o0_ph0a1o0
      = concatArrays(srcPre, new String[]{"  ph0a0o0 <= ph0a1o0;"}, srcPost);
    String[] ge_ph0a0o0_ph0a1o0
      = concatArrays(srcPre, new String[]{"  ph0a0o0 >= ph0a1o0;"}, srcPost);

    // no safety check: different heaps.
    nvlOpenarcCCAndRun("lt_ph0a0o0_ph1a0o0", workDir, "-fno-nvl-add-safety",
                       lt_ph0a0o0_ph1a0o0, stderrNoSafety, 0);
    nvlOpenarcCCAndRun("gt_ph0a0o0_ph1a0o0", workDir, "-fno-nvl-add-safety",
                       gt_ph0a0o0_ph1a0o0, stderrNoSafety, 0);
    nvlOpenarcCCAndRun("le_ph0a0o0_ph1a0o0", workDir, "-fno-nvl-add-safety",
                       le_ph0a0o0_ph1a0o0, stderrNoSafety, 0);
    nvlOpenarcCCAndRun("ge_ph0a0o0_ph1a0o0", workDir, "-fno-nvl-add-safety",
                       ge_ph0a0o0_ph1a0o0, stderrNoSafety, 0);

    // no safety check: different allocations.
    nvlOpenarcCCAndRun("lt_ph0a0o0_ph0a1o0", workDir, "-fno-nvl-add-safety",
                       lt_ph0a0o0_ph0a1o0, stderrNoSafety, 0);
    nvlOpenarcCCAndRun("gt_ph0a0o0_ph0a1o0", workDir, "-fno-nvl-add-safety",
                       gt_ph0a0o0_ph0a1o0, stderrNoSafety, 0);
    nvlOpenarcCCAndRun("le_ph0a0o0_ph0a1o0", workDir, "-fno-nvl-add-safety",
                       le_ph0a0o0_ph0a1o0, stderrNoSafety, 0);
    nvlOpenarcCCAndRun("ge_ph0a0o0_ph0a1o0", workDir, "-fno-nvl-add-safety",
                       ge_ph0a0o0_ph0a1o0, stderrNoSafety, 0);

    // safety check: different heaps.
    nvlOpenarcCCAndRun("s_lt_ph0a0o0_ph1a0o0", workDir, "",
                       lt_ph0a0o0_ph1a0o0, stderrSafety, 1);
    nvlOpenarcCCAndRun("s_gt_ph0a0o0_ph1a0o0", workDir, "",
                       gt_ph0a0o0_ph1a0o0, stderrSafety, 1);
    nvlOpenarcCCAndRun("s_le_ph0a0o0_ph1a0o0", workDir, "",
                       le_ph0a0o0_ph1a0o0, stderrSafety, 1);
    nvlOpenarcCCAndRun("s_ge_ph0a0o0_ph1a0o0", workDir, "",
                       ge_ph0a0o0_ph1a0o0, stderrSafety, 1);

    // safety check: different allocations.
    nvlOpenarcCCAndRun("s_lt_ph0a0o0_ph0a1o0", workDir, "",
                       lt_ph0a0o0_ph0a1o0, stderrSafety, 1);
    nvlOpenarcCCAndRun("s_gt_ph0a0o0_ph0a1o0", workDir, "",
                       gt_ph0a0o0_ph0a1o0, stderrSafety, 1);
    nvlOpenarcCCAndRun("s_le_ph0a0o0_ph0a1o0", workDir, "",
                       le_ph0a0o0_ph0a1o0, stderrSafety, 1);
    nvlOpenarcCCAndRun("s_ge_ph0a0o0_ph0a1o0", workDir, "",
                       ge_ph0a0o0_ph0a1o0, stderrSafety, 1);
  }

  // ---------------------------------------------
  // lower llvm.nvl.sub.v2nv and llvm.nvl.sub.v2wp

  @Test public void lower_nvl_sub_v2nv() throws Exception {
    lower_nvl_sub("nvl int");
  }
  @Test public void lower_nvl_sub_v2nv2nv() throws Exception {
    lower_nvl_sub("nvl int * nvl");
  }
  @Test public void lower_nvl_sub_v2wp() throws Exception {
    lower_nvl_sub("nvl int * nvl_wp");
  }
  private void lower_nvl_sub(String targetTy) throws Exception {
    final String ty = targetTy + "*";
    LLVMTargetData.initializeNativeTarget();
    final String heapFile0 = mkHeapFile("0.nvl").getAbsolutePath();
    final String heapFile1 = mkHeapFile("1.nvl").getAbsolutePath();
    final File workDir = mkTmpDir("");
    final SimpleResult simpleResult = buildLLVMSimple(
      targetTriple, targetDataLayout,
      new String[]{
        "#include <nvl.h>",
        "#include <stddef.h>",
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "ptrdiff_t c_sub_c0_c0 = ("+ty+")0 - ("+ty+")0;",
        "ptrdiff_t c_sub_c1_c0 = (("+ty+")0+1) - ("+ty+")0;",
        "ptrdiff_t get_c_sub_c1_c0() { return c_sub_c1_c0; }",
        "ptrdiff_t c_sub_c0_c1 = ("+ty+")0 - (("+ty+")0+1);",
        "ptrdiff_t get_c_sub_c0_c1() { return c_sub_c0_c1; }",
        "ptrdiff_t c_sub_c2_c1 = (("+ty+")0+2) - (("+ty+")0+1);",
        "ptrdiff_t get_c_sub_c2_c1() { return c_sub_c2_c1; }",

        "nvl_heap_t *heap0 = NULL;",
        "nvl_heap_t *heap1 = NULL;",
        ty+"ph0a0o0 = NULL;",
        ty+"ph0a0o1 = NULL;",
        ty+"ph0a1o0 = NULL;",
        ty+"ph1a0o0 = NULL;",
        "int setup() {",
        "  if (heap0) return 1;",
        "  heap0 = nvl_create(\""+heapFile0+"\", 0, 0600);",
        "  heap1 = nvl_create(\""+heapFile1+"\", 0, 0600);",
        "  if (!heap0 || !heap1) return 2;",
        "  ph0a0o0 = nvl_alloc_nv(heap0, 5, "+targetTy+");",
        "  ph0a0o1 = ph0a0o0 + 1;",
        "  ph0a1o0 = nvl_alloc_nv(heap0, 1, "+targetTy+");",
        "  ph1a0o0 = nvl_alloc_nv(heap1, 1, "+targetTy+");",
        "  return 0;",
        "}",

        "#define A ph0a0o0",
        "ptrdiff_t sub_c0_0() { "+ty+"p = 0; return ("+ty+")0 - p; }",
        "ptrdiff_t sub_0_c0() { "+ty+"p = 0; return p - ("+ty+")0; }",
        "ptrdiff_t sub_a0_a0() {"+ty+"p0 = 0; "+ty+"p1 = 0; return p0 - p1;}",
        "ptrdiff_t sub_a1_a0(){"+ty+"p0=("+ty+")A+1;"+ty+"p1=A;return p0-p1;}",
        "ptrdiff_t sub_ia0_a1(){"+ty+"p=("+ty+")A+1; return ("+ty+")A - p;}",
        "ptrdiff_t sub_a5_a3(){"+ty+"p0=("+ty+")A+5; "+ty+"p1=("+ty+")A+3;",
        "                      return p0 - p1;}",

        // libnvlrt_pmemobj normally supplies this, but we want this test to
        // succeed on systems (OS X) that don't have that.
        "void nvlrt_report_heapAlloc() {",
        "  fprintf(stderr, \"unexpected nvlrt_report_heapAlloc call\");",
        "  exit(99);",
        "}",

        "ptrdiff_t ph0a0o0_ph0a0o0() {return ph0a0o0 - ph0a0o0;}",
        "ptrdiff_t ph0a0o0_ph0a0o1() {return ph0a0o0 - ph0a0o1;}",
        "ptrdiff_t ph0a0o1_ph0a0o0() {return ph0a0o1 - ph0a0o0;}",

        "int cleanup() {",
        "  if (!heap0 || !heap1) return 1;",
        "  nvl_close(heap0);",
        "  nvl_close(heap1);",
        "  return 0;",
        "}",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    final LLVMModulePassManager pm = new LLVMModulePassManager();
    pm.addTargetData(exec.getTargetData());
    addNVLPasses(pm);
    pm.run(mod);
    mod.dump();

    checkIntGlobalVar(mod, "c_sub_c0_c0", 0);
    checkGlobalVarGetInit(mod, "c_sub_c1_c0");
    checkIntFn(exec, mod, "get_c_sub_c1_c0", 1);
    checkGlobalVarGetInit(mod, "c_sub_c0_c1");
    checkIntFn(exec, mod, "get_c_sub_c0_c1", -1);
    checkGlobalVarGetInit(mod, "c_sub_c2_c1");
    checkIntFn(exec, mod, "get_c_sub_c2_c1", 1);

    // The rest require at least nvlrt_inc_v.
    assumePmemobjLibs();
    checkIntFn(exec, mod, "setup", 0);

    checkIntFn(exec, mod, "sub_c0_0", 0);
    checkIntFn(exec, mod, "sub_0_c0", 0);
    checkIntFn(exec, mod, "sub_a0_a0", 0);
    checkIntFn(exec, mod, "sub_a1_a0", 1);
    checkIntFn(exec, mod, "sub_ia0_a1", -1);
    checkIntFn(exec, mod, "sub_a5_a3", 2);

    checkIntFn(exec, mod, "ph0a0o0_ph0a0o0", 0);
    checkIntFn(exec, mod, "ph0a0o0_ph0a0o1", -1);
    checkIntFn(exec, mod, "ph0a0o1_ph0a0o0", 1);
    checkIntFn(exec, mod, "cleanup", 0);

    exec.dispose();

    // The rest trap when safety checks are turned on, so we need to run
    // them in separate processes, or we'll halt the test suite.

    String[] srcPre = new String[]{
      "#include <nvl.h>",
      "#include <signal.h>",
      "#include <stddef.h>",
      "#include <stdio.h>",
      "#include <stdlib.h>",
      "struct sigaction sigill_act;",
      "void sigill_handler(int signum, siginfo_t *info, void*) {",
      "  fprintf(stderr, \"received SIGILL\\n\");",
      "  exit(1);",
      "}",
      "int main() {",
      "  sigill_act.sa_flags = SA_SIGINFO;",
      "  sigill_act.sa_sigaction = sigill_handler;",
      "  sigaction(SIGILL, &sigill_act, 0);",
      "  nvl_heap_t *heap0 = nvl_open(\""+heapFile0+"\");",
      "  nvl_heap_t *heap1 = nvl_open(\""+heapFile1+"\");",
      "  if (!heap0 || !heap1) return 1;",
      "  "+ty+"ph0a0o0 = nvl_alloc_nv(heap0, 1, "+targetTy+");",
      "  "+ty+"ph0a1o0 = nvl_alloc_nv(heap0, 1, "+targetTy+");",
      "  "+ty+"ph1a0o0 = nvl_alloc_nv(heap1, 1, "+targetTy+");",
      "  if (!ph0a0o0 || !ph0a1o0 || !ph1a0o0) return 1;",
      "  fprintf(stderr, \"before instruction\\n\");"};
    String[] srcPost = new String[]{
      "  fprintf(stderr, \"after instruction\\n\");",
      "  return 0;",
      "}"};
    String[] stderrNoSafety = new String[]{
      "before instruction",
      "after instruction",
    };
    String[] stderrSafety = new String[]{
      "before instruction",
      "nvlrt-pmemobj: error: illegal operation on pointers to different NVM"
      +" allocations",
      "received SIGILL",
    };
    String[] ph0a0o0_ph0a1o0
      = concatArrays(srcPre, new String[]{"  ph0a0o0 - ph0a1o0;"}, srcPost);
    String[] ph1a0o0_ph0a0o0
      = concatArrays(srcPre, new String[]{"  ph1a0o0 - ph0a0o0;"}, srcPost);

    nvlOpenarcCCAndRun("ph0a0o0_ph0a1o0", workDir, "-fno-nvl-add-safety",
                       ph0a0o0_ph0a1o0, stderrNoSafety, 0);
    nvlOpenarcCCAndRun("ph1a0o0_ph0a0o0", workDir, "-fno-nvl-add-safety",
                       ph1a0o0_ph0a0o0, stderrNoSafety, 0);

    nvlOpenarcCCAndRun("s_ph0a0o0_ph0a1o0", workDir, "",
                       ph0a0o0_ph0a1o0, stderrSafety, 1);
    nvlOpenarcCCAndRun("s_ph1a0o0_ph0a0o0", workDir, "",
                       ph1a0o0_ph0a0o0, stderrSafety, 1);
  }

  // ---------
  // lower gep
  //
  // gep is also exercised within icmp and sub tests in order to build
  // non-null NVM pointers without nvl_alloc_nv and to compute pointers
  // to different objects in the same allocation. icmp is also exercised
  // here in order to convert constant NVM pointers to _Bool to see if
  // they're null pointers.

  @Test public void lower_gep_v2nv() throws Exception {
    lower_gep("nvl int");
  }
  @Test public void lower_gep_v2nv2nv() throws Exception {
    lower_gep("nvl int * nvl");
  }
  @Test public void lower_gep_v2wp() throws Exception {
    lower_gep("nvl int * nvl_wp");
  }
  private void lower_gep(String targetTy) throws Exception {
    final String ty = targetTy + "*";
    LLVMTargetData.initializeNativeTarget();
    final String heapFile = mkHeapFile("test.nvl").getAbsolutePath();
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "#include <stddef.h>",
        ty+"c_add_cp0_0 = ("+ty+")0 + 0;",
        ty+"c_add_0_cp0 = 0 + ("+ty+")0;",
        ty+"c_sub_cp0_0 = ("+ty+")0 - 0;",
        ty+"c_sub_cp2_2 = (("+ty+")0+2) - 2;",
        "struct S { int f0; int f1; };",
        "nvl int *c_mem_cp0_f0 = &((nvl struct S *)0)->f0;",
        "nvl int *c_arr_cp0    = *(nvl int (*)[5])0;",
        "nvl int *c_arr_cp0_e0 = &(*(nvl int (*)[5])0)[0];",

        "_Bool get_c_add_cp0_0()  { return c_add_cp0_0;  }",
        "_Bool get_c_add_0_cp0()  { return c_add_0_cp0;  }",
        "_Bool get_c_sub_cp0_0()  { return c_sub_cp0_0;  }",
        "_Bool get_c_sub_cp2_2()  { return c_sub_cp2_2;  }",
        "_Bool get_c_mem_cp0_f0() { return c_mem_cp0_f0; }",
        "_Bool get_c_arr_cp0()    { return c_arr_cp0;    }",
        "_Bool get_c_arr_cp0_e0() { return c_arr_cp0_e0; }",

        "nvl_heap_t *heap = NULL;",
        ty+"pa0 = NULL;",
        ty+"pa1 = NULL;",
        "nvl struct S *ps = NULL;",
        "nvl int (*pa5i)[5] = NULL;",
        "int setup() {",
        "  if (heap) return 0;",
        "  heap = nvl_create(\""+heapFile+"\", 0, 0600);",
        "  if (!heap) return 1;",
        "  pa0 = nvl_alloc_nv(heap, 5, "+targetTy+");",
        "  if (!pa0) return 2;",
        "  pa1 = nvl_alloc_nv(heap, 1, "+targetTy+");",
        "  if (!pa1) return 3;",
        "  ps = nvl_alloc_nv(heap, 1, struct S);",
        "  if (!ps) return 4;",
        "  pa5i = nvl_alloc_nv(heap, 1, int[5]);",
        "  if (!pa5i) return 5;",
        "  return 0;",
        "}",

        "_Bool add_p0_0()  {"+ty+"p = 0; return p + 0;}",
        "_Bool add_pa_0()  {"+ty+"p = pa0; return p + 0;}",
        "_Bool add_0_pa()  {"+ty+"p = pa0; return 0 + p;}",
        "_Bool add_m3_pa3(){"+ty+"p = pa0+3; return -3 + p;}",
        "_Bool sub_p0_0()  {"+ty+"p = 0; return p - 0;}",
        "_Bool sub_pa_3()  {"+ty+"p = pa0; return p - 3;}",
        "_Bool sub_pa3_1() {"+ty+"p = pa0+3; return p - 1;}",
        "_Bool sub_pa3_3() {"+ty+"p = pa0+3; return p - 3;}",
        "_Bool mem_p0_f0() {nvl struct S *p = 0; return &p->f0;}",
        "_Bool mem_ps_f0() {nvl struct S *p = ps; return &p->f0;}",
        "_Bool mem_ps_f1() {nvl struct S *p = ps; return &p->f1;}",
        "_Bool arr_p0()    {nvl int (*p)[5] = 0; return *p;}",
        "_Bool arr_p0_e0() {nvl int (*p)[5] = 0; return &(*p)[0];}",
        "_Bool arr_pa()    {nvl int (*p)[5] = pa5i; return *p;}",
        "_Bool arr_pa_e0() {nvl int (*p)[5] = pa5i; return &(*p)[0];}",
        "_Bool arr_pa_e1() {nvl int (*p)[5] = pa5i; return &(*p)[1];}",

        "_Bool add_sub_2() {",
        "  "+ty+"p = pa0 + 2;",
        "  p -= 2;",
        "  return p == pa0;",
        "}",
        "_Bool add_vs_mem() {",
        "  nvl int *pf0 = &ps->f0;",
        "  nvl int *pf1 = &(*ps).f1;",
        "  nvl int *add = pf0 + 1;",
        "  return pf1 == add;",
        "}",
        "_Bool sub_vs_mem() {",
        "  nvl int *pf0 = &ps->f0;",
        "  nvl int *pf1 = &(*ps).f1;",
        "  nvl int *sub = pf1 - 1;",
        "  return pf0 == sub;",
        "}",
        "_Bool add_vs_arr() {",
        "  "+ty+"pe0 = &pa0[0];",
        "  "+ty+"pe1 = &pa0[1];",
        "  "+ty+"add = pe0 + 1;",
        "  return pe1 == add;",
        "}",
        "_Bool sub_vs_arr() {",
        "  "+ty+"pe0 = &pa0[0];",
        "  "+ty+"pe1 = &pa0[1];",
        "  "+ty+"sub = pe1 - 1;",
        "  return pe0 == sub;",
        "}",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    final LLVMModulePassManager pm = new LLVMModulePassManager();
    pm.addTargetData(exec.getTargetData());
    addNVLPasses(pm);
    pm.run(mod);
    mod.dump();
    checkGlobalVarGetInit(mod, "c_add_cp0_0");
    checkGlobalVarGetInit(mod, "c_add_0_cp0");
    checkGlobalVarGetInit(mod, "c_sub_cp0_0");
    checkGlobalVarGetInit(mod, "c_sub_cp2_2");
    checkGlobalVarGetInit(mod, "c_mem_cp0_f0");
    checkGlobalVarGetInit(mod, "c_arr_cp0");
    checkGlobalVarGetInit(mod, "c_arr_cp0_e0");

    // The rest require at least nvlrt_inc_v.
    assumePmemobjLibs();

    checkIntFn(exec, mod, "get_c_add_cp0_0",  0);
    checkIntFn(exec, mod, "get_c_add_0_cp0",  0);
    checkIntFn(exec, mod, "get_c_sub_cp0_0",  0);
    checkIntFn(exec, mod, "get_c_sub_cp2_2",  0);
    checkIntFn(exec, mod, "get_c_mem_cp0_f0", 0);
    checkIntFn(exec, mod, "get_c_arr_cp0",    0);
    checkIntFn(exec, mod, "get_c_arr_cp0_e0", 0);

    checkIntFn(exec, mod, "setup", 0);

    checkIntFn(exec, mod, "add_p0_0",   0);
    checkIntFn(exec, mod, "add_pa_0",   1);
    checkIntFn(exec, mod, "add_0_pa",   1);
    checkIntFn(exec, mod, "add_m3_pa3", 1);
    checkIntFn(exec, mod, "sub_p0_0",   0);
    checkIntFn(exec, mod, "sub_pa_3",   1);
    checkIntFn(exec, mod, "sub_pa3_1",  1);
    checkIntFn(exec, mod, "sub_pa3_3",  1);
    checkIntFn(exec, mod, "mem_p0_f0", 0);
    checkIntFn(exec, mod, "mem_ps_f0", 1);
    checkIntFn(exec, mod, "mem_ps_f1", 1);
    checkIntFn(exec, mod, "arr_p0",    0);
    checkIntFn(exec, mod, "arr_p0_e0", 0);
    checkIntFn(exec, mod, "arr_pa",    1);
    checkIntFn(exec, mod, "arr_pa_e0", 1);
    checkIntFn(exec, mod, "arr_pa_e1", 1);

    checkIntFn(exec, mod, "add_sub_2", 1);
    checkIntFn(exec, mod, "add_vs_mem", 1);
    checkIntFn(exec, mod, "sub_vs_mem", 1);
    checkIntFn(exec, mod, "add_vs_arr", 1);
    checkIntFn(exec, mod, "sub_vs_arr", 1);

    exec.dispose();
  }

  @Test public void lower_load_store() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final File workDir = mkTmpDir("");
    final File heapFile0 = mkHeapFile("0.nvl");
    final File heapFile1 = mkHeapFile("1.nvl");
    final String heapFile0Abs = heapFile0.getAbsolutePath();
    final String heapFile1Abs = heapFile1.getAbsolutePath();
    final SimpleResult simpleResult = buildLLVMSimple(
      targetTriple, targetDataLayout,
      new String[]{
        "#include <nvl.h>",
        "#include <nvlrt-test.h>",
        "#include <stddef.h>",
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "nvl_heap_t *heap = NULL;",
        "int setup() {",
        "  if (heap) return 1;",
        "  heap = nvl_create(\""+heapFile0Abs+"\", 0, 0600);",
        "  if (!heap) return 2;",
        "  return 0;",
        "}",
        // loads and stores here do not require bitcasts from ptr.obj's i8*
        // type because the desired pointer type is i8*.
        "char csum(int n) {",
        "  nvl char *sum = nvl_alloc_nv(heap, 1, char);",
        "  nvl char *arr = nvl_alloc_nv(heap, n, char);",
        "  for (int i = 0; i < n; ++i)",
        "    arr[i] = i+1;", // store
        "  for (int i = 0; i < n; ++i)",
        "    *sum += arr[i];", // load and store
        "  return *sum;", // load
        "}",
        // loads and stores here require bitcasts from ptr.obj's i8* type to
        // i32*.
        "int isum(int n) {",
        "  nvl int *sum = nvl_alloc_nv(heap, 1, int);",
        "  nvl int *arr = nvl_alloc_nv(heap, n, int);",
        "  for (int i = 0; i < n; ++i)",
        "    arr[i] = i+1;", // store
        "  for (int i = 0; i < n; ++i)",
        "    *sum += arr[i];", // load and store
        "  return *sum;", // load
        "}",
        "int ptrs(int i) {",
        "  nvl int *                  pi     = nvl_alloc_nv(heap, 1, int);",
        "  nvl int * nvl    *         ppi    = nvl_alloc_nv(heap, 1, nvl int *);",
        "  nvl int * nvl_wp *         pwpi   = nvl_alloc_nv(heap, 1, nvl int * nvl_wp);",
        "  nvl int * nvl    * nvl    *pppi   = nvl_alloc_nv(heap, 1, nvl int * nvl    *);",
        "  nvl int * nvl_wp * nvl    *ppwpi  = nvl_alloc_nv(heap, 1, nvl int * nvl_wp *);",
        "  nvl int * nvl    * nvl_wp *pwppi  = nvl_alloc_nv(heap, 1, nvl int * nvl    * nvl_wp);",
        "  nvl int * nvl_wp * nvl_wp *pwpwpi = nvl_alloc_nv(heap, 1, nvl int * nvl_wp * nvl_wp);",
        // One operand is not a pointer, so llvm.nvl.check.heap.* not needed.
        "  *pi = i;        if (i != *pi)       return 1;",
        // Both operands are pointers, so each of these exercises one of
        // llvm.nvl.check.heap.* to check for NV-to-NV pointers.
        "  *ppi = pi;      if (i != **ppi)     return 2;", // v2nv_v2nv
        "  *pwpi = pi;     if (i != **pwpi)    return 3;", // v2nv_v2wp
        "  *pppi = ppi;    if (i != ***pppi)   return 4;", // v2nv_v2nv
        "  *ppwpi = pwpi;  if (i != ***ppwpi)  return 5;", // v2wp_v2nv
        "  *pwppi = ppi;   if (i != ***pwppi)  return 6;", // v2nv_v2wp
        "  *pwpwpi = pwpi; if (i != ***pwpwpi) return 7;", // v2wp_v2wp
        // Make sure storing null doesn't fail the heap check because the
        // heap field is different. Use all the various combinations of
        // types as above.
        "  pi = 0;",
        "  *ppi = pi;",
        "  *pwpi = pi;",
        "  ppi = 0;",
        "  pwpi = 0;",
        "  *pppi = ppi;",
        "  *ppwpi = pwpi;",
        "  *pwppi = ppi;",
        "  *pwpwpi = pwpi;",
        // Make sure we didn't clobber i, invalidating all our checks.
        "  return i;",
        "}",
        "struct S {",
        // ptr in struct in array in struct
        "  struct { nvl int *p; } sa[6];",
        // ptr in array in struct
        "  nvl int *pa[6];",
        // double ptr
        "  nvl int * nvl *pp;",
        "};",
        "int heapFieldChange() {",
        "  nvl struct S *s = nvl_alloc_nv(heap, 1, struct S);",
        "  nvl_set_root(heap, s);",
        "  s->sa[0].p = nvl_alloc_nv(heap, 1, int);",
        "  s->sa[2].p = nvl_alloc_nv(heap, 1, int);",
        "  s->sa[4].p = nvl_alloc_nv(heap, 1, int);",
        "  *s->sa[0].p = 10;",
        "  *s->sa[2].p = 20;",
        "  *s->sa[4].p = 30;",
        "  s->pa[1] = nvl_alloc_nv(heap, 1, int);",
        "  s->pa[3] = nvl_alloc_nv(heap, 1, int);",
        "  s->pa[5] = nvl_alloc_nv(heap, 1, int);",
        "  *s->pa[1] = 40;",
        "  *s->pa[3] = 50;",
        "  *s->pa[5] = 60;",
        "  s->pp = nvl_alloc_nv(heap, 1, nvl int *);",
        "  *s->pp = nvl_alloc_nv(heap, 1, int);",
        "  **s->pp = 70;",
        // Keep reopening heap and allocating memory until the nvl_heap_t
        // object is allocated somewhere new. I've tried a few strategies
        // for getting the heap to move. Most strategies do not have
        // consistent behavior from one run to the next. The best strategy
        // so far seems to be to keep allocating sizeof(nvl_heap_t) bytes
        // until the heap moves. To avoid a very long hang in the test suite
        // (but I'm sure it'd run out of memory eventually), we also set a
        // cap on the number of repetitions. However, these days, just one
        // repetition seems to be sufficient.
        //
        // Also, clear the newly allocated memory so that hopefully any
        // dangling pointer to the existing nvl_heap_t won't point to a
        // valid copy.
        "  {",
        "    size_t sizeofNvlHeapT = nvlrt_get_sizeofNvlHeapT();",
        "    nvl_heap_t *oldHeap = heap;",
        "    fprintf(stderr, \"trying to move the nvl_heap_t:\\n\");",
        "    for (int i = 0; oldHeap == heap; ++i) {",
        "      if (i > 100000) {",
        "        fprintf(stderr, \"failed to move heap\\n\");",
        "        exit(1);",
        "      }",
        "      fprintf(stderr, \"  allocating sizeof(nvl_heap_t)=%zu\"",
        "                      \" bytes, repetition %d...\\n\",",
        "                      sizeofNvlHeapT, i);",
        "      nvl_close(heap);",
        "      calloc(1, sizeofNvlHeapT);",
        "      heap = nvl_open(\""+heapFile0Abs+"\");",
        "      if (!heap) {",
        "        fprintf(stderr, \"out of memory\\n\");",
        "        exit(1);",
        "      }",
        "    }",
        "  }",
        "  fprintf(stderr, \"moved heap successfully\\n\");",
        // Make sure nvl_get_root uses the new heap field for the root
        // pointer.
        "  struct S sLoad = *nvl_get_root(heap, struct S);",
        // Make sure this struct load sets the heap fields of all contained
        // non-null pointers or else dereferencing them will fail. For null
        // pointers, make sure it leaves the heap field as null.
        "  if (*sLoad.sa[0].p != 10) return 1;",
        "  if (sLoad.sa[1].p != 0) return 2;",
        "  if (*sLoad.sa[2].p != 20) return 3;",
        "  if (sLoad.sa[3].p != 0) return 4;",
        "  if (*sLoad.sa[4].p != 30) return 5;",
        "  if (sLoad.sa[5].p != 0) return 6;",
        "  if (sLoad.pa[0] != 0) return 7;",
        "  if (*sLoad.pa[1] != 40) return 8;",
        "  if (sLoad.pa[2] != 0) return 9;",
        "  if (*sLoad.pa[3] != 50) return 10;",
        "  if (sLoad.pa[4] != 0) return 11;",
        "  if (*sLoad.pa[5] != 60) return 12;",
        "  if (**sLoad.pp != 70) return 13;",
        "  return 0;",
        "}",

        "int cleanup() {",
        "  nvl_close(heap);",
        "  return 0;",
        "}",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    final LLVMModulePassManager pm = new LLVMModulePassManager();
    pm.addTargetData(exec.getTargetData());
    addNVLPasses(pm);
    pm.run(mod);
    mod.dump();
    // Running requires at least nvlrt_alloc_nv.
    assumePmemobjLibs();
    checkIntFn(exec, mod, "setup", 0);
    checkIntFn(exec, mod, "csum", getIntGeneric(3, ctxt), 6);
    checkIntFn(exec, mod, "csum", getIntGeneric(12, ctxt), 78);
    checkIntFn(exec, mod, "isum", getIntGeneric(5, ctxt), 15);
    checkIntFn(exec, mod, "isum", getIntGeneric(10, ctxt), 55);
    checkIntFn(exec, mod, "ptrs", getIntGeneric(8, ctxt), 8);
    checkIntFn(exec, mod, "ptrs", getIntGeneric(107, ctxt), 107);
    checkIntFn(exec, mod, "heapFieldChange", 0);
    checkIntFn(exec, mod, "heapFieldChange", 0);
    checkIntFn(exec, mod, "cleanup", 0);
    exec.dispose();

    // The rest trap when safety checks are turned on, so we need to run
    // them in separate processes, or we'll halt the test suite.

    String[] srcPre = new String[]{
      "#include <nvl.h>",
      "#include <signal.h>",
      "#include <stddef.h>",
      "#include <stdio.h>",
      "#include <stdlib.h>",
      "struct sigaction sigill_act;",
      "void sigill_handler(int signum, siginfo_t *info, void*) {",
      "  fprintf(stderr, \"received SIGILL\\n\");",
      "  exit(1);",
      "}",
    };
    String[] srcPost = new String[]{
      "int main() {",
      "  sigill_act.sa_flags = SA_SIGINFO;",
      "  sigill_act.sa_sigaction = sigill_handler;",
      "  sigaction(SIGILL, &sigill_act, 0);",
      "  nvl_heap_t *heap0 = nvl_open(\""+heapFile0Abs+"\");",
      "  if (!heap0)",
      "    heap0 = nvl_create(\""+heapFile0Abs+"\", 0, 0600);",
      "  if (!heap0) return 1;",
      "  nvl_heap_t *heap1 = nvl_open(\""+heapFile1Abs+"\");",
      "  if (!heap1)",
      "    heap1 = nvl_create(\""+heapFile1Abs+"\", 0, 0600);",
      "  if (!heap1) return 2;",
      "  T *ph0 = nvl_alloc_nv(heap0, 1, T);",
      "  AggT *ph1 = nvl_alloc_nv(heap1, 1, AggT);",
      "  if (!ph0 || !ph1) return 3;",
      "  fprintf(stderr, \"before instruction\\n\");",
      "#ifndef STORE",
      "  *ph1 = ph0;",
      "#else",
      "  STORE(ph0, *ph1)",
      "#endif",
      "  fprintf(stderr, \"after instruction\\n\");",
      "  return 0;",
      "}",
    };
    String[] stderrNoSafety = new String[]{
      "before instruction",
      "after instruction",
    };
    String[] stderrSafety = new String[]{
      "before instruction",
      "nvlrt-pmemobj: error: creation of interheap NV-to-NV pointer",
      "received SIGILL",
    };
    // Each of these exercises one of llvm.nvl.check.heap.* to check for
    // NV-to-NV pointers.
    String[] pi = concatArrays(
      srcPre,
      new String[]{"typedef nvl int T; typedef T * nvl AggT;"}, // v2nv_v2nv
      srcPost);
    String[] wpi = concatArrays(
      srcPre,
      new String[]{"typedef nvl int T; typedef T * nvl_wp AggT;"}, // v2nv_v2wp
      srcPost);
    String[] ppi = concatArrays(
      srcPre,
      new String[]{"typedef nvl int * nvl T; typedef T * nvl AggT;"}, // v2nv_v2nv
      srcPost);
    String[] pwpi = concatArrays(
      srcPre,
      new String[]{"typedef nvl int * nvl_wp T; typedef T * nvl AggT;"}, // v2wp_v2nv
      srcPost);
    String[] wppi = concatArrays(
      srcPre,
      new String[]{"typedef nvl int * nvl T; typedef T * nvl_wp AggT;"}, // v2nv_v2wp
      srcPost);
    String[] wpwpi = concatArrays(
      srcPre,
      new String[]{"typedef nvl int * nvl_wp T; typedef T * nvl_wp AggT;"}, // v2wp_v2wp
      srcPost);
    String[] agg = concatArrays(
      srcPre,
      new String[]{
        "typedef nvl int T;",
        // ptr in struct in array in struct in struct
        "typedef struct { struct { struct { T *p; } a[2]; } s; } AggTV;",
        "typedef nvl AggTV AggT;",
        "AggTV agg = {{{{0}, {0}}}};",
        "#define STORE(Val, Agg) agg.s.a[1].p = (Val); (Agg) = agg;",
      },
      srcPost);

    // By disabling reference counting below, we're changing the layout of
    // the heaps (different pmemobj layout string), so we must delete the
    // old ones.
    heapFile0.delete();
    heapFile1.delete();

    // By disabling safety, we're permitting corruption of heap1, so we have
    // to delete it before using it again.
    // pmemobj only allows one heap per transaction, but we're writing a
    // pointer into one heap and incrementing its reference count in another
    // heap, so pmemobj would complain if we didn't disable transactions.
    // If we didn't disable automatic reference counting, then, at the end
    // of the program, automatic reference counting would try to free the
    // NVM allocation containing the NV-to-NV pointer, and then it would try
    // to dec the NV-to-NV reference count of the NV-to-NV pointer's target
    // allocation, but it would assume the NV-to-NV pointer is intra-heap
    // even though it is actually inter-heap, and so it would compute an
    // invalid address for the allocation.
    nvlOpenarcCCAndRun("pi", workDir, "-fno-nvl-add-safety", false,
                       pi, stderrNoSafety, 0);
    heapFile1.delete();
    nvlOpenarcCCAndRun("wpi", workDir, "-fno-nvl-add-safety", false,
                       wpi, stderrNoSafety, 0);
    heapFile1.delete();
    nvlOpenarcCCAndRun("ppi", workDir, "-fno-nvl-add-safety", false,
                       ppi, stderrNoSafety, 0);
    heapFile1.delete();
    nvlOpenarcCCAndRun("pwpi", workDir, "-fno-nvl-add-safety", false,
                       pwpi, stderrNoSafety, 0);
    heapFile1.delete();
    nvlOpenarcCCAndRun("wppi", workDir, "-fno-nvl-add-safety", false,
                       wppi, stderrNoSafety, 0);
    heapFile1.delete();
    nvlOpenarcCCAndRun("wpwpi", workDir, "-fno-nvl-add-safety", false,
                       wpwpi, stderrNoSafety, 0);
    heapFile1.delete();
    nvlOpenarcCCAndRun("agg", workDir, "-fno-nvl-add-safety", false,
                       agg, stderrNoSafety, 0);

    heapFile0.delete();
    heapFile1.delete();
    nvlOpenarcCCAndRun("s_pi", workDir, "",
                       pi, stderrSafety, 1);
    nvlOpenarcCCAndRun("s_wpi", workDir, "",
                       wpi, stderrSafety, 1);
    nvlOpenarcCCAndRun("s_ppi", workDir, "",
                       ppi, stderrSafety, 1);
    nvlOpenarcCCAndRun("s_pwpi", workDir, "",
                       pwpi, stderrSafety, 1);
    nvlOpenarcCCAndRun("s_wppi", workDir, "",
                       wppi, stderrSafety, 1);
    nvlOpenarcCCAndRun("s_wpwpi", workDir, "",
                       wpwpi, stderrSafety, 1);
    nvlOpenarcCCAndRun("s_agg", workDir, "",
                       agg, stderrSafety, 1);
  }

  // -----------------------------
  // nvl_set_root and nvl_get_root

  @Test public void nvl_get_root_fileScope() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "call to __builtin_nvl_get_root at file scope"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "nvl_heap_t *heap;",
        "nvl int *p = nvl_get_root(heap, int);",
      });
  }

  @Test public void nvl_get_root_rootPtrToVoid() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "void type is NVM-stored"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  nvl_heap_t *heap;",
        "  nvl_get_root(heap, void);",
        "}",
      });
  }

  @Test public void nvl_get_root_rootPtrToIncompleteStruct()
    throws IOException
  {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored struct type is incomplete in translation unit"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  nvl_heap_t *heap;",
        "  nvl_get_root(heap, struct S);",
        "}",
      });
  }

  @Test public void nvl_get_root_rootPtrToUnion() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "union type is NVM-stored"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "union U {int i; float f;};",
        "void fn() {",
        "  nvl_heap_t *heap;",
        "  nvl_get_root(heap, union U);",
        "}",
      });
  }

  @Test public void nvl_get_root_rootPtrToNvlWp() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "type has both nvl and nvl_wp type qualifiers"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  nvl_heap_t *heap;",
        "  nvl_get_root(heap, int nvl * nvl_wp);",
        "}",
      });
  }

  @Test public void nvl_get_root_assignToNonNvmPtr() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "warning treated as error: initialization discards type qualifiers"
      +" from pointer target type: nvl"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  nvl_heap_t *heap;",
        "  int *p = nvl_get_root(heap, int);",
        "}",
      });
  }

  @Test public void nvl_get_root_explicitTypeChecksum() throws IOException {
    // In this case, the parser is going to fail because
    // __builtin_nvl_get_root requires a special grammar rule. We don't
    // bother to check the error message because it's complicated, but
    // hopefully it's the right one.
    exit.expectSystemExitWithStatus(1);
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "const char *checksum = \"0123456789012345\";",
        "void fn() {",
        "  nvl_heap_t *heap;",
        // We're not interested in exercising preprocessor argument
        // checking, so we use the built-in function directly.
        "  int *p = __builtin_nvl_get_root(heap, int, checksum);",
        "}",
      });
  }

  @Test public void nvl_set_root_rootNotPtr() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "argument 2 to __builtin_nvl_set_root is not a null pointer constant"
      +" or a pointer to an nvl-qualified type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  nvl_heap_t *heap;",
        "  nvl_set_root(heap, 5);",
        "}",
      });
  }

  @Test public void nvl_set_root_rootPtrToNonNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "argument 2 to __builtin_nvl_set_root is not a null pointer constant"
      +" or a pointer to an nvl-qualified type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  nvl_heap_t *heap;",
        "  int *root = 0;",
        "  nvl_set_root(heap, root);",
        "}",
      });
  }

  @Test public void nvl_set_root_rootPtrToNvlWp() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "argument 2 to __builtin_nvl_set_root is not a null pointer constant"
      +" or a pointer to an nvl-qualified type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  nvl_heap_t *heap;",
        "  nvl int * nvl_wp *root = 0;",
        "  nvl_set_root(heap, root);",
        "}",
      });
  }

  @Test public void nvl_set_root_explicitTypeChecksum() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "too many arguments in call to __builtin_nvl_set_root"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "const char *checksum = \"0123456789012345\";",
        "void fn() {",
        "  nvl_heap_t *heap;",
        "  nvl int *root = 0;",
        // We're not interested in exercising preprocessor argument
        // checking, so we use the built-in function directly.
        "  __builtin_nvl_set_root(heap, root, checksum);",
        "}",
      });
  }

  @Test public void nvl_set_root_return() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "initialization requires conversion from <void> to <_Bool>"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "#include <stdbool.h>",
        "void fn() {",
        "  nvl_heap_t *heap;",
        "  nvl int *root = 0;",
        // Make sure return type of llvm.nvl.set_root isn't exposed as
        // nvl_set_root's return type, which should be void.
        "  bool b = nvl_set_root(heap, root);",
        "}",
      });
  }

  @Test public void nvl_get_set_root_llvmDecl() throws IOException {
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "typedef int T[3];",
        "void fn() {",
        "  nvl_heap_t *heap;",
        "  nvl int *root1 = 0;",
        "  nvl int (*root2)[3] = 0;",
        "  nvl_set_root(heap, root1);",
        "  nvl_set_root(heap, root2);",
        "  nvl_get_root(heap, int);",
        "  nvl_get_root(heap, nvl int);",
        "  nvl_get_root(heap, nvl int *);",
        "  nvl_get_root(heap, nvl int * nvl);",
        "  nvl_get_root(heap, int[3]);",
        "  nvl_get_root(heap, nvl int[3]);",
        "  nvl_get_root(heap, T);",
        "  nvl_get_root(heap, nvl T);",
        "  nvl_get_root(heap, nvl int(*)[3]);",
        "  nvl_get_root(heap, nvl int(* nvl)[3]);",
        "}",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMPointerType voidPtrType
      = LLVMPointerType.get(SrcVoidType.getLLVMTypeAsPointerTarget(ctxt),
                            SrcPointerType.LLVM_ADDRSPACE_DEFAULT);
    final LLVMPointerType nvlVoidPtrType
      = LLVMPointerType.get(SrcVoidType.getLLVMTypeAsPointerTarget(ctxt),
                            SrcPointerType.LLVM_ADDRSPACE_NVL);
    final LLVMPointerType charPtrType
      = LLVMPointerType.get(SrcCharType.getLLVMTypeAsPointerTarget(ctxt),
                            SrcPointerType.LLVM_ADDRSPACE_DEFAULT);
    String getName = "llvm.nvl.get_root";
    String setName = "llvm.nvl.set_root";
    final LLVMFunction getFn = simpleResult.llvmModule
                               .getNamedFunction(getName);
    final LLVMFunction setFn = simpleResult.llvmModule
                               .getNamedFunction(setName);
    assertNotNull(getName+" must exist", getFn.getInstance());
    assertNotNull(setName+" must exist", setFn.getInstance());
    assertEquals(getName + "'s parameter count",
                 2, getFn.countParameters());
    assertEquals(setName + "'s parameter count",
                 3, setFn.countParameters());
    assertEquals(getName + " return type",
                 nvlVoidPtrType, getFn.getFunctionType().getReturnType());
    assertEquals(setName + " return type",
                 LLVMVoidType.get(ctxt),
                 setFn.getFunctionType().getReturnType());
    assertEquals(getName + " first parameter type",
                 voidPtrType, getFn.getFunctionType().getParamTypes()[0]);
    assertEquals(setName + " first parameter type",
                 voidPtrType, setFn.getFunctionType().getParamTypes()[0]);
    assertEquals(setName + " second parameter type",
                 nvlVoidPtrType,
                 setFn.getFunctionType().getParamTypes()[1]);
    assertEquals(getName + " second parameter type",
                 charPtrType,
                 getFn.getFunctionType().getParamTypes()[1]);
    assertEquals(setName + " third parameter type",
                 charPtrType,
                 setFn.getFunctionType().getParamTypes()[2]);
    assertEquals(getName + "'s variadicity",
                 false, getFn.getFunctionType().isVarArg());
    assertEquals(setName + "'s variadicity",
                 false, setFn.getFunctionType().isVarArg());
    assertEquals(getName + " definition's existence",
                 false, !getFn.isDeclaration());
    assertEquals(setName + " definition's existence",
                 false, !setFn.isDeclaration());
  }

  @Test public void nvl_get_set_root() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final File workDir = mkTmpDir("");
    final File heapFile0 = mkHeapFile("0.nvl");
    final File heapFile1 = mkHeapFile("1.nvl");
    final String heapFile0Abs = heapFile0.getAbsolutePath();
    final String heapFile1Abs = heapFile1.getAbsolutePath();
    final SimpleResult simpleResult = buildLLVMSimple(
      targetTriple, targetDataLayout,
      new String[]{
        "#include <nvl.h>",
        "#include <stddef.h>",
        // Make sure we don't get stuck in an infinite recursion while
        // trying to build the checksum for a recursive struct definition.
        "struct S {int i; double d; nvl struct S *p;};",

        "int setup() {",
        "  nvl_heap_t *heap = nvl_create(\""+heapFile0Abs+"\", 0, 0600);",
        "  if (!heap) return 1;",
        "  nvl_close(heap);",
        "  return 0;",
        "}",

        "int nullPtrs() {",
        "  nvl_heap_t *heap = nvl_open(\""+heapFile0Abs+"\");",
        "  if (!heap) return 1;",
        // root should be null initially (no previous type), so get with a
        // few different types
        "  nvl char *pc = nvl_get_root(heap, char);",
        "  if (pc) return 2;",
        "  nvl int *pi = nvl_get_root(heap, int);",
        "  if (pi) return 3;",
        // set with integer null pointer constant (no explicit pointer type,
        // but implicitly void*), and get with a few different types
        "  nvl_set_root(heap, 0);",
        "  pc = nvl_get_root(heap, char);",
        "  if (pc) return 4;",
        "  pi = nvl_get_root(heap, int);",
        "  if (pi) return 5;",
        // again but with void* null pointer constant
        "  nvl_set_root(heap, (void*)0);",
        "  pc = nvl_get_root(heap, char);",
        "  if (pc) return 6;",
        "  pi = nvl_get_root(heap, int);",
        "  if (pi) return 7;",
        // again but with stddef null pointer constant
        "  nvl_set_root(heap, NULL);",
        "  pc = nvl_get_root(heap, char);",
        "  if (pc) return 8;",
        "  pi = nvl_get_root(heap, int);",
        "  if (pi) return 9;",
        // again but with dynamic null pointers (type is always explicit in
        // this case)
        "  nvl_set_root(heap, pc);",
        "  pi = nvl_get_root(heap, int);",
        "  if (pi) return 10;",
        "  nvl_set_root(heap, pi);",
        "  pc = nvl_get_root(heap, char);",
        "  if (pc) return 11;",
        "  nvl_close(heap);",
        "  return 0;",
        "}",

        "typedef int I;",
        "int nonNullPtrs() {",
        "  nvl_heap_t *heap = nvl_open(\""+heapFile0Abs+"\");",
        "  if (!heap) return 1;",
        "  if (nvl_get_root(heap, int)) return 2;",
        "  nvl int *root = nvl_alloc_nv(heap, 2, int);",
        "  if (!root) return 3;",
        "  root[0] = 98;",
        "  root[1] = 99;",
        "  nvl_set_root(heap, root);",
        "  if (root != nvl_get_root(heap, int)) return 4;",
        "  nvl_close(heap);",
        "  heap = nvl_open(\""+heapFile0Abs+"\");",
        "  if (*nvl_get_root(heap, int) != 98) return 5;",
        // Make sure grammar allows an nvl_get_root call to appear as the
        // start of a postfix-expression, which ends in [] in this case.
        // The Cetus grammar doesn't quite follow the ISO C99 grammar in
        // this area.
        "  if (nvl_get_root(heap, int)[1] != 99) return 6;",
        // Make sure nvl-qualification doesn't hurt.
        "  if (nvl_get_root(heap, nvl int)[1] != 99) return 7;",
        // Make sure typedef to same type is fine.
        "  if (nvl_get_root(heap, I)[1] != 99) return 7;",
        "  nvl_close(heap);",
        "  return 0;",
        "}",

        "int validTypeChange() {",
        "  nvl_heap_t *heap = nvl_open(\""+heapFile0Abs+"\");",
        "  nvl int *pi = nvl_alloc_nv(heap, 2, int);",
        "  nvl char *pc = nvl_alloc_nv(heap, 2, char);",
        "  pi[0] = 233;",
        "  pi[1] = 9;",
        "  pc[0] = 3;",
        "  pc[1] = 8;",
        // set root as int*
        "  nvl_set_root(heap, pi);",
        "  if (*nvl_get_root(heap, int) != 233) return 1;",
        "  nvl_set_root(heap, pi+1);",
        "  if (*nvl_get_root(heap, int) != 9) return 2;",
        // set root as null pointer
        "  nvl_set_root(heap, 0);",
        "  if (nvl_get_root(heap, int) != 0) return 3;",
        "  if (nvl_get_root(heap, double) != 0) return 4;",
        // set root as char*
        "  nvl_set_root(heap, pc);",
        "  if (*nvl_get_root(heap, char) != 3) return 5;",
        "  nvl_set_root(heap, pc+1);",
        "  if (*nvl_get_root(heap, char) != 8) return 6;",
        // set root as int* again
        "  nvl_set_root(heap, pi);", // type change
        "  if (*nvl_get_root(heap, int) != 233) return 9;",
        "  nvl_set_root(heap, pi+1);",
        "  if (*nvl_get_root(heap, int) != 9) return 10;",
        "  nvl_close(heap);",
        "  return 0;",
        "}",

        // This also exercises a type change after reopening.
        "int recursiveStruct() {",
        "  nvl_heap_t *heap = nvl_open(\""+heapFile0Abs+"\");",
        "  if (!heap) return 1;",
        "  nvl struct S *root = nvl_alloc_nv(heap, 1, struct S);",
        "  if (!root) return 2;",
        "  root->i = 5;",
        "  root->d = 9.8;",
        "  root->p = root;",
        "  nvl_set_root(heap, root);",
        "  if (root != nvl_get_root(heap, struct S)) return 3;",
        "  nvl_close(heap);",
        "  heap = nvl_open(\""+heapFile0Abs+"\");",
        "  root = nvl_get_root(heap, struct S);",
        "  if (nvl_get_root(heap, struct S)->i != 5) return 4;",
        "  if (nvl_get_root(heap, nvl struct S)[0].d != 9.8) return 5;",
        "  if ((*nvl_get_root(heap, struct S nvl)).p != root) return 6;",
        "  nvl_close(heap);",
        "  return 0;",
        "}",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    final LLVMModulePassManager pm = new LLVMModulePassManager();
    pm.addTargetData(exec.getTargetData());
    addNVLPasses(pm);
    pm.run(mod);
    mod.dump();
    // Running requires at least nvlrt_alloc_nv.
    assumePmemobjLibs();
    checkIntFn(exec, mod, "setup", 0);
    checkIntFn(exec, mod, "nullPtrs", 0);
    checkIntFn(exec, mod, "nonNullPtrs", 0);
    checkIntFn(exec, mod, "validTypeChange", 0);
    checkIntFn(exec, mod, "recursiveStruct", 0);
    exec.dispose();

    // The rest call exit, so we need to run them in separate processes, or
    // we'll halt the test suite.

    String[] stderr = new String[]{
      "nvlrt-pmemobj: error: root type checksum mismatch",
    };
    nvlOpenarcCCAndRun(
      "signed_unsigned", workDir, "",
      gen_nvl_set_get_root(heapFile0Abs, "signed", "unsigned"),
      stderr, 1);
    nvlOpenarcCCAndRun(
      "structTagChange", workDir, "",
      gen_nvl_set_get_root(
        heapFile0Abs, "struct S1", "struct S2",
        "struct S1 {int x; int y;};",
        "struct S2 {int x; int y;};"),
      stderr, 1);

    // In these checks, we put the struct bodies at the end of the
    // translation units to be sure the bodies are not omitted from the type
    // checksums when the bodies are defined after the point in the source
    // where the checksums are used.
    nvlOpenarcCCAndRun(
      "setupStructChangeChecks", workDir, "",
      gen_nvl_set_root(
        heapFile0Abs, "struct S",
        new String[]{"struct S;"},
        new String[]{"struct S {int x; int y;};"}),
      new String[0], 0);
    nvlOpenarcCCAndRun(
      "getRootFieldNameChange", workDir, "",
      gen_nvl_get_root(
        heapFile0Abs, "struct S",
        new String[]{"struct S;"},
        new String[]{"struct S {int y; int x;};"}),
      stderr, 1);
    nvlOpenarcCCAndRun(
      "getRootExtraField", workDir, "",
      gen_nvl_get_root(
        heapFile0Abs, "struct S",
        new String[]{"struct S;"},
        new String[]{"struct S {int x; int y; int z;};"}),
      stderr, 1);
    nvlOpenarcCCAndRun(
      "getRootFieldTypeChange", workDir, "",
      gen_nvl_get_root(
        heapFile0Abs, "struct S",
        new String[]{"struct S;"},
        new String[]{"struct S {int x; float y;};"}),
      stderr, 1);
    nvlOpenarcCCAndRun(
      "getRootLostField", workDir, "",
      gen_nvl_get_root(
        heapFile0Abs, "struct S",
        new String[]{"struct S;"},
        new String[]{"struct S {int x;};"}),
      stderr, 1);
    // TODO: There are many type checksums that could be exercised here,
    // both as positive tests and negative tests.

    nvlOpenarcCCAndRun(
      "interheapNV2NV", workDir, "",
      new String[]{
        "#include <nvl.h>",
        "int main() {",
        "  nvl_heap_t *heap0 = nvl_open(\""+heapFile0Abs+"\");",
        "  nvl_heap_t *heap1 = nvl_create(\""+heapFile1Abs+"\", 0, 0600);",
        "  nvl int *root = nvl_alloc_nv(heap0, 1, int);",
        "  nvl_set_root(heap1, root);",
        "  return 0;",
        "}",
      },
      new String[]{
        "nvlrt-pmemobj: error: creation of interheap NV-to-NV pointer",
      }, 1);
  }

  private String[] gen_nvl_set_get_root(String heapFile, String setTy,
                                        String getTy, String... pre)
  {
    return concatArrays(
      new String[]{
        "#include <nvl.h>",
        "#include <stddef.h>",
        "#include <stdio.h>",
      },
      pre,
      new String[]{
        "int main() {",
        "  nvl_heap_t *heap = nvl_open(\""+heapFile+"\");",
        "  nvl "+setTy+" *p = nvl_alloc_nv(heap, 1, "+setTy+");",
        "  nvl_set_root(heap, p);",
        "  nvl_get_root(heap, "+getTy+");",
        "  nvl_close(heap);",
        "  return 0;",
        "}"
      });
  }

  private String[] gen_nvl_set_root(String heapFile, String setTy,
                                    String[] pre, String[] post)
  {
    return concatArrays(
      new String[]{
        "#include <nvl.h>",
        "#include <stddef.h>",
        "#include <stdio.h>",
        "nvl "+setTy+" *allocate();",
      },
      pre,
      new String[]{
        "nvl_heap_t *heap;",
        "int main() {",
        "  heap = nvl_open(\""+heapFile+"\");",
        "  nvl_set_root(heap, allocate());",
        "  nvl_close(heap);",
        "  return 0;",
        "}"
      },
      post,
      new String[]{
        // post might complete the setTy struct definition, so we cannot
        // call nvl_alloc_nv (and thus sizeof) for setTy until after it.
        "nvl "+setTy+" *allocate() {",
        "  return nvl_alloc_nv(heap, 1, "+setTy+");",
        "}",
      });
  }

  private String[] gen_nvl_get_root(String heapFile, String getTy,
                                    String[] pre, String[] post)
  {
    return concatArrays(
      new String[]{
        "#include <nvl.h>",
        "#include <stddef.h>",
        "#include <stdio.h>",
      },
      pre,
      new String[]{
        "int main() {",
        "  nvl_heap_t *heap = nvl_open(\""+heapFile+"\");",
        "  nvl_get_root(heap, "+getTy+");",
        "  nvl_close(heap);",
        "  return 0;",
        "}",
      },
      post);
  }

  // ------------
  // nvl_alloc_nv

  @Test public void nvl_alloc_nv_fileScope() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "call to __builtin_nvl_alloc_nv at file scope"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "nvl_heap_t *heap;",
        "nvl int *p = nvl_alloc_nv(heap, 1, int);",
      });
  }

  @Test public void nvl_alloc_nv_void() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "argument 3 to __builtin_nvl_alloc_nv is of incomplete type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  nvl_heap_t *heap;",
        "  nvl_alloc_nv(heap, 1, void);",
        "}",
      });
  }

  @Test public void nvl_alloc_nv_incompleteStruct() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "argument 3 to __builtin_nvl_alloc_nv is of incomplete type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "struct S;",
        "void fn() {",
        "  nvl_heap_t *heap;",
        // As when an argument to sizeof, struct S must be complete already.
        "  nvl_alloc_nv(heap, 1, struct S);",
        "}",
        "struct S {int i;};",
      });
  }

  @Test public void nvl_alloc_nv_fn() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "argument 3 to __builtin_nvl_alloc_nv is of function type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  nvl_heap_t *heap;",
        "  nvl_alloc_nv(heap, 1, void());",
        "}",
      });
  }

  @Test public void nvl_alloc_nv_union() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "union type is NVM-stored"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "union U {int i; float f;};",
        "void fn() {",
        "  nvl_heap_t *heap;",
        "  nvl_alloc_nv(heap, 1, union U);",
        "}",
      });
  }

  @Test public void nvl_alloc_nv_nvlPtrToNonNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "union U {int i; float f;};",
        "void fn() {",
        "  nvl_heap_t *heap;",
        "  nvl_alloc_nv(heap, 1, int *);",
        "}",
      });
  }

  @Test public void nvl_alloc_nv_nvlWpPtrToNonNVM() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "NVM-stored pointer type has non-NVM-stored target type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "union U {int i; float f;};",
        "void fn() {",
        "  nvl_heap_t *heap;",
        "  nvl_alloc_nv(heap, 1, int *nvl_wp);",
        "}",
      });
  }

  @Test public void nvl_alloc_nv_assignDiscardsNvl() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "warning treated as error: initialization discards type qualifiers"
      +" from pointer target type: nvl"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  nvl_heap_t *heap;",
        "  int *p = nvl_alloc_nv(heap, 1, int);",
        "}",
      });
  }

  @Test public void nvl_alloc_nv_assignDiscardsNvlWp() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "warning treated as error: initialization discards type qualifiers"
      +" from pointer target type: nvl_wp"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "void fn() {",
        "  nvl_heap_t *heap;",
        "  nvl int * nvl *p = nvl_alloc_nv(heap, 1, nvl int * nvl_wp);",
        "}",
      });
  }

  /// There are many successful nvl_alloc_nv calls in previous NVL-C test
  /// groups, so we exercise just a few special cases here.
  @Test public void nvl_alloc_nv() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final String heapFile = mkHeapFile("test.nvl").getAbsolutePath();
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "#include <stddef.h>",
        "typedef int I;",
        "typedef nvl int NI;",
        "int fn() {",
        "  nvl_heap_t *heap = nvl_create(\""+heapFile+"\", 0, 0600);",
        "  if (!heap) return 1;",
        "  if (!nvl_alloc_nv(heap, 1, int)) return 2;",
        // Make sure grammar allows an nvl_alloc_nv call to appear as the
        // start of a postfix-expression, which ends in [] in this case.
        // The Cetus grammar doesn't quite follow the ISO C99 grammar in
        // this area. At the same time, check for zero init.
        "  if (nvl_alloc_nv(heap, 1, int)[0]) return 3;",
        // Again, but make sure nvl-qualification doesn't hurt.
        "  if (*nvl_alloc_nv(heap, 1, nvl int)) return 4;",
        // Again, but make sure typedef to same type is fine.
        "  if (nvl_alloc_nv(heap, 2, I)[0]) return 5;",
        "  if (nvl_alloc_nv(heap, 2, nvl I)[1]) return 6;",
        "  if (nvl_alloc_nv(heap, 3, NI)[2]) return 7;",
        // Check zero init for pointers.
        "  if (*nvl_alloc_nv(heap, 1, nvl int *)) return 8;",
        "  if (*nvl_alloc_nv(heap, 1, nvl int * nvl)) return 9;",
        "  if (*nvl_alloc_nv(heap, 1, nvl int * nvl_wp)) return 10;",
        // Check return types.
        "  nvl int *pi = nvl_alloc_nv(heap, 1, int);",
        "  nvl int * nvl *psp = nvl_alloc_nv(heap, 1, nvl int *);",
        "  nvl int * nvl_wp *pwp = nvl_alloc_nv(heap, 1, nvl int * nvl_wp);",
        "  if (!pi) return 11;",
        "  if (!psp) return 12;",
        "  if (!pwp) return 13;",
        "  nvl_close(heap);",
        "  return 0;",
        "}",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    final LLVMModulePassManager pm = new LLVMModulePassManager();
    pm.addTargetData(exec.getTargetData());
    addNVLPasses(pm);
    pm.run(mod);
    mod.dump();
    // Running requires at least nvlrt_alloc_nv.
    assumePmemobjLibs();
    checkIntFn(exec, mod, "fn", 0);
    exec.dispose();
  }

  // ----------------
  // nvl_alloc_length

  @Test public void nvl_alloc_length() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final String heapFile = mkHeapFile("test.nvl").getAbsolutePath();
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        "#include <nvl.h>",
        "nvl_heap_t *heap;",
        "int createHeap() {",
        "  heap = nvl_create(\""+heapFile+"\", 0, 0600);",
        "  if (!heap) return 1;",
        "  return 0;",
        "}",
        "int constNull() {",
        "  return nvl_alloc_length(0);",
        "}",
        "nvl int *null = 0;",
        "int globalNull() {",
        "  return nvl_alloc_length(null);",
        "}",
        "int allocInt(int n, int off) {",
        "  nvl int *p = nvl_alloc_nv(heap, n, int) + off;",
        "  return nvl_alloc_length(p);",
        "}",
        "struct T {int i; double d;};",
        "int allocStruct(int n, int off) {",
        "  nvl struct T *p = nvl_alloc_nv(heap, n, struct T) + off;",
        "  return nvl_alloc_length(p);",
        "}",
        "int allocChar(int n, int off) {",
        "  nvl char *p = nvl_alloc_nv(heap, n, char) + off;",
        "  return nvl_alloc_length(p);",
        "}",
        "int closeHeap() {",
        "  nvl_close(heap);",
        "  return 0;",
        "}",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    final LLVMModulePassManager pm = new LLVMModulePassManager();
    pm.addTargetData(exec.getTargetData());
    addNVLPasses(pm);
    pm.run(mod);
    mod.dump();
    // Running requires at least nvlrt_alloc_nv.
    assumePmemobjLibs();
    checkIntFn(exec, mod, "createHeap", 0);
    checkIntFn(exec, mod, "constNull", 0);
    checkIntFn(exec, mod, "globalNull", 0);
    checkIntFn(exec, mod, 3, "allocInt", 3, 0);
    checkIntFn(exec, mod, 6, "allocInt", 6, 1);
    checkIntFn(exec, mod, 14, "allocInt", 14, -1);
    checkIntFn(exec, mod, 9, "allocInt", 9, 30);
    checkIntFn(exec, mod, 5, "allocInt", 5, -10);
    checkIntFn(exec, mod, 8, "allocStruct", 8, 5);
    checkIntFn(exec, mod, 4, "allocChar", 4, 0);
    checkIntFn(exec, mod, 6, "allocChar", 6, 2);
    checkIntFn(exec, mod, "closeHeap", 0);
    exec.dispose();
  }

  // ------------------
  // reference counting

  @Test public void nvrefsWithoutMem2reg() throws Exception {
    nvrefs(false);
  }
  @Test public void nvrefsWithMem2reg() throws Exception {
    nvrefs(true);
  }
  // The goal here is to check NV-to-NV ref counting, so we feel free to
  // force V-to-NV decs when we might (depending on -mem2reg's success)
  // otherwise have to wait until the end of the function for them.
  private void nvrefs(boolean mem2regForRefCounting) throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final File heapFile = mkHeapFile("test.nvl");
    final String heapFileAbs = heapFile.getAbsolutePath();
    final SimpleResult simpleResult = buildLLVMSimple(
      targetTriple, targetDataLayout,
      new String[]{
        "#include <nvl.h>",
        "#include <stddef.h>",
        "#include <stdio.h>",
        "#include <nvlrt-test.h>",
        // Recursive, but compiler should avoid infinite loop when
        // generating deep dec function.
        "struct S { nvl struct S *p; };",

        // Check that (shallow) free happens when overwriting old pointer
        // with either non-null or null pointer.

        "int set_root() {",
        "  nvlrt_resetStats();",
        "  nvl_heap_t *heap = nvl_create(\""+heapFileAbs+"\", 0, 0600);",
        "  if (!heap) return 1;",
        "  if (nvlrt_get_numAllocNV() != 0) return 2;",
        "  if (nvlrt_get_numFreeNV() != 0) return 3;",
        "  {",
        "    nvl int *p = nvl_alloc_nv(heap, 1, int);",
        "    if (nvlrt_get_numAllocNV() != 1) return 4;",
        "    if (nvlrt_get_numFreeNV() != 0) return 5;",
        "    nvl_set_root(heap, p);",
        "    p = 0;", // force V-to-NV dec now
        "  }",
        "  if (nvlrt_get_numAllocNV() != 1) return 6;",
        "  if (nvlrt_get_numFreeNV() != 0) return 7;",
        "  {",
        "    nvl int *p = nvl_alloc_nv(heap, 1, int);",
        "    if (nvlrt_get_numAllocNV() != 2) return 8;",
        "    if (nvlrt_get_numFreeNV() != 0) return 9;",
        "    nvl_set_root(heap, p);",
        "    p = 0;", // force V-to-NV dec now
        "  }",
        "  if (nvlrt_get_numAllocNV() != 2) return 10;",
        "  if (nvlrt_get_numFreeNV() != 1) return 11;",
        "  nvl_close(heap);",
        "  heap = nvl_open(\""+heapFileAbs+"\");",
        "  if (!heap) return 12;",
        "  if (nvlrt_get_numAllocNV() != 2) return 13;",
        "  if (nvlrt_get_numFreeNV() != 1) return 14;",
        "  nvl_set_root(heap, (nvl int *)0);",
        "  if (nvlrt_get_numAllocNV() != 2) return 15;",
        "  if (nvlrt_get_numFreeNV() != 2) return 16;",
        "  nvl_close(heap);",
        "  if (nvlrt_get_numAllocNV() != 2) return 17;",
        "  if (nvlrt_get_numFreeNV() != 2) return 18;",
        "  return 0;",
        "}",

        "int store() {",
        "  nvlrt_resetStats();",
        "  nvl_heap_t *heap = nvl_create(\""+heapFileAbs+"\", 0, 0600);",
        "  if (!heap) return 1;",
        "  if (nvlrt_get_numAllocNV() != 0) return 2;",
        "  if (nvlrt_get_numFreeNV() != 0) return 3;",
        "  {",
        "    nvl struct S *s = nvl_alloc_nv(heap, 1, struct S);",
        "    nvl_set_root(heap, s);",
        "    s->p = nvl_alloc_nv(heap, 1, struct S);",
        "    if (nvlrt_get_numAllocNV() != 2) return 4;",
        "    if (nvlrt_get_numFreeNV() != 0) return 5;",
        "    s = 0;", // force V-to-NV dec now
        "  }",
        "  nvl_close(heap);",
        "  heap = nvl_open(\""+heapFileAbs+"\");",
        "  if (!heap) return 6;",
        "  {",
        "    nvl struct S *s = nvl_get_root(heap, struct S);",
        "    if (nvlrt_get_numAllocNV() != 2) return 7;",
        "    if (nvlrt_get_numFreeNV() != 0) return 8;",
        "    s->p = nvl_alloc_nv(heap, 1, struct S);",
        "    if (nvlrt_get_numAllocNV() != 3) return 9;",
        "    if (nvlrt_get_numFreeNV() != 1) return 10;",
        "    s->p = NULL;",
        "    if (nvlrt_get_numAllocNV() != 3) return 11;",
        "    if (nvlrt_get_numFreeNV() != 2) return 12;",
        "    s = 0;", // force V-to-NV dec now
        "  }",
        "  if (nvlrt_get_numAllocNV() != 3) return 13;",
        "  if (nvlrt_get_numFreeNV() != 2) return 14;",
        "  nvl_close(heap);",
        "  if (nvlrt_get_numAllocNV() != 3) return 15;",
        "  if (nvlrt_get_numFreeNV() != 2) return 16;",
        "  return 0;",
        "}",

        // Check that (deep) free happens for struct fields of pointer type,
        // and check that a cycle causes a memory leak but does not cause an
        // infinite loop.

        "int set_root_struct_fields() {",
        "  nvlrt_resetStats();",
        "  nvl_heap_t *heap = nvl_create(\""+heapFileAbs+"\", 0, 0600);",
        "  if (!heap) return 1;",
        "  if (nvlrt_get_numAllocNV() != 0) return 2;",
        "  if (nvlrt_get_numFreeNV() != 0) return 3;",
        // no cycle
        "  {",
        "    nvl struct S *s = nvl_alloc_nv(heap, 1, struct S);",
        "    nvl_set_root(heap, s);",
        "    s->p = nvl_alloc_nv(heap, 1, struct S);",
        "    s = 0;", // force V-to-NV dec now
        "  }",
        "  if (nvlrt_get_numAllocNV() != 2) return 4;",
        "  if (nvlrt_get_numFreeNV() != 0) return 5;",
        "  nvl_set_root(heap, (nvl struct S*)0);",
        "  if (nvlrt_get_numAllocNV() != 2) return 6;",
        "  if (nvlrt_get_numFreeNV() != 2) return 7;",
        // cycle
        "  {",
        "    nvl struct S *s = nvl_alloc_nv(heap, 1, struct S);",
        "    nvl_set_root(heap, s);",
        "    s->p = s;",
        "    s = 0;", // force V-to-NV dec now
        "  }",
        "  nvl_set_root(heap, 0);",
        "  if (nvlrt_get_numAllocNV() != 3) return 8;",
        "  if (nvlrt_get_numFreeNV() != 2) return 9;",
        "  nvl_close(heap);",
        "  if (nvlrt_get_numAllocNV() != 3) return 10;",
        "  if (nvlrt_get_numFreeNV() != 2) return 11;",
        "  return 0;",
        "}",

        "int store_struct_fields() {",
        "  nvlrt_resetStats();",
        "  nvl_heap_t *heap = nvl_create(\""+heapFileAbs+"\", 0, 0600);",
        "  if (!heap) return 1;",
        "  if (nvlrt_get_numAllocNV() != 0) return 2;",
        "  if (nvlrt_get_numFreeNV() != 0) return 3;",
        // no cycle
        "  {",
        "    nvl struct S *s = nvl_alloc_nv(heap, 1, struct S);",
        "    nvl_set_root(heap, s);",
        "    s->p = nvl_alloc_nv(heap, 1, struct S);",
        "    s->p->p = nvl_alloc_nv(heap, 1, struct S);",
        "    s = 0;", // force V-to-NV dec now
        "  }",
        "  if (nvlrt_get_numAllocNV() != 3) return 4;",
        "  if (nvlrt_get_numFreeNV() != 0) return 5;",
        "  nvl_get_root(heap, struct S)->p = NULL;",
        "  if (nvlrt_get_numAllocNV() != 3) return 6;",
        "  if (nvlrt_get_numFreeNV() != 2) return 7;",
        // cycle
        "  {",
        "    nvl struct S *s = nvl_get_root(heap, struct S);",
        "    s->p = nvl_alloc_nv(heap, 1, struct S);",
        "    s->p->p = s->p;",
        "    s = 0;", // force V-to-NV dec now
        "  }",
        "  if (nvlrt_get_numAllocNV() != 4) return 8;",
        "  if (nvlrt_get_numFreeNV() != 2) return 9;",
        "  nvl_get_root(heap, struct S)->p = NULL;",
        "  if (nvlrt_get_numAllocNV() != 4) return 10;",
        "  if (nvlrt_get_numFreeNV() != 2) return 11;",
        "  nvl_close(heap);",
        "  if (nvlrt_get_numAllocNV() != 4) return 12;",
        "  if (nvlrt_get_numFreeNV() != 2) return 13;",
        "  return 0;",
        "}",

        // Check that (deep) free happens for array elements of pointer
        // type, targets of pointers, and nested structs.

        "int array_elements() {",
        "  nvlrt_resetStats();",
        "  nvl_heap_t *heap = nvl_create(\""+heapFileAbs+"\", 0, 0600);",
        "  if (!heap) return 1;",
        "  if (nvlrt_get_numAllocNV() != 0) return 2;",
        "  if (nvlrt_get_numFreeNV() != 0) return 3;",
        "  {",
        "    nvl int * nvl (*p)[3] = nvl_alloc_nv(heap, 1, nvl int *[3]);",
        "    nvl_set_root(heap, p);",
        "    (*p)[0] = nvl_alloc_nv(heap, 1, int);",
        "    (*p)[2] = nvl_alloc_nv(heap, 1, int);",
        "    p = 0;", // force V-to-NV dec now
        "  }",
        "  if (nvlrt_get_numAllocNV() != 3) return 4;",
        "  if (nvlrt_get_numFreeNV() != 0) return 5;",
        "  nvl_set_root(heap, (nvl int * nvl (*)[3])0);",
        "  if (nvlrt_get_numAllocNV() != 3) return 6;",
        "  if (nvlrt_get_numFreeNV() != 3) return 7;",
        "  nvl_close(heap);",
        "  if (nvlrt_get_numAllocNV() != 3) return 8;",
        "  if (nvlrt_get_numFreeNV() != 3) return 9;",
        "  return 0;",
        "}",

        "int multi_element_alloc() {",
        "  nvlrt_resetStats();",
        "  nvl_heap_t *heap = nvl_create(\""+heapFileAbs+"\", 0, 0600);",
        "  if (!heap) return 1;",
        "  if (nvlrt_get_numAllocNV() != 0) return 2;",
        "  if (nvlrt_get_numFreeNV() != 0) return 3;",
        "  {",
        "    nvl int * nvl *p = nvl_alloc_nv(heap, 4, nvl int *);",
        "    nvl_set_root(heap, p);",
        "    p[1] = nvl_alloc_nv(heap, 1, int);",
        "    p[2] = nvl_alloc_nv(heap, 1, int);",
        "    p[3] = nvl_alloc_nv(heap, 1, int);",
        "    p = 0;", // force V-to-NV dec now
        "  }",
        "  if (nvlrt_get_numAllocNV() != 4) return 4;",
        "  if (nvlrt_get_numFreeNV() != 0) return 5;",
        "  nvl_set_root(heap, (nvl int * nvl *)0);",
        "  if (nvlrt_get_numAllocNV() != 4) return 6;",
        "  if (nvlrt_get_numFreeNV() != 4) return 7;",
        "  nvl_close(heap);",
        "  if (nvlrt_get_numAllocNV() != 4) return 8;",
        "  if (nvlrt_get_numFreeNV() != 4) return 9;",
        "  return 0;",
        "}",

        "int pointer_target() {",
        "  nvlrt_resetStats();",
        "  nvl_heap_t *heap = nvl_create(\""+heapFileAbs+"\", 0, 0600);",
        "  if (!heap) return 1;",
        "  if (nvlrt_get_numAllocNV() != 0) return 2;",
        "  if (nvlrt_get_numFreeNV() != 0) return 3;",
        "  {",
        "    nvl int * nvl *p = nvl_alloc_nv(heap, 1, nvl int *);",
        "    nvl_set_root(heap, p);",
        "    *p = nvl_alloc_nv(heap, 1, int);",
        "    p = 0;", // force V-to-NV dec now
        "  }",
        "  if (nvlrt_get_numAllocNV() != 2) return 4;",
        "  if (nvlrt_get_numFreeNV() != 0) return 5;",
        "  nvl_set_root(heap, (nvl int * nvl *)0);",
        "  if (nvlrt_get_numAllocNV() != 2) return 6;",
        "  if (nvlrt_get_numFreeNV() != 2) return 7;",
        "  nvl_close(heap);",
        "  if (nvlrt_get_numAllocNV() != 2) return 8;",
        "  if (nvlrt_get_numFreeNV() != 2) return 9;",
        "  return 0;",
        "}",

        "struct N { struct S s; struct { nvl int *p; } a; };",
        "int nested_struct() {",
        "  nvlrt_resetStats();",
        "  nvl_heap_t *heap = nvl_create(\""+heapFileAbs+"\", 0, 0600);",
        "  if (!heap) return 1;",
        "  if (nvlrt_get_numAllocNV() != 0) return 2;",
        "  if (nvlrt_get_numFreeNV() != 0) return 3;",
        "  {",
        "    nvl struct N *p = nvl_alloc_nv(heap, 1, struct N);",
        "    nvl_set_root(heap, p);",
        "    p->s.p = nvl_alloc_nv(heap, 1, struct S);",
        "    p->a.p = nvl_alloc_nv(heap, 1, int);",
        "    p = 0;", // force V-to-NV dec now
        "  }",
        "  if (nvlrt_get_numAllocNV() != 3) return 4;",
        "  if (nvlrt_get_numFreeNV() != 0) return 5;",
        "  nvl_set_root(heap, (nvl struct N *)0);",
        "  if (nvlrt_get_numAllocNV() != 3) return 6;",
        "  if (nvlrt_get_numFreeNV() != 3) return 7;",
        "  nvl_close(heap);",
        "  if (nvlrt_get_numAllocNV() != 3) return 8;",
        "  if (nvlrt_get_numFreeNV() != 3) return 9;",
        "  return 0;",
        "}",

        // Make sure a nvl_set_root or store of the same pointer doesn't
        // produce a free. That is, the inc must happen before the dec.

        "int inc_before_dec() {",
        "  nvlrt_resetStats();",
        "  nvl_heap_t *heap = nvl_create(\""+heapFileAbs+"\", 0, 0600);",
        "  if (!heap) return 1;",
        "  if (nvlrt_get_numAllocNV() != 0) return 2;",
        "  if (nvlrt_get_numFreeNV() != 0) return 3;",
        "  nvl_set_root(heap, nvl_alloc_nv(heap, 1, struct S));",
        "  nvl_set_root(heap, nvl_get_root(heap, struct S));", // set root
        "  if (nvlrt_get_numAllocNV() != 1) return 4;",
        "  if (nvlrt_get_numFreeNV() != 0) return 5;",
        "  nvl_get_root(heap, struct S)->p",
        "    = nvl_alloc_nv(heap, 1, struct S);",
        "  nvl_get_root(heap, struct S)->p",
        "    = nvl_get_root(heap, struct S)->p;", // store
        "  if (nvlrt_get_numAllocNV() != 2) return 6;",
        "  if (nvlrt_get_numFreeNV() != 0) return 7;",
        "  nvl_close(heap);",
        "  if (nvlrt_get_numAllocNV() != 2) return 8;",
        "  if (nvlrt_get_numFreeNV() != 0) return 9;",
        "  return 0;",
        "}",

        // Make sure free works right when last reference removed is not at
        // the beginning of the allocation.

        "int middle_of_alloc() {",
        "  nvlrt_resetStats();",
        "  nvl_heap_t *heap = nvl_create(\""+heapFileAbs+"\", 0, 0600);",
        "  if (!heap) return 1;",
        "  if (nvlrt_get_numAllocNV() != 0) return 2;",
        "  if (nvlrt_get_numFreeNV() != 0) return 3;",
        "  {",
        "    nvl struct N *p = nvl_alloc_nv(heap, 10, struct N);", 
        "    nvl_set_root(heap, &p[5].a.p);",
        "    for (int i = 0; i < 10; ++i) {",
        "      p[i].s.p = nvl_alloc_nv(heap, 1, struct S);",
        "      if (i%2)",
        "        p[i].s.p->p = nvl_alloc_nv(heap, 1, struct S);",
        "    }",
        "    p = 0;", // force V-to-NV dec now
        "  }",
        "  if (nvlrt_get_numAllocNV() != 16) return 4;",
        "  if (nvlrt_get_numFreeNV() != 0) return 5;",
        "  nvl_set_root(heap, (nvl int * nvl *)0);",
        "  if (nvlrt_get_numAllocNV() != 16) return 6;",
        "  if (nvlrt_get_numFreeNV() != 16) return 7;",
        "  nvl_close(heap);",
        "  if (nvlrt_get_numAllocNV() != 16) return 8;",
        "  if (nvlrt_get_numFreeNV() != 16) return 9;",
        "  return 0;",
        "}",

        // Make sure storing a struct does not destroy the nextPtrOff fields
        // in contained pointers.

        "struct StructWithPtrs {",
        // ptr in struct in array in struct
        "  struct NestedStructWithPtrs { nvl int *p; } sa[6];",
        // ptr in array in struct
        "  nvl int *pa[6];",
        // double ptr
        "  nvl int * nvl *pp;",
        "};",

        "int store_structWithPtrs() {",
        "  nvlrt_resetStats();",
        "  nvl_heap_t *heap = nvl_create(\""+heapFileAbs+"\", 0, 0600);",
        "  if (!heap) return 1;",
        "  if (nvlrt_get_numAllocNV() != 0) return 2;",
        "  if (nvlrt_get_numFreeNV() != 0) return 3;",
        "  {",
        "    nvl struct StructWithPtrs *p",
        "      = nvl_alloc_nv(heap, 1, struct StructWithPtrs);",
        "    nvl_set_root(heap, &p->sa[2].p);",
        // Store a struct whose last nextPtrOff should remain null. Store
        // it again to be sure decs happen on old pointers.
        "    for (int i = 0; i < 2; ++i) {",
        "      struct StructWithPtrs s = {{{0}, {0}, {0}, {0}, {0}, {0}},",
        "                                 {0, 0, 0, 0, 0, 0},",
        "                                 0};",
        "      s.pa[1] = nvl_alloc_nv(heap, 1, int);",
        "      s.pa[3] = nvl_alloc_nv(heap, 1, int);",
        "      s.pa[5] = nvl_alloc_nv(heap, 1, int);",
        "      *s.pa[1] = 40;",
        "      *s.pa[3] = 50;",
        "      *s.pa[5] = 60;",
        "      s.pp = nvl_alloc_nv(heap, 1, nvl int *);",
        "      *s.pp = nvl_alloc_nv(heap, 1, int);",
        "      **s.pp = 70;",
        "      *p = s;",
        "      *p = s;", // inc/dec should balance
        "      if (nvlrt_get_numAllocNV() != 1 + (i+1)*5) return 4+i*2;",
        "      if (nvlrt_get_numFreeNV() != i*5) return 5+i*2;",
               // force V-to-NV decs now
        "      s.pa[1] = 0;",
        "      s.pa[3] = 0;",
        "      s.pa[5] = 0;",
        "      s.pp = 0;",
        "    }",
        // Store a nested struct, whose last nextPtrOff should continue to
        // refer to the pointer after the nested struct. Store it again to
        // be sure decs happen on old pointers.
        "    for (int i = 0; i < 2; ++i) {",
        "      struct NestedStructWithPtrs n = {0};",
        "      n.p = nvl_alloc_nv(heap, 1, int);",
        "      *n.p = 10;",
        "      p->sa[0] = n;",
        "      p->sa[0] = n;", // inc/dec should balance
        "      n.p = nvl_alloc_nv(heap, 1, int);",
        "      p->sa[2] = n;",
        "      p->sa[2] = n;", // inc/dec should balance
        "      n.p = nvl_alloc_nv(heap, 1, int);",
        "      p->sa[4] = n;",
        "      p->sa[4] = n;", // inc/dec should balance
        "      *p->sa[2].p = 20;",
        "      *n.p = 30;",
        "      if (nvlrt_get_numAllocNV() != 11 + (i+1)*3) return 8+i*2;",
        "      if (nvlrt_get_numFreeNV() != 5 + i*3) return 9+i*2;",
               // force V-to-NV decs now
        "      n.p = 0;",
        "    }",
        "    struct StructWithPtrs sLoad = *p;",
        "    if (*sLoad.sa[0].p != 10) return 12;",
        "    if (sLoad.sa[1].p != 0) return 13;",
        "    if (*sLoad.sa[2].p != 20) return 14;",
        "    if (sLoad.sa[3].p != 0) return 15;",
        "    if (*sLoad.sa[4].p != 30) return 16;",
        "    if (sLoad.sa[5].p != 0) return 17;",
        "    if (sLoad.pa[0] != 0) return 18;",
        "    if (*sLoad.pa[1] != 40) return 19;",
        "    if (sLoad.pa[2] != 0) return 20;",
        "    if (*sLoad.pa[3] != 50) return 21;",
        "    if (sLoad.pa[4] != 0) return 22;",
        "    if (*sLoad.pa[5] != 60) return 23;",
        "    if (**sLoad.pp != 70) return 24;",
             // force V-to-NV decs now
        "    sLoad.sa[0].p = 0;",
        "    sLoad.sa[2].p = 0;",
        "    sLoad.sa[4].p = 0;",
        "    sLoad.pa[1] = 0;",
        "    sLoad.pa[3] = 0;",
        "    sLoad.pa[5] = 0;",
        "    sLoad.pp = 0;",
        "    p = 0;",
        "  }",
        "  if (nvlrt_get_numAllocNV() != 17) return 25;",
        "  if (nvlrt_get_numFreeNV() != 8) return 26;",
        "  nvl_set_root(heap, 0);",
        "  if (nvlrt_get_numAllocNV() != 17) return 27;",
        "  if (nvlrt_get_numFreeNV() != 17) return 28;",
        "  nvl_close(heap);",
        "  if (nvlrt_get_numAllocNV() != 17) return 29;",
        "  if (nvlrt_get_numFreeNV() != 17) return 30;",
        "  return 0;",
        "}",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    final LLVMModulePassManager pm = new LLVMModulePassManager();
    pm.addTargetData(exec.getTargetData());
    addNVLPasses(pm, mem2regForRefCounting);
    pm.run(mod);
    mod.dump();
    // Running requires at least nvlrt_alloc_nv.
    assumePmemobjLibs();
    checkIntFn(exec, mod, "set_root", 0); heapFile.delete();
    checkIntFn(exec, mod, "store", 0); heapFile.delete();
    checkIntFn(exec, mod, "set_root_struct_fields", 0); heapFile.delete();
    checkIntFn(exec, mod, "store_struct_fields", 0); heapFile.delete();
    checkIntFn(exec, mod, "array_elements", 0); heapFile.delete();
    checkIntFn(exec, mod, "multi_element_alloc", 0); heapFile.delete();
    checkIntFn(exec, mod, "pointer_target", 0); heapFile.delete();
    checkIntFn(exec, mod, "nested_struct", 0); heapFile.delete();
    checkIntFn(exec, mod, "inc_before_dec", 0); heapFile.delete();
    checkIntFn(exec, mod, "middle_of_alloc", 0); heapFile.delete();
    checkIntFn(exec, mod, "store_structWithPtrs", 0); heapFile.delete();
    exec.dispose();
  }

  /**
   * This test case exercises vrefs inc/decs at stores to volatile memory.
   * Running {@code -mem2reg} moves volatile memory into registers and thus
   * could eliminate store instructions and other forms of indirection we
   * want to exercise here. To prevent that, we store volatile data globally
   * instead of locally in this test. See {@link #vrefsAtStoresOrRegs} for
   * cases where we store volatile data locally.
   */
  @Test public void vrefsAtStoreWithoutMem2reg() throws Exception {
    vrefsAtStore(false);
  }
  @Test public void vrefsAtStoreWithMem2reg() throws Exception {
    vrefsAtStore(true);
  }
  private void vrefsAtStore(boolean mem2regForRefCounting)
    throws Exception
  {
    LLVMTargetData.initializeNativeTarget();
    final File heapFile = mkHeapFile("test.nvl");
    final String heapFileAbs = heapFile.getAbsolutePath();
    final SimpleResult simpleResult = buildLLVMSimple(
      targetTriple, targetDataLayout,
      "#include <nvl.h>",
      "#include <stddef.h>",
      "#include <stdio.h>",
      "#include <stdlib.h>",
      "#include <nvlrt-test.h>",

      "nvl_heap_t *heap;",
      "int setup() {",
      "  nvlrt_resetStats();",
      "  heap = nvl_create(\""+heapFileAbs+"\", 0, 0600);",
      "  if (!heap) return 1;",
      "  if (nvlrt_get_numAllocNV() != 0) return 2;",
      "  if (nvlrt_get_numFreeNV() != 0) return 3;",
      "  return 0;",
      "}",

      // V-to-NV pointer.
      "nvl int *p = 0;",
      "int ptr() {",
      "  nvlrt_resetStats();",
      "  p = 0;", // dec null, inc null
      "  if (nvlrt_get_numAllocNV() != 0) return 1;",
      "  if (nvlrt_get_numFreeNV() != 0) return 2;",

      "  p = nvl_alloc_nv(heap, 1, int);", // dec null, inc
      "  if (!p) return 3;",
      "  if (nvlrt_get_numAllocNV() != 1) return 4;",
      "  if (nvlrt_get_numFreeNV() != 0) return 5;",

      "  p = p;", // dec, inc same address
      "  if (nvlrt_get_numAllocNV() != 1) return 6;",
      "  if (nvlrt_get_numFreeNV() != 0) return 7;",

      "  p = nvl_alloc_nv(heap, 1, int);", // dec, inc
      "  if (!p) return 8;",
      "  if (nvlrt_get_numAllocNV() != 2) return 9;",
      "  if (nvlrt_get_numFreeNV() != 1) return 10;",

      "  p = 0;", // dec, inc null
      "  if (nvlrt_get_numAllocNV() != 2) return 11;",
      "  if (nvlrt_get_numFreeNV() != 2) return 12;",

      "  return 0;",
      "}",

      // V-to-NV pointer via pointer.
      "nvl int **pp = &p;",
      "int ptrViaPtr() {",
      "  nvlrt_resetStats();",
      "  if (p != 0) return 99;",
      "  *pp = 0;", // dec null, inc null
      "  if (nvlrt_get_numAllocNV() != 0) return 1;",
      "  if (nvlrt_get_numFreeNV() != 0) return 2;",

      "  *pp = nvl_alloc_nv(heap, 1, int);", // dec null, inc
      "  if (!*pp) return 3;",
      "  if (nvlrt_get_numAllocNV() != 1) return 4;",
      "  if (nvlrt_get_numFreeNV() != 0) return 5;",

      "  *pp = *pp;", // dec, inc same address
      "  if (nvlrt_get_numAllocNV() != 1) return 6;",
      "  if (nvlrt_get_numFreeNV() != 0) return 7;",

      "  *pp = nvl_alloc_nv(heap, 1, int);", // dec, inc
      "  if (!*pp) return 8;",
      "  if (nvlrt_get_numAllocNV() != 2) return 9;",
      "  if (nvlrt_get_numFreeNV() != 1) return 10;",

      "  *pp = 0;", // dec, inc null
      "  if (nvlrt_get_numAllocNV() != 2) return 11;",
      "  if (nvlrt_get_numFreeNV() != 2) return 12;",

      "  return 0;",
      "}",

      // V-to-NV pointers via struct and array.
      "struct S { nvl int *p[2]; } s0 = {{0, 0}}, s1 = {{0, 0}};",
      "int ptrViaStructAndArray() {",
      "  nvlrt_resetStats();",
      "  s1 = s0;", // dec null, inc null; dec null, inc null
      "  if (nvlrt_get_numAllocNV() != 0) return 1;",
      "  if (nvlrt_get_numFreeNV() != 0) return 2;",

      "  s0.p[0] = nvl_alloc_nv(heap, 1, int);", // dec null, inc
      "  if (!s0.p[0]) return 3;",
      "  if (nvlrt_get_numAllocNV() != 1) return 4;",
      "  if (nvlrt_get_numFreeNV() != 0) return 5;",
      "  s1 = s0;", // dec null, inc; dec null, inc null
      "  if (nvlrt_get_numAllocNV() != 1) return 6;",
      "  if (nvlrt_get_numFreeNV() != 0) return 7;",

      "  s0.p[1] = nvl_alloc_nv(heap, 1, int);", // dec null, inc
      "  if (!s0.p[1]) return 8;",
      "  if (nvlrt_get_numAllocNV() != 2) return 9;",
      "  if (nvlrt_get_numFreeNV() != 0) return 10;",
      "  s1 = s0;", // dec, inc same address; dec null, inc
      "  if (nvlrt_get_numAllocNV() != 2) return 11;",
      "  if (nvlrt_get_numFreeNV() != 0) return 12;",

      "  s0.p[0] = nvl_alloc_nv(heap, 1, int);", // dec, inc
      "  if (!s0.p[0]) return 13;",
      "  if (nvlrt_get_numAllocNV() != 3) return 14;",
      "  if (nvlrt_get_numFreeNV() != 0) return 15;",
      "  s1 = s0;", // dec, inc; dec, inc same address
      "  if (nvlrt_get_numAllocNV() != 3) return 16;",
      "  if (nvlrt_get_numFreeNV() != 1) return 17;",

      "  s0.p[1] = nvl_alloc_nv(heap, 1, int);", // dec, inc
      "  if (!s0.p[1]) return 18;",
      "  if (nvlrt_get_numAllocNV() != 4) return 19;",
      "  if (nvlrt_get_numFreeNV() != 1) return 20;",
      "  s1 = s0;", // dec, inc same address; dec, inc
      "  if (nvlrt_get_numAllocNV() != 4) return 21;",
      "  if (nvlrt_get_numFreeNV() != 2) return 22;",

      "  s0.p[0] = 0;", // dec, inc null
      "  if (nvlrt_get_numAllocNV() != 4) return 23;",
      "  if (nvlrt_get_numFreeNV() != 2) return 24;",
      "  s1 = s0;", // dec, inc null; dec, inc same address
      "  if (nvlrt_get_numAllocNV() != 4) return 25;",
      "  if (nvlrt_get_numFreeNV() != 3) return 26;",

      "  s0.p[1] = 0;", // dec, inc null
      "  if (nvlrt_get_numAllocNV() != 4) return 27;",
      "  if (nvlrt_get_numFreeNV() != 3) return 28;",
      "  s1 = s0;", // dec null, inc null; dec, inc null
      "  if (nvlrt_get_numAllocNV() != 4) return 29;",
      "  if (nvlrt_get_numFreeNV() != 4) return 30;",

      "  return 0;",
      "}");
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    final LLVMModulePassManager pm = new LLVMModulePassManager();
    pm.addTargetData(exec.getTargetData());
    addNVLPasses(pm, mem2regForRefCounting);
    pm.run(mod);
    mod.dump();
    // Running requires at least nvlrt_alloc_nv.
    assumePmemobjLibs();
    checkIntFn(exec, mod, "setup", 0);
    checkIntFn(exec, mod, "ptr", 0);
    checkIntFn(exec, mod, "ptrViaPtr", 0);
    checkIntFn(exec, mod, "ptrViaStructAndArray", 0);
    exec.dispose();
  }

  /**
   * This test case exercises vrefs inc/decs that might be at stores or
   * might be on registers, depending on whether we run {@code -mem2reg} and
   * on how successful it is. To make sure stores to local variables work
   * correctly, we want to exercise cases where {@code -mem2reg} will fail
   * to reliably register-allocate locals. However, rather than trying to
   * find such a case that remains stable across LLVM versions, we just make
   * sure to try without {@code -mem2reg} (as well as with). Because decs
   * are currently called for allocas at function exit but can be called
   * earlier if {@code -mem2reg} is successful, these tests try to keep
   * local variables alive until function exit so we can reliably assume
   * that's where their decs will occur.
   */
  @Test public void vrefsAtStoresOrRegsWithoutMem2reg() throws Exception {
    vrefsAtStoresOrRegs(false);
  }
  @Test public void vrefsAtStoresOrRegsWithMem2reg() throws Exception {
    vrefsAtStoresOrRegs(true);
  }
  private void vrefsAtStoresOrRegs(boolean mem2regForRefCounting)
    throws Exception
  {
    LLVMTargetData.initializeNativeTarget();
    final File heapFile = mkHeapFile("test.nvl");
    final String heapFileAbs = heapFile.getAbsolutePath();
    final SimpleResult simpleResult = buildLLVMSimple(
      targetTriple, targetDataLayout,
      "#include <nvl.h>",
      "#include <stddef.h>",
      "#include <stdio.h>",
      "#include <stdlib.h>",
      "#include <nvlrt-test.h>",

      "nvl_heap_t *heap;",
      "int setup() {",
      "  nvlrt_resetStats();",
      "  heap = nvl_create(\""+heapFileAbs+"\", 0, 0600);",
      "  if (!heap) return 1;",
      "  if (nvlrt_get_numAllocNV() != 0) return 2;",
      "  if (nvlrt_get_numFreeNV() != 0) return 3;",
      "  return 0;",
      "}",

      // V-to-NV pointer.
      "int ptr0() {",
      "  nvl int *lpi = 0;", // dec init, inc null
      "  lpi = 0;", // dec null, inc null
      "  if (nvlrt_get_numAllocNV() != 0) return 1;",
      "  if (nvlrt_get_numFreeNV() != 0) return 2;",

      "  lpi = nvl_alloc_nv(heap, 1, int);", // dec null, inc
      "  if (!lpi) return 3;",
      "  if (nvlrt_get_numAllocNV() != 1) return 4;",
      "  if (nvlrt_get_numFreeNV() != 0) return 5;",

      "  lpi = lpi;", // dec, inc same address
      "  if (nvlrt_get_numAllocNV() != 1) return 6;",
      "  if (nvlrt_get_numFreeNV() != 0) return 7;",

      "  if (!lpi) return 8;", // keep alive
      "  lpi = nvl_alloc_nv(heap, 1, int);", // dec, inc
      "  if (!lpi) return 9;",
      "  if (nvlrt_get_numAllocNV() != 2) return 10;",
      "  if (nvlrt_get_numFreeNV() != 1) return 11;",

      "  if (!lpi) return 12;", // keep alive
      "  return 0;",
      "}",
      "int ptr() {",
      "  nvlrt_resetStats();",

      "  int r = ptr0();",
      "  if (r) return r;",
      "  if (nvlrt_get_numAllocNV() != 2) return 13;",
      "  if (nvlrt_get_numFreeNV() != 2) return 14;",

      "  nvl int *lpi = nvl_alloc_nv(heap, 1, int);", // dec init, inc
      "  if (!lpi) return 15;",
      "  if (nvlrt_get_numAllocNV() != 3) return 16;",
      "  if (nvlrt_get_numFreeNV() != 2) return 17;",

      "  if (!lpi) return 18;", // keep alive
      "  lpi = 0;", // dec, inc null
      "  if (nvlrt_get_numAllocNV() != 3) return 19;",
      "  if (nvlrt_get_numFreeNV() != 3) return 20;",

      "  return 0;",
      "}",

      // V-to-NV pointers via struct and array.
      "struct S { int i; struct { nvl int *p; } a[2]; };",
      "int ptrViaStructAndArray0() {",
      // dec init, inc null; dec init, inc
      "  struct S s0 = {0, {{0}, {nvl_alloc_nv(heap, 1, int)}}};",
      "  if (!s0.a[1].p) return 1;",
      // dec init, inc; dec init, inc null
      "  struct S s1 = {1, {{nvl_alloc_nv(heap, 1, int)}, {0}}};",
      "  if (!s1.a[0].p) return 2;",
      "  if (nvlrt_get_numAllocNV() != 2) return 3;",
      "  if (nvlrt_get_numFreeNV() != 0) return 4;",

      "  if (!s1.a[0].p) return 5;", // keep alive
      "  s1 = s0;", // dec, inc null; dec null, inc
      "  if (nvlrt_get_numAllocNV() != 2) return 6;",
      "  if (nvlrt_get_numFreeNV() != 1) return 7;",

      "  s0.a[0].p = nvl_alloc_nv(heap, 1, int);", // dec null, inc
      "  if (!s0.a[0].p) return 8;",
      "  if (nvlrt_get_numAllocNV() != 3) return 9;",
      "  if (nvlrt_get_numFreeNV() != 1) return 10;",
      "  s1 = s0;", // dec null, inc; dec, inc same address
      "  if (nvlrt_get_numAllocNV() != 3) return 11;",
      "  if (nvlrt_get_numFreeNV() != 1) return 12;",

      "  s0.a[1].p = nvl_alloc_nv(heap, 1, int);", // dec, inc
      "  if (!s0.a[1].p) return 13;",
      "  if (nvlrt_get_numAllocNV() != 4) return 14;",
      "  if (nvlrt_get_numFreeNV() != 1) return 15;",
      "  if (!s1.a[1].p) return 16;", // keep alive
      "  s1 = s0;", // dec, inc same address; dec, inc
      "  if (nvlrt_get_numAllocNV() != 4) return 17;",
      "  if (nvlrt_get_numFreeNV() != 2) return 18;",

      "  if (!s0.a[0].p) return 19;", // keep alive
      "  if (!s0.a[1].p) return 20;", // keep alive
      "  if (!s1.a[0].p) return 21;", // keep alive
      "  if (!s1.a[1].p) return 22;", // keep alive
      "  return 0;",
      "}",
      "int ptrViaStructAndArray() {",
      "  nvlrt_resetStats();",
      "  int r = ptrViaStructAndArray0();",
      "  if (r) return r;",
      "  if (nvlrt_get_numAllocNV() != 4) return 23;",
      "  if (nvlrt_get_numFreeNV() != 4) return 24;",
      "  return 0;",
      "}",

      // The main new check here is that initializing a top-level array is
      // correct.
      "int ptrViaArray0() {",
      // dec init, inc; dec init, inc null
      "  nvl int *a[2] = {nvl_alloc_nv(heap, 1, int), 0};",
      "  if (!a[0]) return 1;",
      "  if (nvlrt_get_numAllocNV() != 1) return 2;",
      "  if (nvlrt_get_numFreeNV() != 0) return 3;",

      // dec null, inc
      "  a[1] = nvl_alloc_nv(heap, 1, int);",
      "  if (!a[1]) return 4;",
      "  if (nvlrt_get_numAllocNV() != 2) return 5;",
      "  if (nvlrt_get_numFreeNV() != 0) return 6;",

      "  if (!a[0]) return 7;", // keep alive
      "  if (!a[1]) return 8;", // keep alive
      "  return 0;",
      "}",
      "int ptrViaArray() {",
      "  nvlrt_resetStats();",
      "  int r = ptrViaArray0();",
      "  if (r) return r;",
      "  if (nvlrt_get_numAllocNV() != 2) return 9;",
      "  if (nvlrt_get_numFreeNV() != 2) return 10;",
      "  return 0;",
      "}");
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    final LLVMModulePassManager pm = new LLVMModulePassManager();
    pm.addTargetData(exec.getTargetData());
    addNVLPasses(pm, mem2regForRefCounting);
    pm.run(mod);
    mod.dump();
    // Running requires at least nvlrt_alloc_nv.
    assumePmemobjLibs();
    checkIntFn(exec, mod, "setup", 0);
    checkIntFn(exec, mod, "ptr", 0);
    checkIntFn(exec, mod, "ptrViaStructAndArray", 0);
    checkIntFn(exec, mod, "ptrViaArray", 0);
    exec.dispose();
  }

  @Test public void vrefsAtClose() throws Exception {
    // Running requires at least nvlrt_alloc_nv.
    assumePmemobjLibs();
    LLVMTargetData.initializeNativeTarget();
    final File workDir = mkTmpDir("");
    final File heapFile = mkHeapFile("test.nvl");
    //final File heapFile = new File("/opt/fio/scratch/jum/test.nvl");
    //heapFile.delete();
    final String heapFileAbs = heapFile.getAbsolutePath();
    // Make sure all NVM allocations with only V-to-NV refs are freed upon
    // opening the heap.  However, that requires creating the allocations in
    // a separate process that terminates without closing the heap. That is,
    // pmemobj will fail to open a heap that is still open in the same
    // process.
    // 
    // Also, make sure allocations that still have NV-to-NV refs are not
    // freed then.
    nvlOpenarcCCAndRun(
      "vrefOnlyNoClose", workDir, "",
      new String[]{
        "#include <nvl.h>",
        "#include <stddef.h>",
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <nvlrt-test.h>",
        "struct S { nvl int *p; nvl int * nvl *pp; };",
        "int main() {",
        "  nvl_heap_t *heap = nvl_create(\""+heapFileAbs+"\", 0, 0600);",
        "  if (!heap) return 1;",
        "  nvl struct S *s0 = nvl_alloc_nv(heap, 2, struct S);",
        "  s0[0].p = nvl_alloc_nv(heap, 1, int);",
        "  s0[0].pp = nvl_alloc_nv(heap, 1, nvl int *);",
        "  *s0[0].pp = nvl_alloc_nv(heap, 1, int);",
        "  s0[1].p = nvl_alloc_nv(heap, 1, int);",
        "  s0[1].pp = nvl_alloc_nv(heap, 1, nvl int *);",
        "  *s0[1].pp = nvl_alloc_nv(heap, 1, int);",
        "  nvl int *i = nvl_alloc_nv(heap, 1, int);",
        "  nvl struct S *s1 = nvl_alloc_nv(heap, 1, struct S);",
        "  s1->p = nvl_alloc_nv(heap, 1, int);",
        "  s1->pp = nvl_alloc_nv(heap, 2, nvl int *);",
        "  nvl_set_root(heap, s1);", // remove from vrefOnly
        "  if (nvlrt_get_numAllocNV() != 11) return 2;",
        "  if (nvlrt_get_numFreeNV() != 0) return 3;",
        "  exit(0);", // terminates program without automatic V-to-NV decs
        "  return s0 && i && s1;", // make these live until after exit
        "}",
      },
      new String[0], 0);
    final SimpleResult simpleResult = buildLLVMSimple(
      targetTriple, targetDataLayout,
      "#include <nvl.h>",
      "#include <stddef.h>",
      "#include <stdio.h>",
      "#include <nvlrt-test.h>",
      "struct S { nvl int *p; nvl int * nvl *pp; };",
      "struct T { struct { nvl int *a[2]; int i; } t; };",
      "int vrefOnlyFreeAll() {",
      "  {",
      "    nvlrt_resetStats();",
      "    nvl_heap_t *heap = nvl_open(\""+heapFileAbs+"\");", // frees 8
      "    if (!heap) return 1;",
      "    if (nvlrt_get_numAllocNV() != 0) return 2;",
      "    if (nvlrt_get_numFreeNV() != 8) return 3;",
      // There are still 3 allocs with NV-to-NV refs.
      "    nvl int * nvl *pp = nvl_get_root(heap, struct S)->pp;",
      "    nvl_get_root(heap, struct S)->pp = 0;", // add to vrefOnly
      "    if (nvlrt_get_numAllocNV() != 0) return 4;",
      "    if (nvlrt_get_numFreeNV() != 8) return 5;",
      "    nvl struct T *t = nvl_alloc_nv(heap, 1, struct T);",
      "    t->t.a[0] = nvl_alloc_nv(heap, 1, int);",
      "    t->t.a[1] = nvl_alloc_nv(heap, 1, int);",
      "    t->t.i = 5;",
      "    nvl int *i = nvl_alloc_nv(heap, 1, int);",
      "    if (!pp || !t || !i) return 6;", // keep alive
      "    nvl_close(heap);", // frees 5
      "    if (nvlrt_get_numAllocNV() != 4) return 7;",
      "    if (nvlrt_get_numFreeNV() != 13) return 8;",
      // Make sure decs that happen after a close aren't corrupt.
      "  }",
      "  if (nvlrt_get_numAllocNV() != 4) return 9;",
      "  if (nvlrt_get_numFreeNV() != 13) return 10;",
      "  nvl_heap_t *heap = nvl_open(\""+heapFileAbs+"\");", // frees 0
      "  nvl_set_root(heap, 0);", // frees 2
      "  if (nvlrt_get_numAllocNV() != 4) return 11;",
      "  if (nvlrt_get_numFreeNV() != 15) return 12;",
      // remove from vrefOnly and free
      "  nvl_alloc_nv(heap, 1, int);",
      "  if (nvlrt_get_numAllocNV() != 5) return 13;",
      "  if (nvlrt_get_numFreeNV() != 16) return 14;",
      "  nvl_close(heap);",
      "  if (nvlrt_get_numAllocNV() != 5) return 15;",
      "  if (nvlrt_get_numFreeNV() != 16) return 16;",
      "  return 0;",
      "}");
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    final LLVMModulePassManager pm = new LLVMModulePassManager();
    pm.addTargetData(exec.getTargetData());
    addNVLPasses(pm);
    pm.run(mod);
    mod.dump();
    checkIntFn(exec, mod, "vrefOnlyFreeAll", 0);
    exec.dispose();

    // Make sure we really are keeping the heap around after closing it
    // if vrefs still need decs. We might have gotten lucky the way we
    // tested it above. Below, we're at least checking that we mark the heap
    // closed.
    nvlOpenarcCCAndRun(
      "doubleClose", workDir, "",
      new String[]{
        "#include <nvl.h>",
        "#include <stddef.h>",
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <nvlrt-test.h>",
        "int main() {",
        "  nvl_heap_t *heap = nvl_open(\""+heapFileAbs+"\");",
        "  if (!heap) return 2;",
        "  nvl int *p = nvl_alloc_nv(heap, 1, int);", // inc vrefs
        "  nvl_close(heap);", // frees p but not heap because p needs dec
        "  if (nvlrt_get_numAllocNV() != 1) return 3;",
        "  if (nvlrt_get_numFreeNV() != 1) return 4;",
        // heap not freed so can complain heap is closed
        "  nvl_close(heap);",
        "  int r = !p;", // keep p live; unreachable
        "  return r;", // unreachable
        "}",
      },
      new String[]{
        "nvlrt-pmemobj: error: access to closed heap",
      },
      1);
  }

  // ------------
  // transactions

  // In the case of multi-dimensional arrays, make sure the full allocation
  // tx.add hoisting mechanism gets the correct size of the array. That is,
  // make sure it doesn't mix the innermost array's elementSize with the
  // outermost array's numElements.
  @Test public void txAddHoistAllocMultiDim() throws Exception {
    final File workDir = mkTmpDir("");
    final File heapFile = mkHeapFile("test.nvl");
    final String heapFileAbs = heapFile.getAbsolutePath();
    final String[] tx = new String[]{
      "#include <nvl.h>",
      "#include <stdlib.h>",
      "int inc = 1;",
      "nvl int (*p1)[5] = NULL;",
      "int main() {",
      "  nvl_heap_t *heap = nvl_create(\""+heapFileAbs+"\", 0, 0600);",
      "  if (!heap)",
      "    return 1;",
      "  nvl int (*p)[5] = nvl_alloc_nv(heap, 5, int[5]);",
         // This extra V-to-NV pointer (global so -mem2reg doesn't eliminate
         // it) suppresses a shadow update, which would cause writes to be
         // redirected to a new allocation, so there would be no change to
         // the allocation to undo, so we wouldn't know if the tx.add used
         // the wrong size.
      "  p1 = p;",
      "  nvl_set_root(heap, p);",
      "  for (int i = 0; i < 5; i += 1)",
      "    for (int j = 0; j < 5; j += 1)",
      "      if (p[i][j])",
      "        return 2;",
      "  #pragma nvl atomic heap(heap)",
      "  {",
           // Using inc not 1 ensures that the aggregated size of the tx.add
           // cannot be computed at compile time, so full allocation
           // hoisting must be used. Currently, the inner loop must use 1 or
           // we fail to hoist at all for some reason.
      "    for (int i = 0; i < 5; i += inc)",
      "      for (int j = 0; j < 5; j += 1)",
      "        p[i][j] = 99;",
           // Exit early so the transaction has to be rolled back.
      "    exit(3);",
      "  }",
      "  nvl_close(heap);",
      "  return 0;",
      "}"};
    nvlOpenarcCCAndRun("txAddHoistAllocMultiDim-tx", workDir, "", tx,
                       new String[0], 3);
    final String[] check = new String[]{
      "#include <nvl.h>",
      "int main() {",
      "  nvl_heap_t *heap = nvl_open(\""+heapFileAbs+"\");",
      "  if (!heap)",
      "    return 1;",
      "  nvl int (*p)[5] = nvl_get_root(heap, int[5]);",
      "  for (int i = 0; i < 5; i += 1)",
      "    for (int j = 0; j < 5; j += 1)",
      "      if (p[i][j])",
      "        return 2;",
      "  nvl_close(heap);",
      "  return 0;",
      "}"};
    nvlOpenarcCCAndRun("txAddHoistAllocMultiDim-check", workDir, "", check,
                       new String[0], 0);
  }

  @Test public void automaticShadowUpdates() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final File heapFile0 = mkHeapFile("0.nvl");
    final File heapFile1 = mkHeapFile("1.nvl");
    final String heapFile0Abs = heapFile0.getAbsolutePath();
    final String heapFile1Abs = heapFile1.getAbsolutePath();
    final SimpleResult simpleResult = buildLLVMSimple(
      targetTriple, targetDataLayout,
      "#include <nvl.h>",
      "#include <stdio.h>",
      "#include <nvlrt-test.h>",

      "nvl_heap_t *heap;",
      "int createHeap() {",
         // We want to make sure shadow updates happen when possible
         // (regardless of the predicted cost) and behave correctly.
      "  nvlrt_setShadowUpdateCostMode(NVLRT_COST_ZERO);",
         // We want to notice when a shadow update's new allocation doesn't
         // copy over the old allocation's data, and it's easier to detect
         // that when it has been zeroed at allocation time.
      "  nvlrt_zeroShadowUpdateAlloc();",
      "  heap = nvl_create(\""+heapFile0Abs+"\", 0, 0600);",
      "  if (!heap)",
      "    return 1;",
      "  return 0;",
      "}",
      "int openHeap() {",
      "  heap = nvl_open(\""+heapFile0Abs+"\");",
      "  if (!heap)",
      "    return 1;",
      "  return 0;",
      "}",
      "int closeHeap() {",
      "  nvl_close(heap);",
      "  return 0;",
      "}",

      "nvl int *vp[2] = {NULL, NULL};",

      // Basic cases with various combinations of V-to-NV pointers and
      // NV-to-NV pointers.
      "int simple() {",
         // Create reg V-to-NV pointer.
      "  nvl int *p = nvl_alloc_nv(heap, 1, int);",
      "  if (!p)",
      "    return 1;",
      "  if (*p != 0)",
      "    return 2;",

         // Shadow update: one reg V-to-NV pointer.
      "  void *pold = nvl_bare_hack(p);",
      "  #pragma nvl atomic heap(heap)",
      "  ++*p;",
      "  if (pold == nvl_bare_hack(p))",
      "    return 3;",
      "  if (*p != 1)",
      "    return 4;",

         // Shadow update: one reg V-to-NV pointer.
      "  pold = nvl_bare_hack(p);",
      "  #pragma nvl atomic heap(heap)",
      "  ++*p;",
      "  if (pold == nvl_bare_hack(p))",
      "    return 5;",
      "  if (*p != 2)",
      "    return 6;",

         // Create NV-to-NV pointer.
      "  nvl int * nvl *np = nvl_alloc_nv(heap, 2, nvl int *);",
      "  if (!np)",
      "    return 7;",
      "  np[0] = p;",

         // Shadow update: one reg V-to-NV and one NV-to-NV pointer.
      "  pold = nvl_bare_hack(p);",
      "  if (pold != nvl_bare_hack(np[0]))",
      "    return 8;",
      "  #pragma nvl atomic heap(heap)",
      "  ++*p;",
      "  if (pold == nvl_bare_hack(p))",
      "    return 9;",
      "  if (p != np[0])",
      "    return 10;",
      "  if (*p != 3)",
      "    return 11;",

         // No shadow update: two reg V-to-NV pointers and one NV-to-NV
         // pointer.
      "  pold = nvl_bare_hack(p);",
      "  #pragma nvl atomic heap(heap)",
      "  ++*np[0];", // this generates the second reg V-to-NV pointer
      "  if (pold != nvl_bare_hack(p))",
      "    return 12;",
      "  if (p != np[0])",
      "    return 13;",
      "  if (*p != 4)",
      "    return 14;",

         // Shadow update: one reg V-to-NV and one NV-to-NV pointer.
      "  pold = nvl_bare_hack(p);",
      "  if (pold != nvl_bare_hack(np[0]))",
      "    return 15;",
      "  #pragma nvl atomic heap(heap)",
      "  ++*p;",
      "  if (pold == nvl_bare_hack(p))",
      "    return 16;",
      "  if (p != np[0])",
      "    return 17;",
      "  if (*p != 5)",
      "    return 18;",

         // Create another NV-to-NV pointer.
      "  np[1] = p;",

         // No shadow update: one reg V-to-NV pointer and two NV-to-NV
         // pointers.
      "  pold = nvl_bare_hack(p);",
      "  if (pold != nvl_bare_hack(np[0]))",
      "    return 19;",
      "  if (pold != nvl_bare_hack(np[1]))",
      "    return 20;",
      "  #pragma nvl atomic heap(heap)",
      "  ++*p;",
      "  if (pold != nvl_bare_hack(p))",
      "    return 21;",
      "  if (p != np[0])",
      "    return 22;",
      "  if (p != np[1])",
      "    return 23;",
      "  if (*p != 6)",
      "    return 24;",

         // Clear one NV-to-NV pointer.
      "  np[0] = 0;",

         // No shadow update: one reg V-to-NV pointer and one NV-to-NV
         // pointer whose address was lost from nvrefs.
      "  pold = nvl_bare_hack(p);",
      "  if (pold != nvl_bare_hack(np[1]))",
      "    return 25;",
      "  #pragma nvl atomic heap(heap)",
      "  ++*p;",
      "  if (pold != nvl_bare_hack(p))",
      "    return 26;",
      "  if (p != np[1])",
      "    return 27;",
      "  if (*p != 7)",
      "    return 28;",

         // Clear the other NV-to-NV pointer.
      "  np[1] = 0;",

         // Create a mem V-to-NV pointer.
      "  vp[0] = p;",

         // Shadow update: one reg V-to-NV pointer and one mem V-to-NV
         // pointer.
      "  pold = nvl_bare_hack(p);",
      "  if (pold != nvl_bare_hack(vp[0]))",
      "    return 29;",
      "  #pragma nvl atomic heap(heap)",
      "  ++*p;",
      "  if (pold == nvl_bare_hack(p))",
      "    return 30;",
      "  if (p != vp[0])",
      "    return 31;",
      "  if (*p != 8)",
      "    return 32;",

         // No shadow update: two reg V-to-NV pointers and one mem V-to-NV
         // pointer.
      "  pold = nvl_bare_hack(p);",
      "  if (pold != nvl_bare_hack(vp[0]))",
      "    return 33;",
      "  #pragma nvl atomic heap(heap)",
      "  ++*vp[0];", // creates second reg V-to-NV pointer
      "  if (pold != nvl_bare_hack(p))",
      "    return 34;",
      "  if (p != vp[0])",
      "    return 35;",
      "  if (*p != 9)",
      "    return 36;",

         // Shadow update: one reg V-to-NV pointer and one mem V-to-NV
         // pointer.
      "  pold = nvl_bare_hack(p);", // p dies here
      "  if (pold != nvl_bare_hack(vp[0]))",
      "    return 37;",
      "  #pragma nvl atomic heap(heap)",
      "  ++*vp[0];", // creates reg V-to-NV pointer
      "  if (pold == nvl_bare_hack(vp[0]))",
      "    return 38;",
      "  if (*vp[0] != 10)",
      "    return 39;",

         // Create another mem V-to-NV pointer.
      "  vp[1] = vp[0];",

         // No shadow update: one reg V-to-NV pointer and two mem V-to-NV
         // pointers.
      "  pold = nvl_bare_hack(vp[0]);",
      "  if (pold != nvl_bare_hack(vp[1]))",
      "    return 40;",
      "  #pragma nvl atomic heap(heap)",
      "  ++*vp[0];", // creates reg V-to-NV pointer
      "  if (pold != nvl_bare_hack(vp[0]))",
      "    return 41;",
      "  if (vp[0] != vp[1])",
      "    return 42;",
      "  if (*vp[0] != 11)",
      "    return 43;",

         // Clear one mem V-to-NV pointer.
      "  vp[1] = 0;",

         // Shadow update: one reg V-to-NV pointer and one mem V-to-NV
         // pointer whose address was not lost.
      "  pold = nvl_bare_hack(vp[0]);",
      "  #pragma nvl atomic heap(heap)",
      "  ++*vp[0];", // creates reg V-to-NV pointer
      "  if (pold == nvl_bare_hack(vp[0]))",
      "    return 44;",
      "  if (*vp[0] != 12)",
      "    return 45;",

         // Swap mem V-to-NV pointers.
      "  vp[1] = vp[0];", // &vp[1] not recorded
      "  vp[0] = 0;", // &vp[0] removed

         // No shadow update: one reg V-to-NV pointer and one mem V-to-NV
         // pointer whose address was not recorded.
      "  pold = nvl_bare_hack(vp[1]);",
      "  #pragma nvl atomic heap(heap)",
      "  ++*vp[1];", // creates reg V-to-NV pointer
      "  if (pold != nvl_bare_hack(vp[1]))",
      "    return 46;",
      "  if (*vp[1] != 13)",
      "    return 47;",

         // Swap mem V-to-NV pointers.
      "  vp[0] = vp[1];", // &vp[0] recorded
      "  vp[1] = 0;",

         // Shadow update: one reg V-to-NV pointer and one mem V-to-NV
         // pointer.
      "  pold = nvl_bare_hack(vp[0]);",
      "  #pragma nvl atomic heap(heap)",
      "  ++*vp[0];", // creates reg V-to-NV pointer
      "  if (pold == nvl_bare_hack(vp[0]))",
      "    return 48;",
      "  if (*vp[0] != 14)",
      "    return 49;",

         // Create NV-to-NV pointer.
      "  np[1] = vp[0];",

         // Shadow update: one reg V-to-NV pointer, one mem V-to-NV pointer,
         // and one NV-to-NV pointer.
      "  pold = nvl_bare_hack(vp[0]);",
      "  if (pold != nvl_bare_hack(np[1]))",
      "    return 50;",
      "  #pragma nvl atomic heap(heap)",
      "  ++*vp[0];", // creates reg V-to-NV pointer
      "  if (pold == nvl_bare_hack(vp[0]))",
      "    return 51;",
      "  if (vp[0] != np[1])",
      "    return 52;",
      "  if (*vp[0] != 15)",
      "    return 53;",

      "  return 0;",
      "}",

      // Is the NV-to-NV pointer created by nvl_set_root updated?
      "int setRoot() {",
      "  nvl int *p = nvl_alloc_nv(heap, 1, int);",
      "  if (!p)",
      "    return 1;",
      "  if (*p != 0)",
      "    return 2;",
      "  nvl_set_root(heap, p);",
      "  void *pold = nvl_bare_hack(p);",
      "  if (pold != nvl_bare_hack(nvl_get_root(heap, int)))",
      "    return 3;",
         // Shadow update: one V-to-NV pointer and one NV-to-NV pointer.
      "  #pragma nvl atomic heap(heap)",
      "  ++*p;",
      "  if (pold == nvl_bare_hack(p))",
      "    return 4;",
      "  if (p != nvl_get_root(heap, int))",
      "    return 5;",
      "  if (*p != 1)",
      "    return 6;",
      "  return 0;",
      "}",

      // Make sure backup and backup_writeFirst clauses don't cause trouble.
      // First, the front end generates the tx.add calls in that case and
      // thus is responsible for declaring the tx.add prototype correctly.
      // Second, make sure the shadow update copies the necessary data.
      "#define N 10",
      "#pragma openarc #define N 10",
      "int backup() {",
      "  nvl int (*p)[N] = nvl_alloc_nv(heap, N, int[N]);",
      "  if (!p)",
      "    return 1;",
      "  for (int i = 0; i < N; ++i) {",
      "    for (int j = 0; j < N; ++j)",
      "      p[i][j] = 1;",
      "  }",
         // With backup, shadow update copies all data from the old
         // allocation.
      "  void *pold = nvl_bare_hack(p);",
      "  #pragma nvl atomic heap(heap) default(readonly) backup(p[1:N-2])",
      "  for (int i = 2; i < N-2; ++i) {",
      "    for (int j = 2; j < N-2; ++j)",
      "      p[i][j] = 2;",
      "  }",
      "  if (pold == nvl_bare_hack(p))",
      "    return 2;",
      "  for (int i = 0; i < N; ++i) {",
      "    for (int j = 0; j < N; ++j) {",
             // unwritten data
      "      if (i < 2 || N-2 <= i || j < 2 || N-2 <= j) {",
      "        if (p[i][j] != 1)",
      "          return 3;",
      "      }",
             // written data
      "      else if (p[i][j] != 2)",
      "        return 4;",
      "    }",
      "  }",
         // With backup_writeFirst, shadow update copies only the unlogged
         // data from the old allocation.
      "  pold = nvl_bare_hack(p);",
      "  #pragma nvl atomic heap(heap) default(readonly) \\",
      "    backup_writeFirst(p[1:N-2])",
      "  for (int i = 2; i < N-2; ++i) {",
      "    for (int j = 2; j < N-2; ++j)",
      "      p[i][j] = 3;",
      "  }",
      "  if (pold == nvl_bare_hack(p))",
      "    return 5;",
      "  for (int i = 0; i < N; ++i) {",
      "    for (int j = 0; j < N; ++j) {",
             // data that was unlogged and unwritten
      "      if (i < 1 || N-1 <= i) {",
      "        if (p[i][j] != 1)",
      "          return 6;",
      "      }",
             // oops: data that was logged but unwritten was left
             // zero-initialized in the new allocation (because we called
             // nvlrt_zeroShadowUpdateAlloc above). This is normally
             // undefined behavior because backup_writeFirst is being used
             // incorrectly. We exploit the behavior here anyway to check if
             // backup_writeFirst is truly skipping memcpy for logged data
             // and thus saving execution time as it's supposed to.
      "      else if (i < 2 || N-2 <= i || j < 2 || N-2 <= j) {",
      "        if (p[i][j] != 0)",
      "          return 7;",
      "      }",
             // data that was logged and written
      "      else if (p[i][j] != 3)",
      "        return i*1000 + j*100 + p[i][j];",
      "    }",
      "  }",

      "  return 0;",
      "}",

      // Check the case where there's an updateAlloc call on a phi's
      // operand(s) and a store on the phi's result. The if and else are
      // significantly different so that optimizations don't merge them and
      // thus eliminate the phi.
      "int checkPhi(int b) {",
      "  nvl int *p = 0;",
      "  void *pold;",
      "  if (b) {",
      "    p = nvl_alloc_nv(heap, 1, int);",
      "    if (!p) ",
      "      return 1;",
           // Shadow update: one V-to-NV pointer.
      "    pold = nvl_bare_hack(p);",
      "    #pragma nvl atomic heap(heap)",
      "    ++*p;",
      "  }",
      "  else {",
      "    p = nvl_alloc_nv(heap, 1, int);",
      "    if (!p)",
      "      return 2;",
      "    pold = 0;",
      "    ++*p;",
      "  }",
         // p's phi here is an alloc root. Hopefully p's phi appears before
         // pold's phi. If so, then NVLAddRefCounting has to be careful to
         // insert the store of p's alloc field after all phis not just
         // immediately after p's phi.
      "  if (nvl_bare_hack(p) == pold)",
      "    return 3;",
      "  if (*p != 1)",
      "    return 4;",
         // NVLAddRefCounting won't insert any store for p's phi if there's
         // no tx.add for it, so add one here. The previous tx.add doesn't
         // suffice because it's a different isoalloc set.
      "  pold = nvl_bare_hack(p);",
      "  #pragma nvl atomic heap(heap)",
      "  ++*p;",
      "  if (nvl_bare_hack(p) == pold)",
      "    return 5;",
      "  if (*p != 2)",
      "    return 6;",
      "  return 0;",
      "}",

      // Is the NV-to-NV pointer address stored in nvrefs correct when the
      // heap is reopened and mapped to a new address?
      "int nvrefsAfterReopen() {",
      "  nvl int *p = nvl_alloc_nv(heap, 1, int);",
      "  nvl_set_root(heap, p);",

         // Force old heap to map to a different address by closing it,
         // opening another heap that will likely take the old address, and
         // then reopening it. Failing to force a remap doesn't mean NVL-C
         // is broken. It means this test case isn't robust enough.
      "  void *pold = nvl_bare_hack(p);",
      "  int res;",
      "  if (res = closeHeap())",
      "    return 100+res;",
      "  nvl_heap_t *heap1 = nvl_create(\""+heapFile1Abs+"\", 0, 0600);",
      "  if (!heap1)",
      "    return 1;",
      "  if (res = openHeap())",
      "    return 200+res;",
      "  nvl_close(heap1);",
      "  p = nvl_get_root(heap, int);",
      "  if (!p)",
      "    return 2;",
      "  if (pold == nvl_bare_hack(p))",
      "    return 3;", // failed to force a remap

         // Now perform a shadow update with one V-to-NV pointer and one
         // NV-to-NV pointer (the root), and check if the latter is updated
         // properly (otherwise nvrefs was probably corrupted by the
         // reopen).
      "  pold = nvl_bare_hack(p);",
      "  #pragma nvl atomic heap(heap)",
      "  ++*p;",
      "  if (pold == nvl_bare_hack(p))",
      "    return 4;", // no shadow update
      "  if (p != nvl_get_root(heap, int))",
      "    return 5;", // NV-to-NV pointer was not updated properly
      "  if (*p != 1)",
      "    return 6;",
      "}");
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    final LLVMModulePassManager pm = new LLVMModulePassManager();
    pm.addTargetData(exec.getTargetData());
    addNVLPasses(pm);
    pm.run(mod);
    mod.dump();
    assumePmemobjLibs();
    checkIntFn(exec, mod, 0, "createHeap");
    checkIntFn(exec, mod, 0, "simple");
    checkIntFn(exec, mod, 0, "setRoot");
    checkIntFn(exec, mod, 0, "backup");
    checkIntFn(exec, mod, 0, "checkPhi", 1);
    checkIntFn(exec, mod, 0, "checkPhi", 0);
    checkIntFn(exec, mod, 0, "nvrefsAfterReopen");
    checkIntFn(exec, mod, 0, "closeHeap");
    exec.dispose();
  }

  @Test public void linkedList_implicitTxs() throws Exception {
    linkedListTxsCheck(false);
  }
  @Test public void linkedList_explicitTxs() throws Exception {
    linkedListTxsCheck(true);
  }
  private void linkedListTxsCheck(boolean checkExplicitTxs)
    throws Exception
  {
    // Randomly generate linked list operations to test.
    final StringBuilder ops = new StringBuilder();
    final Random rand = new Random();
    final int nops = 10;
    final int maxValue = nops/2;
    final LinkedList<String> expout = new LinkedList<>();
    for (int i = 0; i < nops; ++i) {
      final int value = rand.nextInt(maxValue)+1;
      if (rand.nextBoolean()) {
        ops.append("a");
        expout.addFirst(String.valueOf(value));
      }
      else {
        ops.append("d");
        while (expout.remove(String.valueOf(value))) ;
      }
      ops.append(value);
      ops.append(" ");
    }
    final String opsStr = ops.toString();
    System.err.println("operations: "+opsStr);
    System.err.print("expected list:");
    for (String expectedValue : expout)
      System.err.print(" "+expectedValue);
    System.err.println();
    expout.addFirst("rootReady");
    expout.addFirst("heapReady");

    // Without explicit transactions, we can't make incrementing nops and
    // adding a node part of the same transaction, so we might end up with
    // duplicate successive adds. Don't fail in that case.  Duplicate
    // deletes are also possible, but delete is idempotent.
    final File heapFile = mkHeapFile("test.nvl");
    final String heapFileAbs = heapFile.getAbsolutePath();
    txsCheck(
      "linkedList", nops, opsStr, expout, !checkExplicitTxs, heapFile,
      "#include <nvl.h>",
      "#include <stdio.h>",
      "#include <stdlib.h>",
      "struct list {",
      "  int value;",
      "  nvl struct list *next;",
      "};",
      "struct root {",
      "  int nops;",
      // list.value is unused
      // list.next is the list's head node, possibly NULL
      "  struct list list;",
      "};",
      "int add(nvl_heap_t *heap, nvl struct list *list, int k) {",
      "  nvl struct list *node = nvl_alloc_nv(heap, 1, struct list);",
      "  if (!node) {",
      "    perror(\"failed to allocate node\");",
      "    return 1;",
      "  }",
      "  node->value = k;",
      "  node->next = list->next;",
      "  list->next = node;",
      "  return 0;",
      "}",
      "void del(nvl struct list *list, int k) {",
      "  while (list->next != NULL) {",
      "    if (list->next->value == k)",
      "      list->next = list->next->next;",
      "    else",
      "      list = list->next;",
      "  }",
      "}",
      "void fprint(FILE *file, const char *indent,",
      "            nvl struct list *list)",
      "{",
      "  while (list->next != NULL) {",
      "    list = list->next;",
      "    fprintf(file, \"%s%d\\n\", indent, list->value);",
      "  }",
      "}",
      "int main(int argc, char *argv[]) {",
      "  nvl_heap_t *heap = nvl_recover(\""+heapFileAbs+"\", 0, 0600);",
      "  if (!heap) {",
      "    perror(\"failed to recover or create heap\");",
      "    return 1;",
      "  }",
      "  fprintf(stdout, \"heapReady\\n\");",
      "  fflush(stdout);",
      "  nvl struct root *root = nvl_get_root(heap, nvl struct root);",
      "  if (!root) {",
           // Default init from nvl_alloc_nv is sufficient here,
           // and each of nvl_alloc_nv and nvl_set_root are already
           // transactional, so no need for an explicit transaction here.
      "    root = nvl_alloc_nv(heap, 1, nvl struct root);",
      "    if (!root) {",
      "      perror(\"failed to allocate root\");",
      "      return 1;",
      "    }",
      "    nvl_set_root(heap, root);",
      "  }",
      "  fprintf(stdout, \"rootReady\\n\");",
      "  fflush(stdout);",
      "  int nops = argc-1;",
      "  if (root->nops < 0 || root->nops > nops) {",
      "    fprintf(stderr, \"NVM-stored nops is corrupt: %d\\n\",",
      "            root->nops);",
      "    return 1;",
      "  }",
      "  fprintf(stderr, \"current list:\\n\");",
      "  fprint(stderr, \"  \", &root->list);",
      "  fprintf(stderr, \"current nops: %d\\n\", root->nops);",
      "  fflush(stderr);",
      "  for (int i = root->nops; i < nops; ++i) {",
      (checkExplicitTxs ? "    #pragma nvl atomic heap(heap)" : ""),
      "    {",
      "      const char *arg = argv[i+1];",
      "      char op = arg[0];",
      "      int value = op ? atoi(arg+1) : 0;",
      "      if (op == 'a') {",
      "        fprintf(stderr, \"adding   %d...\\n\", value);",
      "        fflush(stderr);",
      "        if (add(heap, &root->list, value))",
      "          return 1;",
      "      }",
      "      else if (op == 'd') {",
      "        fprintf(stderr, \"deleting %d...\\n\", value);",
      "        fflush(stderr);",
      "        del(&root->list, value);",
      "      }",
      "      else {",
      "        fprintf(stderr,",
      "                \"unrecognized command-line argument: %s\",",
      "                arg);",
      "        return 1;",
      "      }",
      "      fprintf(stderr, \"incrementing nops...\\n\");",
      "      fflush(stderr);",
      "      ++root->nops;",
      "    }",
      "    fflush(stderr);",
      "  }",
      "  fprintf(stderr, \"all operations complete\\n\");",
      "  fflush(stderr);",
      "  fprint(stdout, \"\", &root->list);",
      "  nvl_close(heap);",
      "  return 0;",
      "}");
  }

  @Test public void matmul_1d_clobber() throws Exception {
    matmul("clobber", true, false);
  }
  @Test public void matmul_implicit2d_clobber() throws Exception {
    matmul("clobber", false, true);
  }
  @Test public void matmul_explicit2d_clobber() throws Exception {
    matmul("clobber", false, false);
  }
  @Test public void matmul_explicit2d_backup() throws Exception {
    matmul("backup", false, false);
  }
  @Test public void matmul_explicit2d_backup_writeFirst() throws Exception {
    matmul("backup_writeFirst", false, false);
  }
  // TODO: This test uses clobber but doesn't really check that the persist
  // calls clobber is supposed to generate have any effect.  That is, merely
  // killing a process doesn't prevent previously written data from reaching
  // NVM.  Would we need an actual power loss?
  private void matmul(String aClause, boolean flat, boolean implicit2d)
    throws Exception
  {
    final int rows = 128;
    final int rowsPerTx = 4;
    // Without this condition, the clobber clause and i_sub loop would have
    // to be adjusted for the last transaction.
    assertEquals("rows % rowsPerTx", 0, rows % rowsPerTx);
    final File heapFile = mkHeapFile("test.nvl");
    final String heapFileAbs = heapFile.getAbsolutePath();
    LinkedList<String> expout = new LinkedList<>();
    expout.add("heapReady");
    expout.add("rootReady");
    txsCheck(
      "matmul-"+(flat?"1d":(implicit2d?"implicit2d":"explicit2d")),
      rows/rowsPerTx, "", expout, false, heapFile,
      "#include <nvl.h>",
      "#include <stdio.h>",
      "#include <stdlib.h>",
      "#define _N_ "+rows,
      "#pragma openarc #define _N_ "+rows,
      "#define ROWS_PER_TX "+rowsPerTx,
      "#pragma openarc #define ROWS_PER_TX "+rowsPerTx,
      "#define N _N_",
      "#define M _N_",
      "#define P _N_",
      "#pragma openarc #define N _N_",
      "#pragma openarc #define M _N_",
      "#pragma openarc #define P _N_",
      "struct root {",
      "  int i;",
      "  float a[M"+(flat?"*":"][")+"N];",
      "  float b[M"+(flat?"*":"][")+"P];",
      "  float c[P"+(flat?"*":"][")+"N];",
      "};",
      "void mm_nv(nvl_heap_t *heap, nvl int *i_nv,",
      "           nvl float (*a_nv)"+(flat?"":"[N]")+",",
      "           nvl float (*b_nv)"+(flat?"":"[P]")+",",
      "           nvl float (*c_nv)"+(flat?"":"[N]")+")",
      "{",
      "  for (int i=*i_nv; i<M;) {",
      "    #pragma nvl atomic heap(heap) default(readonly) \\",
      "            backup(i_nv[0:1]) \\",
      "            "+(flat?aClause+"(a_nv[i*N:ROWS_PER_TX*N])"
                          :(aClause+"(a_nv[i:ROWS_PER_TX]"
                            +(implicit2d?"":"[0:N]")+")")),
      "    for (int i_sub=0; i_sub<ROWS_PER_TX; ++i_sub, ++i, ++*i_nv) {",
      "      for (int j=0; j<N; j++) {",
      "        float sum = 0.0;",
      "        for (int k=0; k<P; k++) {",
      "          sum +=   b_nv[i"+(flat?"*P+":"][")+"k]",
      "                 * c_nv[k"+(flat?"*N+":"][")+"j];",
      "        }",
      "        a_nv[i"+(flat?"*N+":"][")+"j] = sum;",
      "      }",
      "    }",
      "    fprintf(stderr, \"completed tx ending at i=%d...\\n\", i-1);",
      "    fflush(stderr);",
      "  }",
      "}",
      "int mm_verify(nvl float (*a_nv)"+(flat?"":"[N]")+",",
      "              float *b_v, float *c_v) {",
      "  for (int i=0; i<M; i++) {",
      "    for (int j=0; j<N; j++) {",
      "      float sum = 0.0;",
      "      for (int k=0; k<P; k++) {",
      "        sum += b_v[i*P+k]*c_v[k*N+j];",
      "      }",
      "      if (a_nv[i"+(flat?"*N+":"][")+"j] != sum) {",
      "        fprintf(stderr, \"verification failed at (%d,%d)\\n\",",
      "                i, j);",
      "        fprintf(stderr, \"expected = %g\\n\", sum);",
      "        fprintf(stderr, \"actual   = %g\\n\",",
      "                a_nv[i"+(flat?"*N+":"][")+"j]);",
      "        fflush(stderr);",
      "        return 1;",
      "      }",
      "    }",
      "  }",
      "  return 0;",
      "}",
      "float initB(int i, int j) { return i*P+j; }",
      "float initC(int i, int j) { return 1.0F; }",
      "int main(int argc, char *argv[]) {",
      "  nvl_heap_t *heap = nvl_recover(\""+heapFileAbs+"\", 0, 0600);",
      "  if (!heap) {",
      "    perror(\"failed to recover or create heap\");",
      "    return 1;",
      "  }",
      "  fprintf(stdout, \"heapReady\\n\");",
      "  fflush(stdout);",
      "  nvl struct root *root = nvl_get_root(heap, nvl struct root);",
      "  if (!root) {",
      "    #pragma nvl atomic heap(heap)",
      "    {",
      "      root = nvl_alloc_nv(heap, 1, nvl struct root);",
      "      if (!root) {",
      "        perror(\"failed to allocate root\");",
      "        return 1;",
      "      }",
      "      nvl_set_root(heap, root);",
      "      root->i = 0;",
      "      for (int i = 0; i < M; i++)",
      "        for (int j = 0; j < P; j++)",
      "          root->b[i"+(flat?"*P+":"][")+"j] = initB(i,j);",
      "      for (int i = 0; i < P; i++)",
      "        for (int j = 0; j < N; j++)",
      "          root->c[i"+(flat?"*N+":"][")+"j] = initC(i,j);",
      "    }",
      "  }",
      "  fprintf(stdout, \"rootReady\\n\");",
      "  fflush(stdout);",
      "  if (root->i < 0 || root->i > M) {",
      "    fprintf(stderr, \"NVM-stored i is corrupt: %d\\n\",",
      "            root->i);",
      "    return 1;",
      "  }",
      "  fprintf(stderr, \"current i: %d\\n\", root->i);",
      "  fflush(stderr);",
      "  float *a_v, *b_v, *c_v;",
      "  if (!(a_v = (float *) malloc(M*N*sizeof(float)))",
      "      || !(b_v = (float *) malloc(M*P*sizeof(float)))",
      "      || !(c_v = (float *) malloc(P*N*sizeof(float))))",
      "  {",
      "    perror(\"malloc failed\");",
      "    return 1;",
      "  }",
      "  for (int i = 0; i < M; i++)",
      "    for (int j = 0; j < P; j++)",
      "      b_v[i*P+j] = initB(i,j);",
      "  for (int i = 0; i < P; i++)",
      "    for (int j = 0; j < N; j++)",
      "      c_v[i*N+j] = initC(i,j);",
      "  mm_nv(heap, &root->i, root->a, root->b, root->c);",
      "  fprintf(stderr, \"matmul complete\\n\");",
      "  fflush(stderr);",
      "  if (mm_verify(root->a, b_v, c_v))",
      "    return 1;",
      "  fprintf(stderr, \"verification successful\\n\");",
      "  fflush(stderr);",
      "  nvl_close(heap);",
      "  return 0;",
      "}");
  }

  @Test public void jacobi() throws Exception {
    final int n = 128;
    final int iters = 1000;
    final int itersPerTx = 100;
    // Without this condition, the k_sub loop would have to be adjusted for
    // the last transaction.
    assertEquals("iters % itersPerTx", 0, iters % itersPerTx);
    final File heapFile = mkHeapFile("test.nvl");
    final String heapFileAbs = heapFile.getAbsolutePath();
    LinkedList<String> expout = new LinkedList<>();
    expout.add("heapReady");
    expout.add("rootReady");
    txsCheck(
      "jacobi", iters/itersPerTx, "", expout, false, heapFile,
      "#include <nvl.h>",
      "#include <stdio.h>",
      "#include <stdlib.h>",
      "#define N "+n,
      "#pragma openarc #define N "+n,
      "#define ITERS "+iters,
      "#define ITERS_PER_TX "+itersPerTx,
      "struct root {",
      "  int k;",
      "  nvl float (*a)[N];",
      "};",
      "int jacobi_nv(nvl_heap_t *heap, nvl int *k, nvl float (*a)[N]) {",
      "  float (*tmp)[N] = malloc(sizeof(float) * N * N);",
      "  if (!tmp) {",
      "    perror(\"malloc failed\");",
      "    return 1;",
      "  }",
      "  while (*k<ITERS) {",
      "    #pragma nvl atomic heap(heap)",
      "    for (int k_sub=0; k_sub<ITERS_PER_TX; ++k_sub, ++*k) {",
      "      for (int i=1; i<N-1; ++i) {",
      "        for (int j=1; j<N-1; ++j)",
      "          tmp[i][j] = (  a[i-1][j] + a[i+1][j]",
      "                       + a[i][j-1] + a[i][j+1]) / 4.0f;",
      "      }",
      "      for (int i=1; i<N-1; ++i) {",
      "        for (int j=1; j<N-1; ++j)",
      "          a[i][j] = tmp[i][j];",
      "      }",
      "    }",
      "    fprintf(stderr, \"completed tx ending at k=%d...\\n\", (*k)-1);",
      "    fflush(stderr);",
      "  }",
      "  return 0;",
      "}",
      "int jacobi_verify(nvl float (*a)[N]) {",
      "  float (*a_v)[N];",
      "  float (*tmp)[N];",
      "  if (   !(a_v = malloc(sizeof(float) * N * N))",
      "      || !(tmp = malloc(sizeof(float) * N * N)))",
      "  {",
      "    perror(\"malloc failed\");",
      "    return 1;",
      "  }",
      "  for (int i = 0; i < N; ++i)",
      "    a_v[0][i] = 1;",
      "  for (int i = 1; i < N-1; ++i) {",
      "    a_v[i][0] = 1;",
      "    for (int j = 1; j < N-1; ++j)",
      "      a_v[i][j] = 0;",
      "    a_v[i][N-1] = 1;",
      "  }",
      "  for (int i = 0; i < N; ++i)",
      "    a_v[N-1][i] = 1;",
      "  for (int k = 0; k<ITERS; ++k) {",
      "    for (int i=1; i<N-1; ++i) {",
      "      for (int j=1; j<N-1; ++j)",
      "        tmp[i][j] = (  a_v[i-1][j] + a_v[i+1][j]",
      "                     + a_v[i][j-1] + a_v[i][j+1]) / 4.0f;",
      "    }",
      "    for (int i=1; i<N-1; ++i) {",
      "      for (int j=1; j<N-1; ++j)",
      "        a_v[i][j] = tmp[i][j];",
      "    }",
      "  }",
      "  for (int i=0; i<N; i++) {",
      "    for (int j=0; j<N; j++) {",
      "      if (a[i][j] != a_v[i][j]) {",
      "        fprintf(stderr, \"verification failed at (%d,%d)\\n\",",
      "                i, j);",
      "        fprintf(stderr, \"expected = %g\\n\", a_v[i][j]);",
      "        fprintf(stderr, \"actual   = %g\\n\", a[i][j]);",
      "        return 1;",
      "      }",
      "    }",
      "  }",
      "  return 0;",
      "}",
      "int main(int argc, char *argv[]) {",
      "  nvl_heap_t *heap = nvl_recover(\""+heapFileAbs+"\", 0, 0600);",
      "  if (!heap) {",
      "    perror(\"failed to recover or create heap\");",
      "    return 1;",
      "  }",
      "  fprintf(stdout, \"heapReady\\n\");",
      "  fflush(stdout);",
      "  nvl struct root *root = nvl_get_root(heap, nvl struct root);",
      "  if (!root) {",
      "    #pragma nvl atomic heap(heap)",
      "    {",
      "      root = nvl_alloc_nv(heap, 1, nvl struct root);",
      "      if (!root) {",
      "        perror(\"failed to allocate root\");",
      "        return 1;",
      "      }",
      "      nvl_set_root(heap, root);",
      "      root->k = 0;",
      "      root->a = nvl_alloc_nv(heap, N, nvl float[N]);",
      "      if (!root->a) {",
      "        perror(\"failed to allocate array\");",
      "        return 1;",
      "      }",
      "      for (int i = 0; i < N; ++i)",
      "        root->a[0][i] = 1;",
      "      for (int i = 1; i < N-1; ++i) {",
      "        root->a[i][0] = 1;",
      "        root->a[i][N-1] = 1;",
      "      }",
      "      for (int i = 0; i < N; ++i)",
      "        root->a[N-1][i] = 1;",
      "    }",
      "  }",
      "  fprintf(stdout, \"rootReady\\n\");",
      "  fflush(stdout);",
      "  if (root->k < 0 || root->k > ITERS) {",
      "    fprintf(stderr, \"NVM-stored k is corrupt: %d\\n\",",
      "            root->k);",
      "    return 1;",
      "  }",
      "  fprintf(stderr, \"current k: %d\\n\", root->k);",
      "  fflush(stderr);",
      "  if (jacobi_nv(heap, &root->k, root->a))",
      "    return 1;",
      "  fprintf(stderr, \"jacobi complete\\n\");",
      "  fflush(stderr);",
      "  if (jacobi_verify(root->a))",
      "    return 1;",
      "  fprintf(stderr, \"verification successful\\n\");",
      "  fflush(stderr);",
      "  nvl_close(heap);",
      "  return 0;",
      "}");
  }

  // -----------------------------------------------
  // MPI group transactions: handling MPI_Group type

  @Test public void fnMpiGroupTypeMissing() throws Exception {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "at argument 4 to __builtin_nvl_create_mpi, unknown typedef name:"
      +" MPI_Group"));
    buildLLVMSimple(
      "", "",
      "#include <nvl.h>",
      "void foo() {",
      "  nvl_create_mpi(\"foo.nvl\", 0, 0600, 0);",
      "}");
  }

  @Test public void clauseMpiGroupTypeMissing() throws Exception {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "at mpiGroup clause to nvl atomic pragma, unknown typedef name:"
      +" MPI_Group"));
    buildLLVMSimple(
      "", "",
      "#include <nvl.h>",
      "nvl_heap_t *heap = NULL;",
      "void foo() {",
      "  #pragma nvl atomic heap(heap) mpiGroup(3)",
      "  0;",
      "}");
  }

  @Test public void fnMpiGroupTypeNonIntegerPtr() throws Exception {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "at call to __builtin_nvl_recover_mpi, MPI_Group is a typedef to a"
      +" non-integral, non-pointer type: float"));
    buildLLVMSimple(
      "", "",
      "#include <nvl.h>",
      "typedef float MPI_Group;",
      "typedef struct t *T;",
      "void foo() {",
      "  T group;",
      "  nvl_recover_mpi(\"foo.nvl\", 0, 0600, group);",
      "}");
  }

  @Test public void clauseMpiGroupTypeNonIntegerPtr() throws Exception {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "at call to nvl atomic pragma, MPI_Group is a typedef to a"
      +" non-integral, non-pointer type: struct foo {int i;}"));
    buildLLVMSimple(
      "", "",
      "#include <nvl.h>",
      "typedef struct foo {int i;} MPI_Group;",
      "nvl_heap_t *heap = NULL;",
      "typedef struct t *T;",
      "void foo() {",
      "  T group;",
      "  #pragma nvl atomic heap(heap) mpiGroup(group)",
      "  0;",
      "}");
  }

  @Test public void mpiGroupTypeInt() throws Exception {
    buildLLVMSimple(
      "", "",
      "#include <nvl.h>",
      "typedef int MPI_Group;",
      "nvl_heap_t *heap = NULL;",
      "void foo() {",
      "  MPI_Group group;",
      "  nvl_create_mpi(\"foo.nvl\", 0, 0600, group);",
      "  nvl_recover_mpi(\"foo.nvl\", 0, 0600, group);",
      "  nvl_open_mpi(\"foo.nvl\", group);",
      "  #pragma nvl atomic heap(heap) mpiGroup(group)",
      "  0;",
      "}");
  }

  @Test public void mpiGroupTypePtr() throws Exception {
    buildLLVMSimple(
      "", "",
      "#include <nvl.h>",
      "typedef void *MPI_Group;",
      "nvl_heap_t *heap = NULL;",
      "void foo() {",
      "  MPI_Group group;",
      "  nvl_create_mpi(\"foo.nvl\", 0, 0600, group);",
      "  nvl_recover_mpi(\"foo.nvl\", 0, 0600, group);",
      "  nvl_open_mpi(\"foo.nvl\", group);",
      "  #pragma nvl atomic heap(heap) mpiGroup(group)",
      "  0;",
      "}");
  }

  @Test public void mpiGroupTypeFromHeader() throws Exception {
    buildLLVMSimple(
      "", "",
      "#include <nvl.h>",
      "#include <"+assumeConfig("mpi_includes")+"/mpi.h>",
      "nvl_heap_t *heap = NULL;",
      "void foo() {",
      "  MPI_Group group;",
      "  nvl_create_mpi(\"foo.nvl\", 0, 0600, group);",
      "  nvl_recover_mpi(\"foo.nvl\", 0, 0600, group);",
      "  nvl_open_mpi(\"foo.nvl\", group);",
      "  #pragma nvl atomic heap(heap) mpiGroup(group)",
      "  0;",
      "}");
  }

  @Test public void mpiGroupNullPtrConst() throws Exception {
    buildLLVMSimple(
      "", "",
      "#include <nvl.h>",
      "typedef void *MPI_Group;",
      "nvl_heap_t *heap = NULL;",
      "void foo() {",
      "  nvl_create_mpi(\"foo.nvl\", 0, 0600, 0);",
      "  nvl_recover_mpi(\"foo.nvl\", 0, 0600, 0);",
      "  nvl_open_mpi(\"foo.nvl\", 0);",
      "  nvl_create_mpi(\"foo.nvl\", 0, 0600, (void*)0);",
      "  nvl_recover_mpi(\"foo.nvl\", 0, 0600, (void*)0);",
      "  nvl_open_mpi(\"foo.nvl\", (void*)0);",
      "  nvl_create_mpi(\"foo.nvl\", 0, 0600, NULL);",
      "  nvl_recover_mpi(\"foo.nvl\", 0, 0600, NULL);",
      "  nvl_open_mpi(\"foo.nvl\", NULL);",
      // OpenARC cannot handle NULL or (void*)0 in the mpiGroup clause, so
      // we check only 0.
      "  #pragma nvl atomic heap(heap) mpiGroup(0)",
      "  0;",
      "}");
  }

  // Conceivably, MPI_Group might be a pointer or integer type, depending on
  // the MPI implementation. Either way, 0 should compile OK as a handle.
  @Test public void mpiGroupTypeFromHeaderZeroConst() throws Exception {
    buildLLVMSimple(
      "", "",
      "#include <nvl.h>",
      "#include <"+assumeConfig("mpi_includes")+"/mpi.h>",
      "nvl_heap_t *heap = NULL;",
      "void foo() {",
      "  nvl_create_mpi(\"foo.nvl\", 0, 0600, 0);",
      "  nvl_recover_mpi(\"foo.nvl\", 0, 0600, 0);",
      "  nvl_open_mpi(\"foo.nvl\", 0);",
      "  #pragma nvl atomic heap(heap) mpiGroup(0)",
      "  0;",
      "}");
  }

  @Test public void fnMpiGroupNotTypedef() throws Exception {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "warning treated as error: argument 2 to \"__builtin_nvl_open_mpi\""
      +" requires conversion between pointer types with incompatible target"
      +" types without explicit cast: integer type is incompatible with"
      +" non-integer type: struct __builtin_nvl_heap"));
    buildLLVMSimple(
      "", "",
      "#include <nvl.h>",
      "typedef int *MPI_Group;",
      "nvl_heap_t *heap = NULL;",
      "void foo() {",
      "  nvl_open_mpi(\"foo.nvl\", heap);",
      "}");
  }

  @Test public void clauseMpiGroupNotTypedef() throws Exception {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "warning treated as error: mpiGroup clause to \"nvl atomic pragma\""
      +" requires conversion between pointer type and arithmetic type"
      +" without explicit cast"));
    buildLLVMSimple(
      "", "",
      "#include <nvl.h>",
      "typedef int *MPI_Group;",
      "nvl_heap_t *heap = NULL;",
      "void foo() {",
      "  #pragma nvl atomic heap(heap) mpiGroup(3)",
      "  0;",
      "}");
  }

  @Test public void fnMpiGroupWrongTypedef() throws Exception {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "warning treated as error: argument 4 to \"__builtin_nvl_create_mpi\""
      +" requires conversion between pointer type and arithmetic type"
      +" without explicit cast"));
    buildLLVMSimple(
      "", "",
      "#include <nvl.h>",
      "typedef struct t *MPI_Group;",
      "typedef long T;",
      "void foo() {",
      "  T group;",
      "  nvl_create_mpi(\"foo.nvl\", 0, 0600, group);",
      "}");
  }

  @Test public void clauseMpiGroupWrongTypedef() throws Exception {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "warning treated as error: mpiGroup clause to \"nvl atomic pragma\""
      +" requires conversion between pointer type and arithmetic type"
      +" without explicit cast"));
    buildLLVMSimple(
      "", "",
      "#include <nvl.h>",
      "typedef long MPI_Group;",
      "nvl_heap_t *heap = NULL;",
      "typedef struct t *T;",
      "void foo() {",
      "  T group;",
      "  #pragma nvl atomic heap(heap) mpiGroup(group)",
      "  0;",
      "}");
  }

  // ------------------------------------------------------
  // MPI group transactions: create/open vs. file existence

  // This used to cause a seg fault.
  @Test public void createMpiClobbers() throws Exception {
    final File heapFile = mkHeapFile("test.nvl");
    final String heapFileAbs = heapFile.getAbsolutePath();
    final File workDir = mkTmpDir("");
    nvlOpenarcCCAndRun(
      "test", workDir, "", true, 1,
      new String[]{
        "#include <mpi.h>",
        "#include <nvl.h>",
        "int main(int argc, char *argv[]) {",
        "  MPI_Init(&argc, &argv);",
        "  MPI_Group group;",
        "  MPI_Comm_group(MPI_COMM_WORLD, &group);",
        "  for (int i=0; i<2; ++i) {",
        "    nvl_heap_t *heap = nvl_create_mpi(\""+heapFileAbs+"\", 0,",
        "                                      0600, group);",
        "    if (!heap)",
        "      return i+2;",
        "    nvl_close(heap);",
        "  }",
        "  MPI_Finalize();",
        "  return 0;",
        "}"},
      new String[0], false, 3);
  }

  @Test public void openMpiNoFile() throws Exception {
    final File heapFile = mkHeapFile("test.nvl");
    final String heapFileAbs = heapFile.getAbsolutePath();
    final File workDir = mkTmpDir("");
    nvlOpenarcCCAndRun(
      "test", workDir, "", true, 1,
      new String[]{
        "#include <mpi.h>",
        "#include <nvl.h>",
        "int main(int argc, char *argv[]) {",
        "  MPI_Init(&argc, &argv);",
        "  MPI_Group group;",
        "  MPI_Comm_group(MPI_COMM_WORLD, &group);",
        "  nvl_heap_t *heap = nvl_open_mpi(\""+heapFileAbs+"\", group);",
        "  if (!heap)",
        "    return 2;",
        "  nvl_close(heap);",
        "  MPI_Finalize();",
        "  return 0;",
        "}"},
      new String[0], false, 2);
  }

  // ---------------------------------------
  // MPI group transactions: tx mode changes

  @Test public void txModeMpiGroupButOpenLocal() throws Exception {
    final File heapFile = mkHeapFile("test.nvl");
    final String heapFileAbs = heapFile.getAbsolutePath();
    final File workDir = mkTmpDir("");
    nvlOpenarcCCAndRun(
      "create", workDir, "", true, 1,
      new String[]{
        "#include <mpi.h>",
        "#include <nvl.h>",
        "int main(int argc, char *argv[]) {",
        "  MPI_Init(&argc, &argv);",
        "  MPI_Group group;",
        "  MPI_Comm_group(MPI_COMM_WORLD, &group);",
        "  nvl_heap_t *heap = nvl_create_mpi(\""+heapFileAbs+"\", 0,",
        "                                    0600, group);",
        "  if (!heap)",
        "    return 1;",
        "  MPI_Finalize();",
        "  return 0;",
        "}"},
      new String[0], false, 0);
    nvlOpenarcCCAndRun(
      "open", workDir, "", true, 1,
      new String[]{
        "#include <nvl.h>",
        "int main(int argc, char *argv[]) {",
        "  nvl_open(\""+heapFileAbs+"\");",
        "  return 0;",
        "}"},
      new String[]{
        "nvlrt-pmemobj: error: opening heap in local transaction mode when"
        +" previously in MPI group transaction mode"},
      false, 1);
    nvlOpenarcCCAndRun(
      "recover", workDir, "", true, 1,
      new String[]{
        "#include <nvl.h>",
        "int main(int argc, char *argv[]) {",
        "  nvl_recover(\""+heapFileAbs+"\", 0, 0600);",
        "  return 0;",
        "}"},
      new String[]{
        "nvlrt-pmemobj: error: opening heap in local transaction mode when"
        +" previously in MPI group transaction mode"},
      false, 1);
    nvlOpenarcCCAndRun(
      "open-mpi-group-null", workDir, "", true, 1,
      new String[]{
        "#include <mpi.h>",
        "#include <nvl.h>",
        "int main(int argc, char *argv[]) {",
        "  nvl_open_mpi(\""+heapFileAbs+"\", MPI_GROUP_NULL);",
        "  return 0;",
        "}"},
      new String[]{
        "nvlrt-pmemobj: error: opening heap in local transaction mode when"
        +" previously in MPI group transaction mode"},
      false, 1);
    nvlOpenarcCCAndRun(
      "recover-mpi-group-null", workDir, "", true, 1,
      new String[]{
        "#include <mpi.h>",
        "#include <nvl.h>",
        "int main(int argc, char *argv[]) {",
        "  nvl_recover_mpi(\""+heapFileAbs+"\", 0, 0600, MPI_GROUP_NULL);",
        "  return 0;",
        "}"},
      new String[]{
        "nvlrt-pmemobj: error: opening heap in local transaction mode when"
        +" previously in MPI group transaction mode"},
      false, 1);
  }

  @Test public void txModeLocalButOpenMpiGroup() throws Exception {
    final File heapFile0 = mkHeapFile("test0.nvl");
    final File heapFile1 = mkHeapFile("test1.nvl");
    final String heapFileAbs0 = heapFile0.getAbsolutePath();
    final String heapFileAbs1 = heapFile1.getAbsolutePath();
    final File workDir = mkTmpDir("");
    final File createExe = nvlOpenarcCC(
      "create", workDir, "", true, true,
      new String[]{
        "#include <mpi.h>",
        "#include <nvl.h>",
        "int main(int argc, char *argv[]) {",
        "  MPI_Init(&argc, &argv);",
        "  MPI_Group group;",
        "  int rank;",
        "  MPI_Comm_rank(MPI_COMM_WORLD, &rank);",
        "  const char *file = !rank ? \""+heapFileAbs0+"\"",
        "                           : \""+heapFileAbs1+"\";",
        "  nvl_heap_t *heap = nvl_create(file, 0, 0600);",
        "  if (!heap)",
        "    return 1;",
        "  nvl_close(heap);",
        "  MPI_Finalize();",
        "  return 0;",
        "}"});

    // Create and close in local mode, and then open in group mode.
    nvlRunExe(createExe, workDir, "", -1, 2, new String[0], false, 0);
    nvlOpenarcCCAndRun(
      "open", workDir, "", true, 2,
      new String[]{
        "#include <mpi.h>",
        "#include <nvl.h>",
        "int main(int argc, char *argv[]) {",
        "  MPI_Init(&argc, &argv);",
        "  MPI_Group group;",
        "  MPI_Comm_group(MPI_COMM_WORLD, &group);",
        "  int rank;",
        "  MPI_Comm_rank(MPI_COMM_WORLD, &rank);",
        "  const char *file = !rank ? \""+heapFileAbs0+"\"",
        "                           : \""+heapFileAbs1+"\";",
        "  nvl_heap_t *heap = nvl_open_mpi(file, group);",
        "  if (!heap)",
        "    return 1;",
        "  nvl_close(heap);",
        "  MPI_Finalize();",
        "  return 0;",
        "}"},
      new String[0], false, 0);
    heapFile0.delete();
    heapFile1.delete();

    // Repeat, but use recover instead of open. To prepare for the next
    // check, advance the tx counts with a tx, close only rank 0's heap,
    // thus leaving rank 1's heap in group mode.
    nvlRunExe(createExe, workDir, "", -1, 2, new String[0], false, 0);
    nvlOpenarcCCAndRun(
      "recover", workDir, "", true, 2,
      new String[]{
        "#include <mpi.h>",
        "#include <nvl.h>",
        "int main(int argc, char *argv[]) {",
        "  MPI_Init(&argc, &argv);",
        "  MPI_Group group;",
        "  MPI_Comm_group(MPI_COMM_WORLD, &group);",
        "  int rank;",
        "  MPI_Comm_rank(MPI_COMM_WORLD, &rank);",
        "  const char *file = !rank ? \""+heapFileAbs0+"\"",
        "                           : \""+heapFileAbs1+"\";",
        "  nvl_heap_t *heap = nvl_recover_mpi(file, 0, 0600, group);",
        "  if (!heap)",
        "    return 1;",
        "  #pragma nvl atomic heap(heap) mpiGroup(group)",
        "    1;", // 1 because BuildLLVM currently ignores null statement
        "  if (rank == 0)",
        "    nvl_close(heap);",
        "  MPI_Finalize();",
        "  return 0;",
        "}"},
        new String[]{
          "nvlrt-pmemobj: warning: "+heapFileAbs0+": recovered heap",
          "nvlrt-pmemobj: warning: "+heapFileAbs1+": recovered heap"},
        false, 0);

    // Open in group mode again. If rank 0, which was in local mode, doesn't
    // detect that rank 1 was in group mode, then only rank 0 resets its tx
    // count to 0. The tx counts will then be out of sync when the next tx
    // fails and recovers.
    nvlOpenarcCCAndRun(
      "re-open-fail", workDir, "", true, 2,
      new String[]{
        "#include <mpi.h>",
        "#include <nvl.h>",
        "#include <unistd.h>",
        "int main(int argc, char *argv[]) {",
        "  MPI_Init(&argc, &argv);",
        "  MPI_Group group;",
        "  MPI_Comm_group(MPI_COMM_WORLD, &group);",
        "  int rank;",
        "  MPI_Comm_rank(MPI_COMM_WORLD, &rank);",
        "  const char *file = !rank ? \""+heapFileAbs0+"\"",
        "                           : \""+heapFileAbs1+"\";",
        "  nvl_heap_t *heap = nvl_open_mpi(file, group);",
        "  if (!heap)",
        "    return 1;",
        "  nvl int *root = nvl_alloc_nv(heap, 1, int);",
        "  if (!root)",
        "    return 1;",
        "  #pragma nvl atomic heap(heap) mpiGroup(group)",
        "  {",
        "    nvl_set_root(heap, root);",
        "    if (rank == 1) {",
        "      sleep(1);", // give rank 0 a chance to sync NVM
        "      MPI_Abort(MPI_COMM_WORLD, 2);",
        "    }",
        "  }",
        "  MPI_Finalize();", // should be unreachable because of barrier
        "  return 3;",
        "}"},
      new String[0], false, 2);

    // Now try to recover from transaction failure. If rank 0 erroneously
    // reset its tx count to 0 as described above but rank 1 didn't, then
    // rank 0 should think rank 1 passed the barrier from the most recent
    // transaction, so rank 0 should not roll back the transction even
    // though rank 1 does roll it back. Rank 0 probably had time to sync to
    // NVM, so we should then see that only rank 0 has its root set. That
    // is, rank 0 then returns 10 and rank 1 returns 0. If rank 0 did not
    // erroneously reset its tx count to 0 as described above, then both
    // ranks return 0.
    nvlOpenarcCCAndRun(
      "out-of-sync", workDir, "", true, 2,
      new String[]{
        "#include <mpi.h>",
        "#include <nvl.h>",
        "int main(int argc, char *argv[]) {",
        "  MPI_Init(&argc, &argv);",
        "  MPI_Group group;",
        "  MPI_Comm_group(MPI_COMM_WORLD, &group);",
        "  int rank;",
        "  MPI_Comm_rank(MPI_COMM_WORLD, &rank);",
        "  const char *file = !rank ? \""+heapFileAbs0+"\"",
        "                           : \""+heapFileAbs1+"\";",
        "  nvl_heap_t *heap = nvl_open_mpi(file, group);",
        "  if (!heap)",
        "    return 1;",
        "  if (nvl_get_root(heap, int))",
        "    return 10+rank;",
        "  nvl_close(heap);",
        "  MPI_Finalize();",
        "  return 0;",
        "}"},
      new String[0], false, 0);
  }

  // ---------------------------------------
  // MPI group transactions: tx world errors

  @Test public void mpiTxWorldNotReflexive() throws Exception {
    final File heapFile = mkHeapFile("test.nvl");
    final String heapFileAbs = heapFile.getAbsolutePath();
    final File workDir = mkTmpDir("");
    final String[] src = new String[]{
      "#include <mpi.h>",
      "#include <nvl.h>",
      "int main(int argc, char *argv[]) {",
      "  MPI_Init(&argc, &argv);",
      "  int rank;",
      "  MPI_Comm_rank(MPI_COMM_WORLD, &rank);",
      "  MPI_Group group1, group2;",
      "  MPI_Comm_group(MPI_COMM_WORLD, &group1);",
      "  MPI_Group_excl(group1, 1, &rank, &group2);",
      "  if (rank == 0) {",
      "    nvl_heap_t *heap = nvl_create_mpi(\""+heapFileAbs+"\", 0, 0600,",
      "                                      group2);",
      "    if (!heap)",
      "      return 1;",
      "    nvl_close(heap);",
      "  }",
      "  MPI_Finalize();",
      "  return 0;",
      "}"};
    final File exe = nvlOpenarcCC("create", workDir, "", true, true, src);
    // empty
    nvlRunExe(exe, workDir, "", -1, 1, new String[]{
      "nvlrt-pmemobj: error: new MPI transaction world does not include the"
      +" process's own MPI rank"},
      false, 1);
    heapFile.delete();
    // non-empty
    nvlRunExe(exe, workDir, "", -1, 2, new String[]{
      "nvlrt-pmemobj: error: new MPI transaction world does not include the"
      +" process's own MPI rank"},
      false, 1);
  }

  @Test public void mpiRankChange() throws Exception {
    final File heapFile0 = mkHeapFile("test0.nvl");
    final File heapFile1 = mkHeapFile("test1.nvl");
    final String heapFileAbs0 = heapFile0.getAbsolutePath();
    final String heapFileAbs1 = heapFile1.getAbsolutePath();
    final File workDir = mkTmpDir("");
    nvlOpenarcCCAndRun(
      "create", workDir, "", true, 2,
      new String[]{
        "#include <mpi.h>",
        "#include <nvl.h>",
        "int main(int argc, char *argv[]) {",
        "  MPI_Init(&argc, &argv);",
        "  MPI_Group group;",
        "  MPI_Comm_group(MPI_COMM_WORLD, &group);",
        "  int rank;",
        "  MPI_Comm_rank(MPI_COMM_WORLD, &rank);",
        "  const char *file;",
        "  switch (rank) {",
        "  case 0: file = \""+heapFileAbs0+"\"; break;",
        "  case 1: file = \""+heapFileAbs1+"\"; break;",
        "  default: return 2;",
        "  }",
        "  nvl_heap_t *heap = nvl_create_mpi(file, 0, 0600, group);",
        "  if (!heap)",
        "    return 3;",
        "  return 4;",
        "}"},
      new String[0], false, 4);
    nvlOpenarcCCAndRun(
      "open", workDir, "", true, 2,
      new String[]{
        "#include <mpi.h>",
        "#include <nvl.h>",
        "int main(int argc, char *argv[]) {",
        "  MPI_Init(&argc, &argv);",
        "  MPI_Group group;",
        "  MPI_Comm_group(MPI_COMM_WORLD, &group);",
        "  int rank;",
        "  MPI_Comm_rank(MPI_COMM_WORLD, &rank);",
        "  const char *file;",
        "  switch (rank) {",
        "  case 0: file = \""+heapFileAbs1+"\"; break;",
        "  case 1: file = \""+heapFileAbs0+"\"; break;",
        "  default: return 2;",
        "  }",
        "  nvl_heap_t *heap = nvl_open_mpi(file, group);",
        "  if (!heap)",
        "    return 3;",
        "  nvl_close(heap);",
        "  MPI_Finalize();",
        "  return 0;",
        "}"},
      new String[]{
        "nvlrt-pmemobj: error: opening heap with a different MPI_COMM_WORLD"
        +" rank: old=0, new=1",
        "nvlrt-pmemobj: error: opening heap with a different MPI_COMM_WORLD"
        +" rank: old=1, new=0"},
      true, 1);
  }

  @Test public void mpiTxWorldChange() throws Exception {
    final File heapFile0 = mkHeapFile("test0.nvl");
    final File heapFile1 = mkHeapFile("test1.nvl");
    final File heapFile2 = mkHeapFile("test2.nvl");
    final File heapFile3 = mkHeapFile("test3.nvl");
    final String heapFileAbs0 = heapFile0.getAbsolutePath();
    final String heapFileAbs1 = heapFile1.getAbsolutePath();
    final String heapFileAbs2 = heapFile2.getAbsolutePath();
    final String heapFileAbs3 = heapFile3.getAbsolutePath();
    final File workDir = mkTmpDir("");

    // Shrink tx world size.
    nvlOpenarcCCAndRun(
      "create-shrink", workDir, "", true, 2,
      new String[]{
        "#include <mpi.h>",
        "#include <nvl.h>",
        "int main(int argc, char *argv[]) {",
        "  MPI_Init(&argc, &argv);",
        "  MPI_Group group;",
        "  MPI_Comm_group(MPI_COMM_WORLD, &group);",
        "  int rank;",
        "  MPI_Comm_rank(MPI_COMM_WORLD, &rank);",
        "  const char *file;",
        "  switch (rank) {",
        "  case 0: file = \""+heapFileAbs0+"\"; break;",
        "  case 1: file = \""+heapFileAbs1+"\"; break;",
        "  default: return 2;",
        "  }",
        "  nvl_heap_t *heap = nvl_create_mpi(file, 0, 0600, group);",
        "  if (!heap)",
        "    return 3;",
        "  return 4;",
        "}"},
      new String[0], false, 4);
    nvlOpenarcCCAndRun(
      "open-shrink", workDir, "", true, 2,
      new String[]{
        "#include <mpi.h>",
        "#include <nvl.h>",
        "int main(int argc, char *argv[]) {",
        "  MPI_Init(&argc, &argv);",
        "  MPI_Group group1, group2;",
        "  MPI_Comm_group(MPI_COMM_WORLD, &group1);",
        "  int rank;",
        "  MPI_Comm_rank(MPI_COMM_WORLD, &rank);",
        "  if (rank == 0) {",
        "    int otherRank = 1;",
        "    MPI_Group_excl(group1, 1, &otherRank, &group2);",
        "  }",
        "  else",
        "    group2 = group1;",
        "  const char *file;",
        "  switch (rank) {",
        "  case 0: file = \""+heapFileAbs0+"\"; break;",
        "  case 1: file = \""+heapFileAbs1+"\"; break;",
        "  default: return 2;",
        "  }",
        "  nvl_heap_t *heap = nvl_open_mpi(file, group2);",
        "  if (!heap)",
        "    return 3;",
        "  nvl_close(heap);",
        "  MPI_Finalize();",
        "  return 0;",
        "}"},
      new String[]{
        "nvlrt-pmemobj: error: opening heap with a different MPI"
         +" transaction world"},
      false, 1);
    heapFile0.delete();
    heapFile1.delete();

    // The only way for a process to add new processes to its tx world
    // (which happens when enlarging or when changing membership without
    // changing size) without hanging is for the new processes to add that
    // process to their tx worlds as well. In both tests below then,
    // multiple processes report a tx world change, but which processes
    // manage to report before MPI kills them is non-deterministic. Thus,
    // these tests make sure that all processes make the same kind of change
    // (either (1) enlarge or (2) change membership without changing size)
    // in order to be sure that any report means that the NVL runtime is
    // able to detect that specific kind of change.

    // Enlarge tx world size.
    nvlOpenarcCCAndRun(
      "create-enlarge", workDir, "", true, 3,
      new String[]{
        "#include <mpi.h>",
        "#include <nvl.h>",
        "int main(int argc, char *argv[]) {",
        "  MPI_Init(&argc, &argv);",
        "  MPI_Group group1, group2;",
        "  MPI_Comm_group(MPI_COMM_WORLD, &group1);",
        "  int rank, rankZero = 0;",
        "  MPI_Comm_rank(MPI_COMM_WORLD, &rank);",
        "  const char *file;",
        "  switch (rank) {",
        "  case 0:",
        "    file = \""+heapFileAbs0+"\";",
        "    MPI_Group_incl(group1, 1, &rankZero, &group2);",
        "    break;",
        "  case 1:",
        "    file = \""+heapFileAbs1+"\";",
        "    MPI_Group_excl(group1, 1, &rankZero, &group2);",
        "    break;",
        "  case 2:",
        "    file = \""+heapFileAbs2+"\";",
        "    MPI_Group_excl(group1, 1, &rankZero, &group2);",
        "    break;",
        "  default: return 2;",
        "  }",
        "  nvl_heap_t *heap = nvl_create_mpi(file, 0, 0600, group2);",
        "  if (!heap)",
        "    return 3;",
        "  return 4;",
        "}"},
      new String[0], false, 4);
    nvlOpenarcCCAndRun(
      "open-enlarge", workDir, "", true, 3,
      new String[]{
        "#include <mpi.h>",
        "#include <nvl.h>",
        "int main(int argc, char *argv[]) {",
        "  MPI_Init(&argc, &argv);",
        "  MPI_Group group;",
        "  MPI_Comm_group(MPI_COMM_WORLD, &group);",
        "  int rank;",
        "  MPI_Comm_rank(MPI_COMM_WORLD, &rank);",
        "  const char *file;",
        "  switch (rank) {",
        "  case 0: file = \""+heapFileAbs0+"\"; break;",
        "  case 1: file = \""+heapFileAbs1+"\"; break;",
        "  case 2: file = \""+heapFileAbs2+"\"; break;",
        "  default: return 2;",
        "  }",
        "  nvl_heap_t *heap = nvl_open_mpi(file, group);",
        "  if (!heap)",
        "    return 3;",
        "  nvl_close(heap);",
        "  MPI_Finalize();",
        "  return 0;",
        "}"},
      new String[]{
        "nvlrt-pmemobj: error: opening heap with a different MPI"
         +" transaction world",
        "nvlrt-pmemobj: error: opening heap with a different MPI"
         +" transaction world",
        "nvlrt-pmemobj: error: opening heap with a different MPI"
         +" transaction world"},
      true, 1);
    heapFile0.delete();
    heapFile1.delete();
    heapFile2.delete();

    // Change tx world membership but not size.
    nvlOpenarcCCAndRun(
      "create-membership", workDir, "", true, 4,
      new String[]{
        "#include <mpi.h>",
        "#include <nvl.h>",
        "int main(int argc, char *argv[]) {",
        "  MPI_Init(&argc, &argv);",
        "  MPI_Group group1, group2;",
        "  MPI_Comm_group(MPI_COMM_WORLD, &group1);",
        "  int rank, exclRanks[2];",
        "  MPI_Comm_rank(MPI_COMM_WORLD, &rank);",
        "  const char *file;",
        "  switch (rank) {",
        "  case 0:",
        "    file = \""+heapFileAbs0+"\";",
        "    exclRanks[0] = 2; exclRanks[1] = 3;",
        "    break;",
        "  case 1:",
        "    file = \""+heapFileAbs1+"\";",
        "    exclRanks[0] = 2; exclRanks[1] = 3;",
        "    break;",
        "  case 2:",
        "    file = \""+heapFileAbs2+"\";",
        "    exclRanks[0] = 0; exclRanks[1] = 1;",
        "    break;",
        "  case 3:",
        "    file = \""+heapFileAbs3+"\";",
        "    exclRanks[0] = 0; exclRanks[1] = 1;",
        "    break;",
        "  default: return 2;",
        "  }",
        "  MPI_Group_excl(group1, 2, exclRanks, &group2);",
        "  nvl_heap_t *heap = nvl_create_mpi(file, 0, 0600, group2);",
        "  if (!heap)",
        "    return 3;",
        "  return 4;",
        "}"},
      new String[0], false, 4);
    nvlOpenarcCCAndRun(
      "open-membership", workDir, "", true, 4,
      new String[]{
        "#include <mpi.h>",
        "#include <nvl.h>",
        "int main(int argc, char *argv[]) {",
        "  MPI_Init(&argc, &argv);",
        "  MPI_Group group1, group2;",
        "  MPI_Comm_group(MPI_COMM_WORLD, &group1);",
        "  int rank, exclRanks[2];",
        "  MPI_Comm_rank(MPI_COMM_WORLD, &rank);",
        "  const char *file;",
        "  switch (rank) {",
        "  case 0:",
        "    file = \""+heapFileAbs0+"\";",
        "    exclRanks[0] = 1; exclRanks[1] = 3;",
        "    break;",
        "  case 1:",
        "    file = \""+heapFileAbs1+"\";",
        "    exclRanks[0] = 0; exclRanks[1] = 2;",
        "    break;",
        "  case 2:",
        "    file = \""+heapFileAbs2+"\";",
        "    exclRanks[0] = 1; exclRanks[1] = 3;",
        "    break;",
        "  case 3:",
        "    file = \""+heapFileAbs3+"\";",
        "    exclRanks[0] = 0; exclRanks[1] = 2;",
        "    break;",
        "  default: return 2;",
        "  }",
        "  MPI_Group_excl(group1, 2, exclRanks, &group2);",
        "  nvl_heap_t *heap = nvl_open_mpi(file, group2);",
        "  if (!heap)",
        "    return 3;",
        "  nvl_close(heap);",
        "  MPI_Finalize();",
        "  return 0;",
        "}"},
      new String[]{
        "nvlrt-pmemobj: error: opening heap with a different MPI"
         +" transaction world",
        "nvlrt-pmemobj: error: opening heap with a different MPI"
         +" transaction world",
        "nvlrt-pmemobj: error: opening heap with a different MPI"
         +" transaction world",
        "nvlrt-pmemobj: error: opening heap with a different MPI"
         +" transaction world"},
      true, 1);
  }

  // ---------------------------------------
  // MPI group transactions: tx group errors

  @Test public void nestedMpiTxGroup() throws Exception {
    final File heapFile0 = mkHeapFile("test0.nvl");
    final File heapFile1 = mkHeapFile("test1.nvl");
    final String heapFileAbs0 = heapFile0.getAbsolutePath();
    final String heapFileAbs1 = heapFile1.getAbsolutePath();
    final File workDir = mkTmpDir("");

    nvlOpenarcCCAndRun(
      "static-inner", workDir, "", true, 1,
      new String[]{
        "#include <mpi.h>",
        "#include <nvl.h>",
        "int main(int argc, char *argv[]) {",
        "  MPI_Init(&argc, &argv);",
        "  MPI_Group group;",
        "  MPI_Comm_group(MPI_COMM_WORLD, &group);",
        "  int rank;",
        "  MPI_Comm_rank(MPI_COMM_WORLD, &rank);",
        "  nvl_heap_t *heap = nvl_create_mpi(\""+heapFileAbs0+"\", 0,",
        "                                    0600, group);",
        "  if (!heap)",
        "    return 2;",
        "  #pragma nvl atomic heap(heap)",
        "  {",
        "    #pragma nvl atomic heap(heap) mpiGroup(group)",
        "      1;",
        "  }",
        "  return 3;",
        "}"},
      new String[]{
        "nvlrt-pmemobj: error: nested MPI group transaction"},
      false, 1);
    heapFile0.delete();

    nvlOpenarcCCAndRun(
      "static-both", workDir, "", true, 1,
      new String[]{
        "#include <mpi.h>",
        "#include <nvl.h>",
        "int main(int argc, char *argv[]) {",
        "  MPI_Init(&argc, &argv);",
        "  MPI_Group group;",
        "  MPI_Comm_group(MPI_COMM_WORLD, &group);",
        "  int rank;",
        "  MPI_Comm_rank(MPI_COMM_WORLD, &rank);",
        "  nvl_heap_t *heap = nvl_create_mpi(\""+heapFileAbs0+"\", 0,",
        "                                    0600, group);",
        "  if (!heap)",
        "    return 2;",
        "  #pragma nvl atomic heap(heap) mpiGroup(group)",
        "  {",
        "    #pragma nvl atomic heap(heap) mpiGroup(group)",
        "      1;",
        "  }",
        "  return 3;",
        "}"},
      new String[]{
        "nvlrt-pmemobj: error: nested MPI group transaction"},
      false, 1);
    heapFile0.delete();

    nvlOpenarcCC(
      "fn.o", workDir, "-c", true, true,
      new String[]{
        "#include <mpi.h>",
        "#include <nvl.h>",
        "void fn(nvl_heap_t *heap, MPI_Group group) {",
        "  #pragma nvl atomic heap(heap) mpiGroup(group)",
        "    1;",
        "}",
      });
    nvlOpenarcCCAndRun(
      "dynamic-inner", workDir, "fn.o", true, 1,
      new String[]{
        "#include <mpi.h>",
        "#include <nvl.h>",
        "void fn(nvl_heap_t *heap, MPI_Group group);",
        "int main(int argc, char *argv[]) {",
        "  MPI_Init(&argc, &argv);",
        "  MPI_Group group;",
        "  MPI_Comm_group(MPI_COMM_WORLD, &group);",
        "  int rank;",
        "  MPI_Comm_rank(MPI_COMM_WORLD, &rank);",
        "  nvl_heap_t *heap = nvl_create_mpi(\""+heapFileAbs0+"\", 0,",
        "                                    0600, group);",
        "  if (!heap)",
        "    return 2;",
        "  #pragma nvl atomic heap(heap)",
        "    fn(heap, group);",
        "  return 3;",
        "}"},
      new String[]{
        "nvlrt-pmemobj: error: nested MPI group transaction"},
      false, 1);
    heapFile0.delete();

    nvlOpenarcCCAndRun(
      "dynamic-both", workDir, "fn.o", true, 1,
      new String[]{
        "#include <mpi.h>",
        "#include <nvl.h>",
        "void fn(nvl_heap_t *heap, MPI_Group group);",
        "int main(int argc, char *argv[]) {",
        "  MPI_Init(&argc, &argv);",
        "  MPI_Group group;",
        "  MPI_Comm_group(MPI_COMM_WORLD, &group);",
        "  int rank;",
        "  MPI_Comm_rank(MPI_COMM_WORLD, &rank);",
        "  nvl_heap_t *heap = nvl_create_mpi(\""+heapFileAbs0+"\", 0,",
        "                                    0600, group);",
        "  if (!heap)",
        "    return 2;",
        "  #pragma nvl atomic heap(heap) mpiGroup(group)",
        "    fn(heap, group);",
        "  return 3;",
        "}"},
      new String[]{
        "nvlrt-pmemobj: error: nested MPI group transaction"},
      false, 1);
    heapFile0.delete();

    // mpiGroup(MPI_GROUP_NULL) must be semantically equivalent to no
    // mpiGroup clause, but it exercises slightly more code to check the
    // mpiGroup clause, and that code used to fail an assertion here.
    nvlOpenarcCCAndRun(
      "inner-group-null", workDir, "", true, 2,
      new String[]{
        "#include <mpi.h>",
        "#include <nvl.h>",
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <unistd.h>",
        "int main(int argc, char *argv[]) {",
        "  MPI_Init(&argc, &argv);",
        "  MPI_Group group1, group2=MPI_GROUP_NULL;",
        "  MPI_Comm_group(MPI_COMM_WORLD, &group1);",
        "  int rank;",
        "  MPI_Comm_rank(MPI_COMM_WORLD, &rank);",
        "  const char *file = !rank ? \""+heapFileAbs0+"\"",
        "                           : \""+heapFileAbs1+"\";",
        "  nvl_heap_t *heap = nvl_create_mpi(file, 0, 0600, group1);",
        "  if (!heap)",
        "    return 2;",
        "  #pragma nvl atomic heap(heap) mpiGroup(group1)",
        "  {",
        "    #pragma nvl atomic heap(heap) mpiGroup(group2)",
        "      1;",
        "    if (rank == 0) {",
        "      sleep(1);", // try to let rank 1 get as far as possible
        "      exit(3);",
        "    }",
        "  }",
           // If outer tx barrier works, this should be unreachable by both
           // processes because one of them fails before the commit.
        "  fprintf(stderr, \"UNREACHABLE\\n\");",
        "  fflush(stderr);",
        "  return 4;",
        "}"},
      new String[0], false, 3);
  }

  @Test public void mpiTxGroupInLocalTxMode() throws Exception {
    final File heapFile = mkHeapFile("test.nvl");
    final String heapFileAbs = heapFile.getAbsolutePath();
    final File workDir = mkTmpDir("");
    nvlOpenarcCCAndRun(
      "test", workDir, "", true, 1,
      new String[]{
        "#include <mpi.h>",
        "#include <nvl.h>",
        "#include <stdio.h>",
        "int main(int argc, char *argv[]) {",
        "  MPI_Init(&argc, &argv);",
        "  nvl_heap_t *heap = nvl_create(\""+heapFileAbs+"\", 0, 0600);",
        "  if (!heap)",
        "    return 2;",
        "  MPI_Group group = MPI_GROUP_NULL;",
        "  #pragma nvl atomic heap(heap) mpiGroup(group)",
        "    1;",
        "  fprintf(stderr, \"OK\\n\");",
        "  MPI_Comm_group(MPI_COMM_WORLD, &group);",
        "  #pragma nvl atomic heap(heap) mpiGroup(group)",
        "    1;",
        "  nvl_close(heap);",
        "  MPI_Finalize();",
        "  return 0;",
        "}"},
      new String[]{
        "OK",
        "nvlrt-pmemobj: error: MPI group transaction on heap that is in"
        +" local transaction mode"},
      false, 1);
  }

  @Test public void mpiTxGroupNotInTxWorld() throws Exception {
    final File heapFile0 = mkHeapFile("test0.nvl");
    final File heapFile1 = mkHeapFile("test1.nvl");
    final File heapFile2 = mkHeapFile("test2.nvl");
    final File heapFile3 = mkHeapFile("test3.nvl");
    final String heapFileAbs0 = heapFile0.getAbsolutePath();
    final String heapFileAbs1 = heapFile1.getAbsolutePath();
    final String heapFileAbs2 = heapFile2.getAbsolutePath();
    final String heapFileAbs3 = heapFile3.getAbsolutePath();
    final File workDir = mkTmpDir("");
    nvlOpenarcCCAndRun(
      "test", workDir, "", true, 4,
      new String[]{
        "#include <mpi.h>",
        "#include <nvl.h>",
        "#include <stdio.h>",
        "int main(int argc, char *argv[]) {",
        "  MPI_Init(&argc, &argv);",
        "  MPI_Group group1, group2;",
        "  MPI_Comm_group(MPI_COMM_WORLD, &group1);",
        "  int rank, exclRanks[2];",
        "  MPI_Comm_rank(MPI_COMM_WORLD, &rank);",
        "  const char *file;",
        "  switch (rank) {",
        "  case 0:",
        "    file = \""+heapFileAbs0+"\";",
        "    exclRanks[0] = 2; exclRanks[1] = 3;",
        "    break;",
        "  case 1:",
        "    file = \""+heapFileAbs1+"\";",
        "    exclRanks[0] = 2; exclRanks[1] = 3;",
        "    break;",
        "  case 2:",
        "    file = \""+heapFileAbs2+"\";",
        "    exclRanks[0] = 0; exclRanks[1] = 1;",
        "    break;",
        "  case 3:",
        "    file = \""+heapFileAbs3+"\";",
        "    exclRanks[0] = 0; exclRanks[1] = 1;",
        "    break;",
        "  default: return 2;",
        "  }",
        "  MPI_Group_excl(group1, 2, exclRanks, &group2);",
        "  nvl_heap_t *heap = nvl_create_mpi(file, 0, 0600, group2);",
        "  if (!heap)",
        "    return 2;",
        "  #pragma nvl atomic heap(heap) mpiGroup(group1)",
        "    1;",
        "  nvl_close(heap);",
        "  MPI_Finalize();",
        "  return 0;",
        "}"},
      new String[]{
        "nvlrt-pmemobj: error: transaction's MPI group is not a subset of"
        +" the heap's MPI transaction world",
        "nvlrt-pmemobj: error: transaction's MPI group is not a subset of"
        +" the heap's MPI transaction world",
        "nvlrt-pmemobj: error: transaction's MPI group is not a subset of"
        +" the heap's MPI transaction world",
        "nvlrt-pmemobj: error: transaction's MPI group is not a subset of"
        +" the heap's MPI transaction world"},
      true, 1);
  }

  @Test public void mpiTxGroupNotReflexive() throws Exception {
    final File heapFile0 = mkHeapFile("test0.nvl");
    final File heapFile1 = mkHeapFile("test1.nvl");
    final String heapFileAbs0 = heapFile0.getAbsolutePath();
    final String heapFileAbs1 = heapFile1.getAbsolutePath();
    final File workDir = mkTmpDir("");
    final String[] src = new String[]{
      "#include <mpi.h>",
      "#include <nvl.h>",
      "int main(int argc, char *argv[]) {",
      "  MPI_Init(&argc, &argv);",
      "  int rank;",
      "  MPI_Comm_rank(MPI_COMM_WORLD, &rank);",
      "  MPI_Group group1, group2;",
      "  MPI_Comm_group(MPI_COMM_WORLD, &group1);",
      "  MPI_Group_excl(group1, 1, &rank, &group2);",
      "  const char *file = !rank ? \""+heapFileAbs0+"\"",
      "                           : \""+heapFileAbs1+"\";",
      "  nvl_heap_t *heap = nvl_create_mpi(file, 0, 0600, group1);",
      "  if (!heap)",
      "    return 1;",
      "  if (rank == 0) {",
      "    #pragma nvl atomic heap(heap) mpiGroup(group2)",
      "      1;",
      "  }",
      "  nvl_close(heap);",
      "  MPI_Finalize();",
      "  return 0;",
      "}"};
    final File exe = nvlOpenarcCC("create", workDir, "", true, true, src);
    // empty
    nvlRunExe(exe, workDir, "", -1, 1, new String[]{
      "nvlrt-pmemobj: error: transaction's MPI group does not include the"
      +" process's own MPI rank"},
      false, 1);
    heapFile0.delete();
    heapFile1.delete();
    // non-empty
    nvlRunExe(exe, workDir, "", -1, 2, new String[]{
      "nvlrt-pmemobj: error: transaction's MPI group does not include the"
      +" process's own MPI rank"},
      false, 1);
  }

  // ------------------------------
  // MPI group transactions: jacobi

  @Test public void jacobiMPI() throws Exception {
    final int n = 1024;
    final int iters = 10;
    final int itersPerTx = 1;
    final int mpiRanks = 5;
    final String sumRegex = "5512.098305";
    final File heapDir = mkHeapDir("jacobiMpi");
    final String heapDirAbs = heapDir.getAbsolutePath();
    final File workDir = mkTmpDir("");
    final String workDirAbs = workDir.getAbsolutePath();
    final File srcFile = new File(getTopDir(),
                                  "test/benchmarks/nvl-c/kernels/"
                                  +"jacobi-mpi-checkpoints/jacobi.c");
    final File exe = nvlOpenarcCC(
      "jacobiMPI", workDir,
      "-DN="+n+" -Warc,-macro=N="+n
      +" -DITERS="+iters+" -Warc,-macro=ITERS="+iters
      +" -DITERS_PER_TX="+itersPerTx+" -Warc,-macro=ITERS_PER_TX="+itersPerTx
      +" -DTXS=4",
      true, true, srcFile);

    LinkedList<String> expout = new LinkedList<>();
    for (int i = 0; i < mpiRanks; ++i)
      expout.add("rank = *"+i+"/"+mpiRanks
                 +", rowsLocal = *"+(n/mpiRanks+(i<(n%mpiRanks)?1:0))+"/"+n
                 +", procName = .*");
    for (int i = 0; i < mpiRanks; ++i)
      expout.add("rank "+i+": heapReady");
    for (int i = 0; i < mpiRanks; ++i)
      expout.add("rank "+i+": rootReady");
    expout.add("NVMDIR = "+heapDirAbs+", RESDIR = "+workDirAbs);
    expout.add("ITERS = "+iters+", N = "+n+", usesMsync = [01],"
               +" tx mode = 4, ITERS_PER_TX = "+itersPerTx);
    expout.add("sum = "+sumRegex);
    txsCheck("jacobiMPI", iters/itersPerTx,
             shesc(heapDirAbs)+" "+shesc(workDirAbs),
             expout, false, heapDir, workDir, exe, mpiRanks);
  }
}
