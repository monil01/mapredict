package openacc.codegen.llvmBackend;

import static org.junit.Assert.assertEquals;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import openacc.exec.BuildConfig;

import org.junit.FixMethodOrder;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runners.MethodSorters;

/**
 * Runs SPEC CPU 2006 (tested with v1.0.1) benchmarks that contain C source
 * code.
 * 
 * <p>
 * {@code make.header} must be configured with the location of your SPEC CPU
 * 2006 installation, or these tests will fail. Changes to {@code make.header}
 * must be applied by rebuilding OpenARC using the {@code build.sh} script.
 * </p>
 * 
 * <p>
 * Some of these benchmarks fail to compile due to C constructs that
 * OpenARC+LLVM does not currently support. To eliminate some of those
 * constructs, we have provided a patch file, which can be applied to a SPEC
 * installation as follows:
 * </p>
 * 
 * <pre>
 * $ cd $SPEC_INSTALL_DIR
 * $ patch -p0 < $OPENARC_DIR/test/junit/openacc/codegen/llvmBackend/spec-cpu-2006-v1.0.1.patch
 * </pre>
 * 
 * <p>
 * WARNING: After applying this patch, your SPEC installation will no longer
 * obey the SPEC rules for reporting performance measurements. However, the
 * only source code modified is covered under an open license, so these
 * modifications do not violate the SPEC license.
 * </p>
 * 
 * <p>
 * Because of the above dependencies and because this test class takes a long
 * time to complete, it is disabled by default. To enable it, remove the test
 * class's Ignore attribute.
 * </p>
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
@Ignore
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class BuildLLVMTest_SpecCPU2006 extends BuildLLVMTest {
  private File makeSpecCfg(BuildConfig config) throws IOException {
    final String specCfgInStr = config.getProperty("spec_cfg");
    final File specCfgIn;
    if (specCfgInStr.contains("/"))
      specCfgIn = new File(specCfgInStr);
    else
      specCfgIn = new File(getSrcDir(), specCfgInStr);
    final File specCfg = tmpFolder.newFile("spec.cfg");
    final Map<String, String> substs = new HashMap<>();
    substs.put("CC",  new File(getTopDir(), "bin/openarc-cc")
                      .getAbsolutePath());
    substs.put("CXX", config.getProperty("cxx"));
    substs.put("FC",  config.getProperty("fc"));
    configureFile(specCfgIn, specCfg, substs);
    return specCfg;
  }

  private void configureFile(File inFile, File outFile,
                             Map<String,String> substs) throws IOException
  {
    final FileReader fr = new FileReader(inFile);
    final BufferedReader bfr = new BufferedReader(fr);
    final FileWriter fw = new FileWriter(outFile);
    final String lineSep = System.getProperty("line.separator");
    System.err.println("--------------------" + inFile.getAbsolutePath()
                     + "--------------------");
    String line;
    while (null != (line = bfr.readLine())) {
      for (Map.Entry<String,String> entry : substs.entrySet())
        line = line.replace("@"+entry.getKey()+"@", entry.getValue());
      fw.write(line);
      fw.write(lineSep);
      System.err.println(line);
    }
    fw.close();
    bfr.close();
  }

  private void runTest(String testName)
    throws IOException, InterruptedException
  {
    final BuildConfig config = BuildConfig.getBuildConfig();
    final File specDir = new File(config.getProperty("spec_cpu2006"));
    final File specCfg = makeSpecCfg(config);
    runspec(specDir, specCfg, testName, "clobber", "");
    runspec(specDir, specCfg, testName, "build", "");
    runspec(specDir, specCfg, testName, "run",
            "--size=test --tune=base --noreportable --iterations=1");
  }

  private void runspec(File specDir, File specCfg, String testName,
                       String action, String opts)
    throws IOException, InterruptedException
  {
    ProcessBuilder builder = new ProcessBuilder(
      "sh", "-c",
      ". ./shrc && runspec"
      + " --config "+shesc(specCfg.getAbsolutePath())
      + " --action="+action
      + (opts.isEmpty() ? "" : " "+opts)
      + " " + shesc(testName));
    builder.directory(specDir);
    builder.inheritIO();
    Process process = builder.start();
    assertEquals(testName+" "+action+" exit value", 0, process.waitFor());
  }

  @XFail(exception=AssertionError.class,
         message="400.perlbench build exit value expected:<0> but was:<1>")
  @Test public void t400_perlbench()  throws Exception { runTest("400.perlbench"); }

  @XFail(exception=AssertionError.class,
         message="401.bzip2 run exit value expected:<0> but was:<1>")
  @Test public void t401_bzip2()      throws Exception { runTest("401.bzip2"); }

  @Ignore // run does not terminate
  @Test public void t403_gcc()        throws Exception { runTest("403.gcc"); }

  @Test public void t429_mcf()        throws Exception { runTest("429.mcf"); }
  @Test public void t433_milc()       throws Exception { runTest("433.milc"); }
  @Test public void t435_gromacs()    throws Exception { runTest("435.gromacs"); }

  @XFail(exception=AssertionError.class,
         message="436.cactusADM run exit value expected:<0> but was:<1>")
  @Test public void t436_cactusADM()  throws Exception { runTest("436.cactusADM"); }

  @XFail(exception=AssertionError.class,
         message="445.gobmk run exit value expected:<0> but was:<1>")
  @Test public void t445_gobmk()      throws Exception { runTest("445.gobmk"); }

  @Test public void t454_calculix()   throws Exception { runTest("454.calculix"); }
  @Test public void t456_hmmer()      throws Exception { runTest("456.hmmer"); }
  @Test public void t458_sjeng()      throws Exception { runTest("458.sjeng"); }

  @XFail(exception=AssertionError.class,
         message="462.libquantum build exit value expected:<0> but was:<1>")
  @Test public void t462_libquantum() throws Exception { runTest("462.libquantum"); }

  @Test public void t464_h264ref()    throws Exception { runTest("464.h264ref"); }

  // Fails on OS X but passses on newark.
  @Test public void t470_lbm()        throws Exception { runTest("470.lbm"); }

  @Test public void t481_wrf()        throws Exception { runTest("481.wrf"); }

  @XFail(exception=AssertionError.class,
         message="482.sphinx3 run exit value expected:<0> but was:<1>")
  @Test public void t482_sphinx3()    throws Exception { runTest("482.sphinx3"); }

  @Test public void t998_specrand()   throws Exception { runTest("998.specrand"); }
  @Test public void t999_specrand()   throws Exception { runTest("999.specrand"); }
}
