package openacc.codegen.llvmBackend;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.junit.Test;

/**
 * Runs LULESH.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class BuildLLVMTest_LULESH extends BuildLLVMTest {
  private void runTest(String ccOpts, String exePre, String mpiTasks,
                       String niters, double energy)
    throws IOException, InterruptedException
  {
    final File luleshSrcDir = new File(getTopDir(),
                                       "test/benchmarks/impacc/lulesh");
    String ccArgs = shesc(luleshSrcDir.getAbsolutePath()) + "/*.c"
                    +" -lm "+ccOpts;
    final HashMap<String, String> vals = new HashMap<>();
    openarcCCAndRun(
      "lulesh", mkTmpDir(""), ccArgs, exePre, "",
      new RunChecker(){
        @Override
        public void checkDuring(String exeName, InputStream outStream,
                                InputStream errStream)
          throws IOException
        {
          final BufferedReader out
            = new BufferedReader(new InputStreamReader(outStream));
          final Pattern pat = Pattern.compile(
            "^\\s*([^\\s=][^=]*[^\\s=])\\s*=\\s*([^\\s]*)\\s*$");
          String line;
          while (null != (line = out.readLine())) {
            Matcher matcher = pat.matcher(line);
            if (matcher.matches())
              vals.put(matcher.group(1), matcher.group(2));
          }
        }
      });

    // verify based on RefResults.txt from LULESH source code
    String k, v;
    k = "Problem size";    assertEquals(k, "30",     vals.get(k));
    k = "MPI tasks";       assertEquals(k, mpiTasks, vals.get(k));
    k = "Iteration count"; assertEquals(k, niters,    vals.get(k));

    k = "Final Origin Energy"; v = vals.get(k);
    assertNotNull(k + " must be defined", v);
    assertEquals(k, energy, Float.parseFloat(vals.get(k)), 0.000005e+05);

    final String diffs[] = {"MaxAbsDiff", "TotalAbsDiff", "MaxRelDiff"};
    for (int i = 0; i < diffs.length; ++i) {
      k = diffs[i]; v = vals.get(k);
      assertNotNull(k + " must be defined", v);
      assertTrue(k + " must be small", Float.parseFloat(v) <= 1e-8);
    }
  }

  @Test public void serial()
    throws IOException, InterruptedException
  {
    // Expected results from RefResults.txt from LULESH source code.
    runTest("-DUSE_MPI=0", "", "1", "932", 2.025075e+05);
  }

  @Test public void mpi()
    throws IOException, InterruptedException
  {
    final String includes = assumeConfig("mpi_includes");
    final String libdir = assumeConfig("mpi_libdir");
    final String mpiexec = assumeConfig("mpi_exec");
    // Expected results from running with 1 rank (mpiexec -n1) and problem
    // size of 60, which gives the same number of elements (1*60*60*60 =
    // 8*30*30*30) and thus the same results.
    runTest("-DUSE_MPI=1 -lmpi"+" -I"+shesc(includes)+" -L"+shesc(libdir),
            "LD_LIBRARY_PATH="+shesc(libdir)+":$LD_LIBRARY_PATH"
            +" "+mpiexec+" -n 8", "8", "2031", 7.130703e+05);
  }
}
