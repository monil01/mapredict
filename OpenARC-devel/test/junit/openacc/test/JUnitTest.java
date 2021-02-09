package openacc.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import static org.junit.Assume.assumeTrue;

import java.io.Closeable;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PipedInputStream;
import java.io.PipedOutputStream;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.lang.reflect.Method;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

import openacc.exec.BuildConfig;

import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.contrib.java.lang.system.ExpectedSystemExit;
import org.junit.rules.ExpectedException;
import org.junit.rules.TemporaryFolder;
import org.junit.rules.TestName;
import org.junit.rules.TestRule;
import org.junit.runner.Description;
import org.junit.runners.model.Statement;

/**
 * Super class for all JUnit test cases for OpenARC.  It declares rules that
 * should be common to all tests, and it encapsulates functionality that is
 * potentially useful throughout the test suite.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public abstract class JUnitTest {
  /**
   * Directory to which tests can write temporary files or subdirectories.
   */
  @Rule public final TemporaryFolder tmpFolder = new TemporaryFolder();
  /**
   * A way for methods to access the current test name.
   */
  @Rule public final TestName testName = new TestName();
  /**
   * Without this, when OpenARC calls {@link System#exit} to report errors, the
   * entire test suite terminates.
   */
  @Rule public final ExpectedSystemExit exit = ExpectedSystemExit.none();
  /**
   * A way for methods to indicate exceptions that are expected as part of
   * normal behavior. For failures that are expected due to some bug that
   * should eventually be fixed, see {@link XFail} instead.
   */
  @Rule public final ExpectedException exception = ExpectedException.none();

  /**
   * Annotation for test methods that are expected to fail due to some bug
   * that should eventually be fixed.
   * 
   * <p>
   * The benefits of using this annotation are similar to the benefits of GNU
   * Autotest's AT_XFAIL_IF macro:
   * </p>
   * 
   * <ol>
   * <li>
   * Expected failure verification and suppression: if the annotated test
   * method fails in the expected way, it is reported as a pass. When all
   * expected failures are annotated in this way, it is much easier to notice
   * new failures.</li>
   * 
   * <li>
   * Unexpected pass notification: once an expected failure is fixed, it is
   * then reported as a failure with a message suggesting the annotation is no
   * longer appropriate.</li>
   * 
   * <li>
   * Expected failure summary: all expected failures are listed by the
   * {@link XFailSummary} class's test method, which fails if there are any
   * expected failures. In other words, all expected failures are reduced to
   * one failure so you can easily notice new failures but you don't forget
   * that there are still expected failures in your test suite. Specifically,
   * {@link XFailSummary} searches the OpenARC class directory for any test
   * method in any class file that is annotated with {@link XFail}.</li>
   * </ol>
   * 
   * <p>
   * In comparison, the {@code expected} property of the {@link Test}
   * annotation has benefits 1 and 2 but not 3. It is useful for cases where
   * an exception is actually <em>desired</em> behavior rather than a bug to
   * be fixed later. Likewise for {@link ExpectedException}, which can be
   * accessed via {@link #exception}.
   * </p>
   * 
   * <p>
   * In comparison, the {@link Ignore} annotation suppresses failures (partial
   * benefit 1), and JUnit test runners normally produce a summary of
   * {@link Ignore} annotations (benefit 3). The {@link Ignore} annotation
   * does not verify the kind of failure (partial benefit 1) or check for an
   * unexpected pass (benefit 2). It is useful for temporarily disabling a
   * test method if, for example, a test method behaves unpredictably or
   * crashes the test suite.
   * </p>
   */
  @Retention(RetentionPolicy.RUNTIME)
  @Target({ElementType.METHOD})
  public static @interface XFail {
    /** The type of exception that the test method should throw. */
    Class<? extends Throwable> exception();
    /**
     * What {@link Throwable#getMessage} should return for the thrown
     * exception.
     */
    String message();
  }
  /**
   * See {@link XFail}.
   */
  @Rule public final TestRule xFailRule = new TestRule() {
    @Override
    public Statement apply(final Statement base,
                           final Description description)
    {
      return new Statement() {
        @Override
        public void evaluate() throws Throwable {
          XFail xFail = description.getAnnotation(XFail.class);
          try {
            base.evaluate();
          } catch (Throwable e) {
            if (xFail == null)
              throw e;
            if (!(xFail.exception().isAssignableFrom(e.getClass()))
                || !xFail.message().equals(e.getMessage()))
              throw new AssertionError("test failure does not match "
                                       + XFail.class.getSimpleName()
                                       + " annotation",
                                       e);
            System.err.print("EXPECTED FAILURE: ");
            e.printStackTrace(System.err);
            return;
          }
          if (xFail != null)
            fail("test passes but has an " + XFail.class.getSimpleName()
                 + " annotation");
        }
      };
    }
  };
  /**
   * See {@link XFail}.
   */
  public static class XFailSummary {
    @Test public void listXFails() throws Exception {
      URL url = JUnitTest.class.getResource("");
      final File curDir = new File(url.toURI());
      assert(curDir.isDirectory());
      final File dir = curDir.getParentFile().getParentFile();
      List<String> xFails = new ArrayList<>();
      searchDir(dir.getAbsolutePath(), dir, xFails);
      if (xFails.isEmpty())
        return;
      StringBuilder str
        = new StringBuilder(xFails.size() + " test(s) annotated with "
                            + XFail.class.getSimpleName() + ":");
      for (String xFail : xFails)
        str.append(System.getProperty("line.separator") + xFail);
      fail(str.toString());
    }
    private void searchDir(String baseDir, File dir, List<String> xFails)
      throws ClassNotFoundException
    {
      for (File file : dir.listFiles()) {
        if (file.isDirectory()) {
          searchDir(baseDir, file, xFails);
          continue;
        }
        final String fileName = file.getAbsolutePath();
        if (!fileName.endsWith(".class"))
          continue;
        final String className
          = fileName.substring(baseDir.length()+1, fileName.length()-6)
            .replace("/",  ".");
        final Class<?> clazz
          = ClassLoader.getSystemClassLoader().loadClass(className);
        for (Method method : clazz.getMethods()) {
          if (method.isAnnotationPresent(XFail.class))
            xFails.add(clazz.getCanonicalName() + "#" + method.getName());
        }
      }
    }
  }

  /**
   * Create a temporary directory whose name is based on the test case/method.
   * 
   * @param dirNameSuffix
   *          the suffix to add to the directory name. Within a single test
   *          method, vary the suffix among {@link #writeTmpFile} and
   *          {@link #mkTmpDir} calls to avoid file name conflicts.
   * @param reuseExisting
   *           if false, fail if the directory already exists
   * @return the directory
   * @throws IOException
   *           if there's a problem creating the directory
   */
  public final File mkTmpDir(String dirNameSuffix, boolean reuseExisting)
    throws IOException
  {
    String dirName = testName.getMethodName() + dirNameSuffix;
    if (reuseExisting) {
      File dir = new File(tmpFolder.getRoot(), dirName);
      if (dir.exists())
        return dir;
    }
    return tmpFolder.newFolder(dirName);
  }

  /**
   * Same as {@link #mkTmpDir(String, boolean)} but with
   * {@code reuseExisting} set to false.
   */
  public final File mkTmpDir(String dirNameSuffix) throws IOException {
    return mkTmpDir(dirNameSuffix, false);
  }

  /**
   * Create and write a temporary file whose name is based on the test
   * case/method.
   * 
   * @param fileNameSuffix
   *          the suffix, such as {@code .c}, to add to the file name. Within
   *          a single test method, vary the suffix among
   *          {@link #writeTmpFile} and {@link #mkTmpDir} calls to avoid file
   *          name conflicts.
   * @param lines
   *          the content of the file, line by line
   * @return the file
   * @throws IOException
   *           if there's a problem writing to the file
   */
  public final File writeTmpFile(String fileNameSuffix, String... lines)
    throws IOException
  {
    final File file = tmpFolder.newFile(testName.getMethodName()
                                        + fileNameSuffix);
    writeFile(file, lines);
    return file;
  }

  /**
   * Create a subdirectory in the specified directory.
   * 
   * @param parentDir
   *          the parent directory. Should normally be the result of
   *          {@link #mkTmpDir} or {@link #mkDirInTmpDir}.
   * @param subdirName
   *          the name of the subdirectory
   * @throws IOException
   *           if there's a problem writing to the file
   */
  public final File mkDirInTmpDir(File parentDir, String subdirName)
    throws IOException
  {
    assert(parentDir.isDirectory());
    final File dir = new File(parentDir, subdirName);
    if (!dir.mkdirs())
      throw new IOException("failed to create directory: " + dir);
    return dir;
  }

  /**
   * Create and write a file in the specified directory.
   * 
   * @param dir
   *          the directory. Should normally be the result of
   *          {@link #mkTmpDir} or {@link #mkDirInTmpDir}.
   * @param fileName
   *          the name of the file
   * @param lines
   *          the content of the file, line by line
   * @throws IOException
   *           if there's a problem writing to the file
   */
  public final File writeFileInTmpDir(File dir, String fileName,
                                      String... lines)
    throws IOException
  {
    assert(dir.isDirectory());
    final File file = new File(dir, fileName);
    if (!file.createNewFile())
      throw new IOException("failed to create file: " + file);
    writeFile(file, lines);
    return file;
  }

  /**
   * Write a file.
   * 
   * @param file
   *          the file to write. Should normally be a file within a directory
   *          created by {@link #mkTmpDir} or {@link #mkDirInTmpDir}.
   * @param lines
   *          the content of the file, line by line
   * @throws IOException
   *           if there's a problem writing to the file
   */
  private final void writeFile(File file, String... lines) throws IOException {
    final FileWriter fw = new FileWriter(file);
    System.err.println( "---------------------" + file
                       +"---------------------");
    for (String line : lines) {
      System.err.println(line);
      fw.write(line);
      fw.write(System.getProperty("line.separator"));
    }
    fw.close();
  }

  /**
   * The OpenARC subdirectory where Java class files are generated. Specified
   * relatively. {@link #getTopClassDir} computes it absolutely.
   */
  private static final String[] CLASS_DIR_NAMES = {"class"};
  /**
   * The OpenARC subdirectory where JUnit test source code is stored.
   * Specified relatively. {@link #getTopSrcDir} computes it absolutely.
   */
  private static final String[] SRC_DIR_NAMES = {"test", "junit"};
  /**
   * The subdirectory within {@link #SRC_DIR_NAMES} where {@link JUnitTest}
   * source is stored.
   */
  private static final String[] THIS_DIR_NAMES = {"openacc", "test"};

  /**
   * Get the top OpenARC directory. Its subdirectories include {@code bin},
   * for example.
   */
  public static final File getTopDir() {
    File dir = getTopClassDir();
    for (int i = CLASS_DIR_NAMES.length-1; i >= 0; --i) {
      assert(dir.getName().equals(CLASS_DIR_NAMES[i]));
      dir = dir.getParentFile();
      assert(dir != null);
    }
    return dir;
  }

  /**
   * Get the top OpenARC subdirectory where Java class files are generated.
   * It is specified relatively by {@link #CLASS_DIR_NAMES}.
   */
  public static final File getTopClassDir() {
    final URL dirURL = JUnitTest.class.getResource("");
    File dir;
    try {
      dir = new File(dirURL.toURI());
    } catch (URISyntaxException e) {
      throw new IllegalStateException(
        "could not convert "+JUnitTest.class.getName()+"'s class directory to"
        +" a URI", e);
    }
    for (int i = THIS_DIR_NAMES.length-1; i >= 0; --i) {
      assert(dir.getName().equals(THIS_DIR_NAMES[i]));
      dir = dir.getParentFile();
      assert(dir != null);
    }
    return dir;
  }

  /**
   * Get the top OpenARC subdirectory where JUnit test source is stored. It is
   * specified relatively by {@link #SRC_DIR_NAMES}.
   */
  public static final File getTopSrcDir() {
    File dir = getTopDir();
    for (int i = 0; i < SRC_DIR_NAMES.length; ++i)
      dir = new File(dir, SRC_DIR_NAMES[i]);
    return dir;
  }

  /**
   * Get the OpenARC subdirectory where Java class files are generated for the
   * runtime class from which this method is called.
   */
  public final File getClassDir() {
    final URL dirURL = getClass().getResource("");
    final File dir;
    try {
      dir = new File(dirURL.toURI());
    } catch (URISyntaxException e) {
      throw new IllegalStateException(
        "could not convert "+getClass().getName()+"'s class directory to a"
        +" URI", e);
    }
    assert(dir.isDirectory());
    return dir;
  }

  private String[] fileToDirArray(File file) {
    final List<String> dirList = new ArrayList<>();
    while (file != null) {
      dirList.add(file.getName());
      file = file.getParentFile();
    }
    final String[] dirs = new String[dirList.size()];
    for (int i = 0; i < dirs.length; ++i)
      dirs[i] = dirList.get(dirs.length-1-i);
    return dirs;
  }

  /**
   * Get the OpenARC subdirectory where JUnit test source is stored for the
   * runtime class from which this method is called. In other words, a test
   * method can use this to look for other files in its own source directory.
   */
  public File getSrcDir() {
    final String[] topClassDir = fileToDirArray(getTopClassDir());
    final String[] classDir = fileToDirArray(getClassDir());
    assert(topClassDir.length <= classDir.length);
    int i;
    for (i = 0; i < topClassDir.length; ++i)
      assert(topClassDir[i].equals(classDir[i]));
    File dir = getTopSrcDir();
    for (; i < classDir.length; ++i)
      dir = new File(dir, classDir[i]);
    return dir;
  }

  /**
   * Quote a string for sh.
   * 
   * @param str the string to be quoted
   * @return the sh-quoted version of {@code str}
   */
  public String shesc(String str) {
    return "'" + str.replace("'", "'\\''") + "'";
  }

  private class Tee implements Closeable {
    private final InputStream in;
    private final OutputStream out;
    private final PipedInputStream pipedIn;
    private final PipedOutputStream pipedOut;
    private final IOException[] ex;
    private final Thread thread;
    private volatile boolean interrupted = false;
    public Tee(InputStream in_, OutputStream out_) throws IOException {
      in = in_;
      out = out_;
      // If there's a lot of output and the test suite deadlocks, you
      // probably need to increase the pipe size, or just read off the
      // output as it's generated.
      pipedIn = new PipedInputStream(100*1024);
      pipedOut= new PipedOutputStream(pipedIn);
      ex = new IOException[]{null};
      thread = new Thread() {
        public void run() {
          int i;
          try {
            while (-1 != (i = in.read())) {
              out.write(i);
              pipedOut.write(i);
            }
            pipedOut.close();
          }
          catch (IOException e) {
            try {
              pipedOut.close();
            }
            catch (IOException e2) {
            }
            if (!interrupted)
              ex[0] = e;
          }
        }
      };
      thread.start();
    }
    public InputStream getInputStream() {
      return pipedIn;
    }
    public void interrupt() {
      // This is helpful when we're done with the Tee (because we killed
      // the process whose stream it's reading), but pipedOut might be full
      // because nobody's reading it, and we don't want close to get stuck
      // while waiting for the thread to join.
      interrupted = true;
      thread.interrupt();
    }
    public void close() throws IOException {
      try {
        thread.join();
      } catch (InterruptedException e) {
        throw new IOException(e);
      }
      if (ex[0] != null)
        throw ex[0];
    }
  }

  /**
   * Used by {@link #openarcCCAndRun} to verify the run of an executable.
   * Instances of {@link RunChecker} check that the exit value is zero and
   * that stderr is empty.
   */
  public class RunChecker {
    /**
     * Called while executable is running. Default implementation does
     * nothing.
     * 
     * @param exeName
     *          the executable name, for the sake of diagnostics
     * @param outStream
     *          the executable's stdout, which can be examined as it's
     *          produced
     * @param errStream
     *          the executable's stderr, which can be examined as it's
     *          produced
     * @throws IOException
     */
    public void checkDuring(String exeName, InputStream outStream,
                            InputStream errStream)
      throws IOException
    {
    }
    /**
     * Called after executable has terminated.
     * 
     * @param exeName
     *          the executable name, for the sake of diagnostics
     * @param outStream
     *          the executable's stdout. Default implementation does not
     *          examine it.
     * @param errStream
     *          the executable's stderr. Default implementation asserts that
     *          it is empty.
     * @param exitValue
     *          the executable's exit value. Default implementation asserts
     *          that it is zero.
     * @throws IOException
     */
    public void checkAfter(String exeName, InputStream outStream,
                           InputStream errStream, int exitValue)
      throws IOException
    {
      assertTrue(exeName+": stderr must be empty",
                 errStream.available() == 0);
      assertEquals(exeName+": exit value", 0, exitValue);
    }
  }

  /**
   * Use the openarc-cc script to compile a test program, and then run it.
   * 
   * This method just calls {@link #openarcCC} followed by {@link #runExe}.
   */
  public void openarcCCAndRun(String exeName, File workDir, String ccArgs,
                              String exePre, String exeArgs,
                              RunChecker runChecker)
    throws IOException, InterruptedException
  {
    final File exe = openarcCC(exeName, workDir, ccArgs);
    runExe(exe, workDir, exePre, exeArgs, runChecker, -1);
  }

  /**
   * Use the openarc-cc script to compile a test program.
   * 
   * @param outFileName
   *          the desired name of the output file, relative to
   *          {@code workDir}
   * @param workDir
   *          the directory in which to compile
   * @param ccArgs
   *          the arguments to pass to openarc-cc except the {@code -o}
   *          option. Must be sh-quoted by caller.
   * @return the executable
   * @throws IOException
   * @throws InterruptedException
   */
  public File openarcCC(String outFileName, File workDir, String ccArgs)
    throws IOException, InterruptedException
  {
    final File openarcCC = new File(getTopDir(), "bin/openarc-cc");
    final File outFile = new File(workDir, outFileName);
    final String cmd = shesc(openarcCC.getAbsolutePath())
                 +" -Warc,-WerrorLLVM "+ccArgs
                 +" -o "+shesc(outFile.getAbsolutePath());
    System.err.println(outFileName+": compiling: "+cmd);
    final ProcessBuilder builder = new ProcessBuilder("sh", "-c", cmd);
    builder.directory(workDir);
    builder.inheritIO();
    final Process process = builder.start();
    assertEquals(outFileName+": compile exit value", 0, process.waitFor());
    return outFile;
  }

  /**
   * Run an executable.
   * 
   * @param exe
   *          the executable
   * @param workDir
   *          the directory in which to run it
   * @param exePre
   *          prefix for command-line call of executable. For example, it
   *          might set LD_LIBRARY_PATH. Must be sh-quoted by caller.
   * @param exeArgs
   *          the arguments to pass to executable. Must be sh-quoted by
   *          caller.
   * @param runChecker
   *          checker for stdout, stderr, and exit value of executable run
   * @param killTimeNano
   *          number of nanoseconds after which to kill the process. Ignored
   *          if negative.
   * @throws IOException
   * @throws InterruptedException
   */
  public void runExe(File exe, File workDir, String exePre,
                     String exeArgs, RunChecker runChecker,
                     final long killTimeNano)
    throws IOException, InterruptedException
  {
    final String exeName = exe.getName();

    // run
    String cmd = exePre;
    if (!cmd.isEmpty())
      cmd += " ";
    cmd += exe.getAbsolutePath();
    if (!exeArgs.isEmpty())
      cmd += " "+exeArgs;
    System.err.println(exeName+": running: "+cmd);
    final ProcessBuilder builder = new ProcessBuilder("sh", "-c", cmd);
    builder.directory(workDir);
    final Process process = builder.start();
    final Tee outTee = new Tee(process.getInputStream(), System.out);
    final Tee errTee = new Tee(process.getErrorStream(), System.err);
    if (killTimeNano >= 0) {
      new Thread() {
        @Override
        public void run() {
          try {
            Thread.sleep(killTimeNano/1000000);
          } catch (InterruptedException e) {
          }
          System.err.println(exeName+": killing...");
          System.err.flush();
          outTee.interrupt();
          errTee.interrupt();
          process.destroy();
        }
      }.start();
    }

    // check during
    runChecker.checkDuring(exeName, outTee.getInputStream(),
                           errTee.getInputStream());

    // terminate
    int exitValue = process.waitFor();
    outTee.close();
    errTee.close();

    // check after
    runChecker.checkAfter(exeName, outTee.getInputStream(),
                          errTee.getInputStream(), exitValue);
  }

  /**
   * Skip test if a configuration property is not defined, and print message
   * about skip on stderr.
   * 
   * @param prop
   *          the name of the property in {@link BuildConfig}
   * @return the value of the property
   * @throws AssumptionViolatedException
   *           if the property is not defined (causes test to be skipped)
   * @throws IllegalStateException
   *           if configure.mk has not been extended to copy the property to
   *           build.cfg (causes test to fail)
   */
  public String assumeConfig(String prop) {
    final String msg = prop.toUpperCase()+" is not configured in"
                       +" make.header";
    final String value = BuildConfig.getBuildConfig().getProperty(prop);
    final boolean hasProp = !value.isEmpty();
    // "./build.sh check" (that is, org.junit.runner.JUnitCore) doesn't
    // report failed assumptions, so report failure here.
    if (!hasProp)
      System.err.println("skipping remainder of test: " + msg);
    assumeTrue(msg, hasProp);
    return value;
  }
}
