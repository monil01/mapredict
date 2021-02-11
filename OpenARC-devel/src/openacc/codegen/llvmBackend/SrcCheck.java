package openacc.codegen.llvmBackend;

/**
 * Interface for a source code validation check.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public abstract class SrcCheck {
  protected final boolean warn;
  protected final boolean warningsAsErrors;
  protected final String msg;

  /**
   * @param warn
   *          whether to report warnings instead of errors
   * @param warningsAsErrors
   *          whether to treat warnings as errors. This is irrelevant if
   *          {@code warn} is false. (Errors are reported if either
   *          {@code warn} is false or both {@code warn} and
   *          {@code warningsAsErrors} are true, but the message in the
   *          latter case points out that a warning is being treated as an
   *          error.)
   * @param msg
   *          the message for any warning or error reported
   */
  public SrcCheck(boolean warn, boolean warningsAsErrors, String msg) {
    this.warn = warn;
    this.warningsAsErrors = warningsAsErrors;
    this.msg = msg;
  }

  /**
   * Same as {@link #SrcCheck(boolean, boolean, String)} except {@code warn}
   * is always false.
   */
  public SrcCheck(String msg) {
    this(false, false, msg);
  }

  /**
   * Run the check.
   */
  public abstract void run();
}