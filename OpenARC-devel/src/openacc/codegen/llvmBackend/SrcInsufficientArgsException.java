package openacc.codegen.llvmBackend;

/**
 * An exception indicating that a function builtin call received too few
 * actual source-level arguments but that it is the caller's responsibility
 * to build the error message.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public class SrcInsufficientArgsException extends SrcException {
  private static final long serialVersionUID = 1L;
  public SrcInsufficientArgsException() {}
}