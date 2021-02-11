package openacc.codegen.llvmBackend;


/**
 * Same as {@link SrcRuntimeException} except that callers should be careful
 * to catch it, often in order to throw a {@link SrcException} or
 * {@link SrcRuntimeException} with a more specific message.
 *
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class SrcException extends Exception {
  private static final long serialVersionUID = 1L;
  public SrcException() {}
  public SrcException(String message) { super(message); }
}