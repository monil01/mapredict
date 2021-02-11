package openacc.codegen.llvmBackend;


/**
 * Thrown when {@link BuildLLVM} encounters source that is not valid C. For
 * unsupported C features, {@link UnsupportedOperationException} is thrown
 * instead.
 *
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class SrcRuntimeException extends RuntimeException {
  private static final long serialVersionUID = 1L;
  private boolean hasLine;
  private String bareMessage; // without line
  public SrcRuntimeException(String message) {
    super(message);
    this.hasLine = false;
    this.bareMessage = message;
  }
  /** line=-1 means unknown. */
  public SrcRuntimeException(SrcRuntimeException e, int line) {
    // Our heuristic is that if the line number is already set, it was set
    // within a descendant node's visit call, which gives a more specific
    // location than the ancestor node.
    super(e.hasLine || line == -1
          ? e.getMessage()
          : "within statement starting or ending on line "+line+": "
            +e.bareMessage);
    this.hasLine = true;
    this.bareMessage = e.bareMessage;
    setStackTrace(e.getStackTrace());
  }
}