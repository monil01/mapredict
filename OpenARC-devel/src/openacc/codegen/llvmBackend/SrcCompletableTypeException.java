package openacc.codegen.llvmBackend;

/**
 * An exception indicating that a type was encountered that is currently
 * incomplete but could be completed later, so the caller might need to
 * arrange for the type to be handled then.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public class SrcCompletableTypeException extends SrcException {
  private static final long serialVersionUID = 1L;
  private final SrcStructOrUnionType type;
  public SrcCompletableTypeException(String message, SrcStructOrUnionType type)
  {
    super(message);
    this.type = type;
  }
  /** What incomplete type was encountered? */
  public SrcStructOrUnionType getType() {
    return type;
  }
}