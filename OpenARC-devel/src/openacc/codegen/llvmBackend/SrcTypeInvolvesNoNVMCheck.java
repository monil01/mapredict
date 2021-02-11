package openacc.codegen.llvmBackend;

import openacc.codegen.llvmBackend.SrcQualifiedType.SrcTypeQualifier;

/**
 * A check that a type doesn't involve NVM storage in any way.
 * 
 * <p>
 * That is, this check checks every component of a type's definition for an
 * {@code nvl} qualifier and reports a warning or error if it finds one. If
 * it encounters an incomplete type, it defers the check for that type until
 * that type is completed. If that type is never completed, then the check
 * never runs on that type.
 * </p>
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public final class SrcTypeInvolvesNoNVMCheck extends SrcCheck {
  final SrcType srcType;

  /**
   * See
   * {@link SrcCheck#SrcTypeCheck(SrcType, boolean, boolean, String)}
   * for parameter documentation.
   * 
   * @param srcType is the type on which to perform the check
   */
  public SrcTypeInvolvesNoNVMCheck(SrcType srcType, boolean warn,
                                   boolean warningsAsErrors, String msg)
  {
    super(warn, warningsAsErrors, msg);
    this.srcType = srcType;
  }
  /**
   * See {@link SrcCheck#SrcTypeCheck(SrcType, String)} for parameter
   * documentation.
   * 
   * @param srcType is the type on which to perform the check
   */
  public SrcTypeInvolvesNoNVMCheck(SrcType srcType, String msg) {
    super(msg);
    this.srcType = srcType;
  }

  @Override
  public final void run() {
    // Keep implementation in sync with
    // SrcTypesCompatibleOrInvolveNoNVMCheck.
    for (final SrcTypeIterator i = srcType.iterator(false); i.hasNext();) {
      try {
        // There's no need to check for nvl_wp because, if such a pointer is
        // present, its target type must have nvl, which will be found.
        if (i.next().expandSrcTypeQualifiers()
            .contains(SrcTypeQualifier.NVL))
        {
          BuildLLVM.warnOrError(warn, warningsAsErrors, msg);
          // We stop at the first warning so that, for example, we don't
          // warn about a pointer and its target type, and that type's
          // target type, etc. The first warning ought to be sufficient to
          // signal we've found NVM storage.
          return;
        }
      }
      catch (SrcCompletableTypeException e) {
        e.getType().checkAtTypeCompletion(
          new SrcTypeInvolvesNoNVMCheck(e.getType(), warn, warningsAsErrors,
                                        msg));
      }
    }
  }
}