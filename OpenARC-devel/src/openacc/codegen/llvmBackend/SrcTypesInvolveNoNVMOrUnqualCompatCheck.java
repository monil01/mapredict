package openacc.codegen.llvmBackend;

import openacc.codegen.llvmBackend.SrcQualifiedType.SrcTypeQualifier;

/**
 * A check that two types don't involve NVM storage in any way or their
 * unqualified versions are compatible.
 * 
 * <p>
 * That is, this check checks every component of two type definitions for an
 * {@code nvl} qualifier and then reports a warning or error if it finds one
 * and the two types' unqualified versions are not compatible. If it
 * encounters an incomplete type, it copies the entire check to be repeated
 * when that incomplete type is completed. If no such incomplete type is
 * ever completed, then the check never repeats.
 * </p>
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public final class SrcTypesInvolveNoNVMOrUnqualCompatCheck extends SrcCheck
{
  final SrcType srcType0;
  final SrcType srcType1;

  /**
   * See
   * {@link SrcCheck#SrcTypeCheck(SrcType, boolean, boolean, String)}
   * for parameter documentation.
   * 
   * @param srcType0 is the first type on which to perform the check
   * @param srcType1 is the second type on which to perform the check
   */
  public SrcTypesInvolveNoNVMOrUnqualCompatCheck(
    SrcType srcType0, SrcType srcType1, boolean warn,
    boolean warningsAsErrors, String msg)
  {
    super(warn, warningsAsErrors, msg);
    this.srcType0 = srcType0;
    this.srcType1 = srcType1;
  }
  /**
   * See {@link SrcCheck#SrcTypeCheck(SrcType, String)} for parameter
   * documentation.
   * 
   * @param srcType0 is the first type on which to perform the check
   * @param srcType1 is the second type on which to perform the check
   */
  public SrcTypesInvolveNoNVMOrUnqualCompatCheck(
    SrcType srcType0, SrcType srcType1, String msg)
  {
    super(msg);
    this.srcType0 = srcType0;
    this.srcType1 = srcType1;
  }

  @Override
  public final void run() {
    // Keep implementation for checking for no NVM storage in sync with
    // SrcTypeInvolvesNoNVMCheck. One difference is that this check copies
    // the entire check to be repeated if an incomplete type is found, but
    // SrcTypeInvolvesNoNVMCheck just defers the check for the incomplete
    // type.
    for (final SrcType srcType : new SrcType[]{srcType0, srcType1}) {
      for (final SrcTypeIterator i = srcType.iterator(false); i.hasNext();)
      {
        try {
          if (i.next().expandSrcTypeQualifiers()
              .contains(SrcTypeQualifier.NVL))
          {
            srcType0.toIso(SrcBaldType.class).checkCompatibility(
              srcType1.toIso(SrcBaldType.class),
              warn, warningsAsErrors, msg);
            return;
          }
        }
        catch (SrcCompletableTypeException e) {
          e.getType().checkAtTypeCompletion(this);
        }
      }
    }
  }
}