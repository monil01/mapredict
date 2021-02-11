package openacc.codegen.llvmBackend;

import openacc.codegen.llvmBackend.SrcQualifiedType.SrcTypeQualifier;

/**
 * A check that storage for a type's objects do not contain pointers to
 * non-NVM storage.
 * 
 * <p>
 * That is, this check checks every component of a type's definition that
 * corresponds to storage of objects of the type (see {@code storageOnly}
 * parameter of {@link SrcType#iterator}) and reports a warning or error if
 * it finds one that is a pointer with a non-NVM stored target type. If it
 * encounters an incomplete type, it defers the check for that type until
 * that type is completed. If that type is never completed, then the check
 * never runs on that type.
 * </p>
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public final class SrcStorageHasNoNonNVMTargetCheck extends SrcCheck {
  final SrcType srcType;

  /**
   * See
   * {@link SrcCheck#SrcTypeCheck(SrcType, boolean, boolean, String)}
   * for other parameter documentation.
   * 
   * @param srcType is the type on which to perform the check
   */
  public SrcStorageHasNoNonNVMTargetCheck(
    SrcType srcType, boolean warn, boolean warningsAsErrors, String msg)
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
  public SrcStorageHasNoNonNVMTargetCheck(SrcType srcType, String msg) {
    super(msg);
    this.srcType = srcType;
  }

  @Override public void run() {
    for (final SrcTypeIterator i = srcType.iterator(true);
         i.hasNext();)
    {
      try {
        final SrcPointerType p = i.next().toIso(SrcPointerType
                                                .class);
        if (p != null
            && !p.getTargetType()
               .hasEffectiveQualifier(SrcTypeQualifier.NVL,
                                      SrcTypeQualifier.NVL_WP))
          // We continue checking after the first warning so that we report
          // every offending piece of storage.
          BuildLLVM.warnOrError(warn, warningsAsErrors, msg);
      }
      catch (SrcCompletableTypeException e) {
        e.getType().checkAtTypeCompletion(
          new SrcStorageHasNoNonNVMTargetCheck(e.getType(), warn,
                                               warningsAsErrors, msg));
      }
    }
  }
}