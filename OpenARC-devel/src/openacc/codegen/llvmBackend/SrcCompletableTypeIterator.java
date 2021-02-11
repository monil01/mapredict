package openacc.codegen.llvmBackend;

import java.util.NoSuchElementException;

/**
 * An iterator that visits one (conceptual) item such that, at that item,
 * {@link #next} throws an {@link SrcCompletableTypeException}. Afterward the
 * iterator behaves as if there are no items remaining.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public class SrcCompletableTypeIterator implements SrcTypeIterator {
  private final SrcStructOrUnionType completableType;
  private boolean thrown = false;

  /**
   * @param completableType
   *          the incomplete type to pass to the exception's constructor
   */
  public SrcCompletableTypeIterator(SrcStructOrUnionType completableType) {
    assert(completableType.isIncompleteType());
    this.completableType = completableType;
  }

  @Override
  public boolean hasNext() {
    return !thrown;
  }

  @Override
  public SrcType next() throws SrcCompletableTypeException {
    if (thrown)
      throw new NoSuchElementException();
    thrown = true;
    throw new SrcCompletableTypeException(
      "iteration of incomplete struct or union type",
      completableType);
  }
}