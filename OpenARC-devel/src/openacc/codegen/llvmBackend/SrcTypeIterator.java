package openacc.codegen.llvmBackend;

import java.util.NoSuchElementException;

/**
 * An iterator that visits a sequence of {@link SrcType}s.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public interface SrcTypeIterator {
  /** Does the iteration have any more types to visit? */
  public boolean hasNext();

  /**
   * Get the next type in the iteration.
   *
   * @throws NoSuchElementException
   *           if the iteration has no more elements
   * @throws SrcCompletableTypeException
   *           if type components of an incomplete
   *           {@link SrcStructOrUnionType} are to be iterated next. Any
   *           subsequent {@link #next} call on this iterator will proceed to
   *           any types that would have been iterated after those type
   *           components, or it will throw {@link NoSuchElementException} if
   *           there are none.
   */
  public SrcType next() throws SrcCompletableTypeException;
}