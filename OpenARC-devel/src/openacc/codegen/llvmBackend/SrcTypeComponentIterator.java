package openacc.codegen.llvmBackend;

import java.util.NoSuchElementException;
import java.util.Set;

/**
 * An iterator that visits the component types returned by calling
 * {@link SrcType#iterator} on each type in a specified sequence of types.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public class SrcTypeComponentIterator implements SrcTypeIterator {
  private final boolean storageOnly;
  private final Set<SrcType> skipTypes;
  private final SrcType[][] srcTypes;
  private int row = 0;
  private int col = -1;
  private SrcTypeIterator childItr = null;

  /**
   * @param storageOnly
   *          a parameter for {@link SrcType#iterator} calls
   * @param skipTypes
   *          a parameter for {@link SrcType#iterator} calls
   * @param srcTypes
   *          the series of types on which to call {@link SrcType#iterator}
   */
  public SrcTypeComponentIterator(
    boolean storageOnly, Set<SrcType> skipTypes, SrcType[]... srcTypes)
  {
    this.storageOnly = storageOnly;
    this.skipTypes = skipTypes;
    this.srcTypes = srcTypes;
    advance();
  }

  @Override
  public boolean hasNext() {
    return row != srcTypes.length;
  }

  @Override
  public SrcType next() throws SrcCompletableTypeException {
    if (!hasNext())
      throw new NoSuchElementException();
    SrcType childType;
    try {
      childType = childItr.next();
    }
    finally {
      advance();
    }
    return childType;
  }

  private void advance() {
    while (row < srcTypes.length && (childItr == null
                                     || !childItr.hasNext()))
    {
      if (++col < srcTypes[row].length)
        childItr = srcTypes[row][col].iterator(storageOnly, skipTypes);
      else {
        ++row;
        col = -1;
      }
    }
  }
}