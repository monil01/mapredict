package openacc.codegen.llvmBackend;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Set;

/**
 * An iterator that visits a type and then its component types, recursively
 * and depth-first.
 * 
 * <p>
 * The first call to {@link #hasNext} is guaranteed to return true.
 * </p>
 * 
 * <p>
 * To avoid the possibility of infinite recursion, if a pointer target type is
 * a struct or union type that has been visited before, then it is skipped.
 * </p>
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public class SrcTypeAndComponentsIterator implements SrcTypeIterator {
  private final SrcType type;
  private final boolean storageOnly;
  private final Set<SrcType> skipTypes;
  private SrcTypeIterator childItr;

  /**
   * @param type
   *          the top-level type
   * @param storageOnly
   *          if false, then recursively visit all types referenced in the
   *          definition of {@code type}. Otherwise, visit only types that
   *          correspond to components of the <em>storage</em> of objects of
   *          {@code type} (that is, member types in the case of a struct or
   *          union type, and the element type in the case of an array type).
   *          In the latter case, this type must not be a function type or
   *          void type, neither of which can be allocated as an object.
   *          Moreover, a pointer type's target type does not require storage
   *          when the pointer type is allocated, so pointer target types are
   *          not visited. For these reasons, function types, parameter types,
   *          and return type are unreachable.
   */
  public SrcTypeAndComponentsIterator(SrcType type, boolean storageOnly)
  {
    this(type, storageOnly,
         Collections.newSetFromMap(new IdentityHashMap<SrcType, Boolean>()));
  }

  /**
   * Same as {@link #SrcTypeAndComponentsIterator(SrcType, boolean)} except
   * also specify a set of types to skip as necessary to avoid infinite
   * recursion. The contents of this set are modified during the iteration.
   */
  public SrcTypeAndComponentsIterator(SrcType type, boolean storageOnly,
                                      Set<SrcType> skipTypes)
  {
    this.type = type;
    this.storageOnly = storageOnly;
    this.skipTypes = skipTypes;
    this.childItr = null;
  }

  @Override
  public boolean hasNext() {
    return childItr == null || childItr.hasNext();
  }

  @Override
  public SrcType next() throws SrcCompletableTypeException {
    if (childItr == null) {
      childItr = type.toIso(SrcBaldType.class)
                 .componentIterator(storageOnly, skipTypes);
      return type;
    }
    return childItr.next();
  }
}