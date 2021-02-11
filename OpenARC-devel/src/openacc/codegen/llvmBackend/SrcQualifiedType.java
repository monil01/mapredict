package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcPrimitiveNonNumericType.SrcVoidType;

import java.lang.ref.WeakReference;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;

import org.jllvm.LLVMContext;

import cetus.hir.Specifier;

/**
 * The LLVM backend's class for all types from the C source that have
 * top-level type qualifiers before typedef expansion. That is,
 * {@link #getSrcTypeQualifiers} never returns an empty set.
 * 
 * <p>
 * These types might also have type qualifiers on target, member, or element
 * types. Moreover, they might have different type qualifiers at the top level
 * after typedef expansion (if one day we actually represent typedef names in
 * this hierarchy). Not only can typedef expansion add type qualifiers, but a
 * {@link SrcQualifiedType} can become a {@link SrcUnqualifiedType} because
 * type qualifiers on an array type are folded into the element type.
 * </p>
 * 
 * <p>
 * By ISO C99 sec. 6.7.3p8, {@link #unqualifiedUnexpanded} never returns an
 * array type or a function type (regardless of typedef expansion). See
 * {@link #get} for details of how we handle these cases.
 * </p>
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public final class SrcQualifiedType extends SrcType {
  private static final HashMap<SrcQualifiedType,
                               WeakReference<SrcQualifiedType>>
    cache = new HashMap<>();

  /**
   * Lower-case versions of these enum members' names are used as the
   * corresponding C keywords in diagnostics.
   */
  public static enum SrcTypeQualifier {CONST, RESTRICT, VOLATILE, NVL, NVL_WP}

  /**
   * After typedef expansion, this might have type qualifiers or be an array
   * type.
   */
  private final SrcUnqualifiedType unqualifiedType;
  private final EnumSet<SrcTypeQualifier> srcTypeQualifiers;

  /**
   * Called only by {@link #get}, which detects some special cases where
   * {@link SrcQualifiedType} is not appropriate and otherwise caches the new
   * type.
   */
  private SrcQualifiedType(SrcType oldType,
                           EnumSet<SrcTypeQualifier> newTypeQualifierSet)
  {
    unqualifiedType = oldType.unqualifiedUnexpanded();
    srcTypeQualifiers = oldType.getSrcTypeQualifiers();
    srcTypeQualifiers.addAll(newTypeQualifierSet);
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + unqualifiedType.hashCode();
    result = prime * result + srcTypeQualifiers.hashCode();
    return result;
  }
  @Override
  public boolean equals(Object obj) {
    if (this == obj) return true;
    if (obj == null) return false;
    if (getClass() != obj.getClass()) return false;
    final SrcQualifiedType other = (SrcQualifiedType)obj;
    if (unqualifiedType != other.unqualifiedType) return false;
    if (!srcTypeQualifiers.equals(other.srcTypeQualifiers)) return false;
    return true;
  }

  /**
   * Get the specified qualified type.
   * 
   * <p>
   * After this method has been called once to build a particular qualified
   * type, it is guaranteed to return the same object for the same qualified
   * type until that object can be garbage-collected because it is no longer
   * referenced outside this class's internal cache.
   * </p>
   * 
   * @param oldType
   *          the type to which to add the specified type qualifiers. It is
   *          fine if {@code oldType} is already a {@link SrcQualifiedType}
   *          and thus has top-level type qualifiers.
   * @param newTypeQualifierSet
   *          the new type qualifiers. When the type qualifiers from
   *          {@code newTypeQualifierSet} and the top level of {@code oldType}
   *          are merged, duplicates are discarded quietly.
   * @return the new type, which is a {@link SrcUnqualifiedType} if either (1)
   *         {@code newTypeQualifierSet} is empty and {@code oldType} has no
   *         top-level type qualifiers or (2) {@code oldType} is a
   *         {@link SrcArrayType}. In the latter case, the new qualifiers are
   *         applied to the array type's element type instead. In any case,
   *         typedefs are not expanded here (if one day we actually represent
   *         typedef names in this hierarchy), so old type qualifiers or an
   *         array type might be hidden behind a typedef, and new type
   *         qualifiers won't be applied to a hidden array type's element type
   *         until typedefs are expanded and the new array type is gotten
   *         using this method.
   * @throws SrcRuntimeException
   *           if {@code oldType} is a function type after typedef expansion
   *           or if the resulting qualified type does not satisfy NVL-related
   *           type qualifier constraints. (Notice that typedef expansion is
   *           performed for the sake of catching these errors even though
   *           it's not performed for the sake of array types.)
   */
  public static SrcType get(SrcType oldType,
                            EnumSet<SrcTypeQualifier> newTypeQualifierSet)
  { 
    // Be careful here to avoid infinite recursions and missed validation:
    //
    // 1. Don't call methods on cacheKey. Just access its fields. The trouble
    //    is that some methods might call this method in order to produce
    //    the same type again, and then an infinite recursion would result.
    //    The place you might be most tempted to call methods on cacheKey is
    //    during validation of the type.
    // 2. In order to be able to avoid this infinite recursion without being
    //    careful about what methods you call here, you might think you can
    //    just insert the type into the cache before validation and then
    //    skip validation if the type was already in the cache. However, then
    //    the test suite would sometimes fail because one test case might
    //    insert a type before validation, and a later test case would then
    //    fail to report any validation error for that type. Whether such an
    //    error occurs depends on the order in which tests are run and how
    //    quickly the java garbage collector cleans up old types.
    // 3. The main method that's caused me trouble with infinite recursion is
    //    hasEffectiveQualifier. However, it's fine to call it on oldType,
    //    which has already been validated, and then check newTypeQualifierSet
    //    for new type qualifiers.
    final SrcQualifiedType cacheKey
      = new SrcQualifiedType(oldType, newTypeQualifierSet);

    // Do not cache SrcUnqualifiedType objects here.
    if (cacheKey.srcTypeQualifiers.isEmpty()) {
      assert(oldType instanceof SrcUnqualifiedType);
      return oldType;
    }

    // ISO C99 sec. 6.7.3p8: type qualifiers on array apply to element.
    //
    // This is one of those rare cases where we use instanceof and Java casts
    // for type comparison and down-casting. The justification here is that we
    // want to implement the above rule from C while maintaining the typedef
    // structure for the sake of error messages.
    if (cacheKey.unqualifiedType instanceof SrcArrayType) {
      final SrcArrayType unqualifiedArrayType
        = (SrcArrayType)cacheKey.unqualifiedType;
      final SrcType elementType = get(unqualifiedArrayType.getElementType(),
                                      cacheKey.srcTypeQualifiers);
      if (unqualifiedArrayType.numElementsIsSpecified())
        return SrcArrayType.get(elementType,
                                unqualifiedArrayType.getNumElements());
      return SrcArrayType.get(elementType);
    }

    // Look up the qualified type in the cache. Validate it only if it's
    // not already in the cache (to save time), and insert it only if it
    // validates (see comments above about the test suite).
    WeakReference<SrcQualifiedType> ref;
    synchronized (cache) {
      ref = cache.get(cacheKey);
    }
    SrcQualifiedType type;
    if (ref == null || (type = ref.get()) == null) {
      // ISO C99 sec. 6.7.3p8: type qualifiers on function produces undefined
      // behavior.
      if (cacheKey.unqualifiedType.iso(SrcFunctionType.class))
        throw new SrcRuntimeException(
          "type qualifiers specified on function type");

      // NVL constraints.
      final boolean
        nvlQual = oldType.hasEffectiveQualifier(SrcTypeQualifier.NVL)
                  || newTypeQualifierSet.contains(SrcTypeQualifier.NVL),
        nvlWpQual = oldType.hasEffectiveQualifier(SrcTypeQualifier.NVL_WP)
                    || newTypeQualifierSet.contains(SrcTypeQualifier.NVL_WP);
      if (nvlQual && nvlWpQual)
        throw new SrcRuntimeException(
          "type has both nvl and nvl_wp type qualifiers");
      final SrcArrayType arrayType
        = cacheKey.unqualifiedType.toIso(SrcArrayType.class);
      if (nvlWpQual
          && !cacheKey.unqualifiedType.iso(SrcPointerType.class)
          && (arrayType == null
              || !arrayType.getElementType().iso(SrcPointerType.class)))
        throw new SrcRuntimeException(
          "non-pointer type has nvl_wp type qualifier");
      if (nvlQual || nvlWpQual) {
        if (cacheKey.unqualifiedType.iso(SrcVoidType))
          throw new SrcRuntimeException("void type is NVM-stored");
        if (cacheKey.unqualifiedType.iso(SrcUnionType.class))
          throw new SrcRuntimeException("union type is NVM-stored");
        new SrcStorageHasNoNonNVMTargetCheck(
          cacheKey.unqualifiedType,
          "NVM-stored pointer type has non-NVM-stored target type")
        .run();
        final SrcStructOrUnionType structOrUnion
          = cacheKey.unqualifiedType.toIso(SrcStructOrUnionType.class);
        if (structOrUnion != null)
          structOrUnion.recordNVMStoredUse();
      }

      type = new SrcQualifiedType(oldType, newTypeQualifierSet);
      ref = new WeakReference<>(type);
      synchronized (cache) {
        cache.put(cacheKey, ref);
      }
    }
    return type;
  }

  /**
   * Same as {@link #get(SrcType, EnumSet)} except type qualifiers are passed
   * as a list of {@link SrcTypeQualifier}s, which can contain duplicates,
   * which are ignored.
   */
  public static SrcType get(SrcType oldType,
                            SrcTypeQualifier... srcTypeQualifiers)
  {
    return get(oldType,
               EnumSet.copyOf(java.util.Arrays.asList(srcTypeQualifiers)));
  }

  /**
   * Same as {@link #get(SrcType, EnumSet)} except type qualifiers are
   * passed as a list of {@link Specifier}s, which can contain duplicates,
   * which are ignored.
   * 
   * @throws IllegalStateException
   *           if {@code typeQualifiers} contains specifiers that are not type
   *           qualifiers
   */
  public static SrcType get(SrcType oldType, Specifier... typeQualifiers) {
    final EnumSet<SrcTypeQualifier> srcTypeQualifiers
      = EnumSet.noneOf(SrcTypeQualifier.class);
    for (Specifier s : typeQualifiers) {
      if (s == Specifier.CONST)
        // TODO: useful for LLVM global const?
        srcTypeQualifiers.add(SrcTypeQualifier.CONST);
      else if (s == Specifier.RESTRICT)
        // TODO: handle
        srcTypeQualifiers.add(SrcTypeQualifier.RESTRICT);
      else if (s == Specifier.VOLATILE)
        // TODO: volatile specifier will have to be handled in
        // load/store/memcpy instructions operating on these types.
        srcTypeQualifiers.add(SrcTypeQualifier.VOLATILE);
      else if (s == Specifier.NVL)
        srcTypeQualifiers.add(SrcTypeQualifier.NVL);
      else if (s == Specifier.NVL_WP)
        srcTypeQualifiers.add(SrcTypeQualifier.NVL_WP);
      else throw new IllegalStateException("unknown specifier: "
                                           + s.toString());
    }
    return get(oldType, srcTypeQualifiers);
  }

  @Override
  protected void finalize() {
    synchronized (cache) {
      if (cache.get(this).get() == null)
        cache.remove(this);
    }
  }

  @Override
  public SrcUnqualifiedType unqualifiedUnexpanded() {
    return unqualifiedType;
  }
  @Override
  public EnumSet<SrcTypeQualifier> getSrcTypeQualifiers() {
    return srcTypeQualifiers.clone();
  }
  @Override
  protected SrcTopLevelExpandedType expandTopLevelTypedefs(
    EnumSet<SrcTypeQualifier> addQuals)
  {
    final EnumSet<SrcTypeQualifier> quals = srcTypeQualifiers.clone();
    quals.addAll(addQuals);
    return unqualifiedType.expandTopLevelTypedefs(quals);
  }

  @Override
  public String toString(String nestedDecl, Set<SrcType> skipTypes) {
    final StringBuilder str = new StringBuilder();
    for (Iterator<SrcTypeQualifier> itr = srcTypeQualifiers.iterator();
         itr.hasNext();)
    {
      str.append(itr.next().name().toLowerCase());
      if (itr.hasNext() || !nestedDecl.isEmpty())
        str.append(" ");
    }
    str.append(nestedDecl);
    return unqualifiedType.toString(str.toString(), skipTypes);
  }

  @Override
  public String toCompatibilityString(Set<SrcStructOrUnionType> structSet,
                                      LLVMContext ctxt)
  {
    final StringBuilder str = new StringBuilder(
      unqualifiedType.toCompatibilityString(structSet, ctxt));
    for (Iterator<SrcTypeQualifier> itr = srcTypeQualifiers.iterator();
         itr.hasNext();)
    {
      str.append(" ");
      str.append(itr.next().name().toLowerCase());
    }
    return str.toString();
  }
}
