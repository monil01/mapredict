package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SRC_PTRDIFF_T_TYPE;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcIntType;
import static openacc.codegen.llvmBackend.SrcPrimitiveNonNumericType.SrcVoidType;

import java.util.Collections;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Set;

import openacc.codegen.llvmBackend.SrcQualifiedType.SrcTypeQualifier;
import openacc.codegen.llvmBackend.ValueAndType.AssignKind;

import org.jllvm.LLVMAndInstruction;
import org.jllvm.LLVMContext;
import org.jllvm.LLVMExtendCast;
import org.jllvm.LLVMExtendCast.ExtendType;
import org.jllvm.LLVMInstructionBuilder;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMOrInstruction;
import org.jllvm.LLVMShiftInstruction;
import org.jllvm.LLVMShiftInstruction.ShiftType;
import org.jllvm.LLVMTargetData;
import org.jllvm.LLVMType;
import org.jllvm.LLVMValue;
import org.jllvm.LLVMXorInstruction;

import cetus.hir.Traversable;

/**
 * The LLVM backend's base class for all types that can be declared in the C
 * source.
 * 
 * <p>
 * So far, we do not represent typedef names. Whenever the type of something
 * is a typedef name, the LLVM backend instead stores the type to which that
 * typedef name resolves as that's simpler to examine. See
 * {@link SrcTypedefType}.
 * </p>
 * 
 * <p>
 * For type comparisons and down-casting, use {@link #eqv}, {@link #iso},
 * {@link #toEqv}, and {@link #toIso} methods, which conveniently handle type
 * qualifiers and typedef names (if one day we actually represent typedef
 * names in this hierarchy), and which could be extended to handle other type
 * comparison issues that might one day arise. We never use {@code ==},
 * {@code !=}, {@code equals}, {@code instanceof}, or Java casts for type
 * comparisons and down-casting except in a few special cases: (1) they are
 * used in the implementations of the {@link #toEqv} and {@link #toIso}
 * methods, (2) they are used in some internal code for type caching and for
 * constructing {@link SrcQualifiedType} objects, (3) {@code instanceof} is
 * used in some {@code toString} methods to help represent the exact typedef
 * structure of a type, and (4) {@code ==} and {@code !=} are used to optimize
 * the special case of identical types before a deep comparison of types
 * begins (for example, see {@link #buildComposite}).
 * </p>
 * 
 * <p>
 * Even when {@code ==}, {@code !=}, {@code equals}, {@code instanceof}, or a
 * Java cast is perfectly sufficient for a particular type comparison or
 * down-cast, use {@link #eqv} or {@link #toEqv} instead as we will not be
 * able to alter the behavior of the former operators if we one day need to
 * adjust type comparison and down-cast behavior throughout the LLVM backend.
 * For readability, when type qualifiers are guaranteed not to be present, we
 * prefer {@link #eqv} and {@link #toEqv} over {@link #iso} and {@link #toIso}
 * as the latter imply that type qualifiers are sometimes present and need to
 * be ignored.
 * </p>
 * 
 * <p>
 * Types are uniqued to save memory and so that they can be compared by
 * reference in the special cases described above. Primitive types are stored
 * as static final members of their associated classes. For array, function,
 * and pointer types, we have factory methods named {@code get} that retrieve
 * from an object cache. For enum, struct, and union types, each new type is
 * distinct, so we have public constructors. We consider each bit-field to
 * have a unique {@link SrcBitFieldType} because it never proves useful to
 * check for identical bit-field types (besides, {@link SrcBitFieldType}
 * stores the bit-field's offset within its storage unit, and that seems
 * irrelevant to type comparisons).
 * </p>
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public abstract class SrcType extends SrcParamAndReturnType.Impl {
  @Override
  public SrcType getCallSrcType(
    Traversable callNode, String calleeName, String paramName,
    ValueAndType args[], SrcType typeArg, SrcSymbolTable srcSymbolTable,
    LLVMModule llvmModule, boolean warningsAsErrors)
  {
    return this;
  }

  /**
   * After expanding all typedefs throughout both types, is this type
   * identical to the specified type? If so, return the specified type.
   * 
   * <p>
   * For example, given types {@code t1} and {@code t2} that expand to the
   * type {@code int const * const volatile}, and given a type {@code t3} that
   * expands to the type {@code int const * const}, then {@code t1.toEqv(t2)}
   * returns t2, but {@code t1.toEqv(t3)} returns null.
   * </p>
   * 
   * <p>
   * Do not assume it is safe to use this method instead of {@link #toIso(T)}
   * just because you're working with instances of {@link SrcUnqualifiedType},
   * which might have type qualifiers after typedef expansion. See
   * {@link SrcUnqualifiedType} documentation for details.
   * </p>
   * 
   * <p>
   * TODO: All eqv-related methods and {@link #expandTypeQualifiers} could be
   * more performant if, for every type we construct, we store a fully
   * typedef-expanded version of the type (canonical type?) along with the
   * type. Then eqv would require just a simple ==.
   * </p>
   * 
   * @param other
   *          the type with which to compare
   * @return {@code other}, or null if the eqv relation does not hold
   */
  public final <T extends SrcType> T toEqv(T other) {
    final SrcTopLevelExpandedType
      thisExpanded = expandTopLevelTypedefs(),
      otherExpanded = other.expandTopLevelTypedefs();
    if (!thisExpanded.getSrcTypeQualifiers()
        .equals(otherExpanded.getSrcTypeQualifiers()))
      return null;
    final SrcBaldType thisBaldType = thisExpanded.bald(),
                      otherBaldType = otherExpanded.bald();
    if (!thisBaldType.eqvBald(otherBaldType))
      return null;
    return other;
  }
  /**
   * Wrapper around {@link #toEqv(SrcType)} that returns true iff it returns
   * non-null.
   */
  public final boolean eqv(SrcType other) {
    return null != toEqv(other);
  }

  /**
   * After expanding all typedefs throughout both types and then discarding
   * top-level type qualifiers from both types, is this type identical to the
   * specified type? If so, return the specified type.
   * 
   * <p>
   * For example, given a type {@code t1} that expands to the type
   * {@code int const * const volatile}, and given a type {@code t2} that
   * expands to the type {@code int const * const}, then {@code t1.toIso(t2)}
   * returns t2.
   * </p>
   * 
   * <p>
   * In the case of array types, the result is the same as for {@link @toEqv}
   * because top-level type qualifiers are folded into array element types.
   * </p>
   * 
   * @param other
   *          the type with which to compare
   * @return {@code other}, or null if the iso relation does not hold
   */
  public final <T extends SrcType> T toIso(T other) {
    if (!expandTopLevelTypedefs().bald()
        .eqvBald(other.expandTopLevelTypedefs().bald()))
      return null;
    return other;
  }
  /**
   * Wrapper around {@link #toIso(SrcType)} that returns true iff it returns
   * non-null.
   */
  public final boolean iso(SrcType other) {
    return null != toIso(other);
  }

  /**
   * After expanding top-level typedefs, is this type an instance of the
   * specified type class? If so, return the so-expanded type.
   * 
   * <p>
   * Expanding typedefs is performed only at the top level. That is, expanding
   * typedefs is not performed on member, target, or element types. Type
   * qualifiers are not discarded to uncover the type except that, in the case
   * of an array type, type qualifiers are folded into the array element type.
   * </p>
   * 
   * <p>
   * For example, given a typedef {@code T}, given a typedef {@code t1} that
   * expands to the type {@code T const *}, and given a typedef {@code t2}
   * that expands to the type {@code T const * const}, then
   * {@code t1.toEqv(SrcPointerType.class)} returns a {@link SrcPointerType}
   * that represents the type {@code T const *}, but
   * {@code t2.toEqv(SrcPointerType.class)} returns null.
   * </p>
   * 
   * <p>
   * Do not assume it is safe to use this method instead of
   * {@link #toIso(Class)} just because you're working with instances of
   * {@link SrcUnqualifiedType}, which might have type qualifiers after
   * typedef expansion. See {@link SrcUnqualifiedType} documentation for
   * details.
   * </p>
   * 
   * <p>
   * This method uses reflection to make this class hierarchy easier to
   * maintain. To improve performance, we could replace this method with a set
   * of methods that don't use reflection, one method per subtype. That is,
   * {@code eqvPointerCast()}, {@code eqvFunctionCast()},
   * {@code eqvStructCast()}, etc.
   * </p>
   * 
   * @param typeClass
   *          the type class with which to compare
   * @return this type with top-level typedefs expanded, or null if the eqv
   *         relation does not hold
   */
  public final <T extends SrcBaldType> T toEqv(Class<T> typeClass) {
    final SrcTopLevelExpandedType expanded = expandTopLevelTypedefs();
    if (!expanded.getSrcTypeQualifiers().isEmpty())
      return null;
    final SrcBaldType bald = expanded.bald();
    return typeClass.isInstance(bald) ? typeClass.cast(bald) : null;
  }
  /**
   * Wrapper around {@link #toEqv(Class)} that returns true iff it returns
   * non-null.
   */
  public final boolean eqv(Class<? extends SrcBaldType> typeClass) {
    return null != toEqv(typeClass);
  }

  /**
   * After expanding top-level typedefs and then discarding top-level type
   * qualifiers, is this type an instance of the specified type class? If so,
   * return the so-expanded type.
   * 
   * <p>
   * Expanding typedefs and then discarding type qualifiers is performed only
   * at the top level. That is, expanding typedefs and discarding type
   * qualifiers is not performed on member, target, or element types. In the
   * case of an array type, top-level type qualifiers are folded into the
   * array element type rather than discarded.
   * </p>
   * 
   * <p>
   * For example, given a typedef {@code T}, and given a typedef {@code t}
   * that expands to the type {@code T const * const}, then
   * {@code t.toIso(SrcPointerType.class)} returns a {@link SrcPointerType}
   * that represents the type {@code T const *}.
   * </p>
   * 
   * <p>
   * This method uses reflection to make this class hierarchy easier to
   * maintain. To improve performance, we could replace this method with a set
   * of methods that don't use reflection, one method per subtype. That is,
   * {@code isoPointerCast()}, {@code isoFunctionCast()},
   * {@code isoStructCast()}, etc.
   * </p>
   * 
   * @param typeClass
   *          the type class with which to compare
   * @return this type with top-level typedefs expanded and top-level type
   *         qualifiers then discarded, or null if the iso relation does not
   *         hold
   */
  public final <T extends SrcBaldType> T toIso(Class<T> typeClass) {
    final SrcBaldType bald = expandTopLevelTypedefs().bald();
    return typeClass.isInstance(bald) ? typeClass.cast(bald) : null;
  }
  /**
   * Wrapper around {@link #toIso(Class)} that returns true iff it returns
   * non-null.
   */
  public final boolean iso(Class<? extends SrcBaldType> typeClass) {
    return null != toIso(typeClass);
  }

  /**
   * Get the type's top-level type qualifiers after typedef expansion.
   * 
   * <p>
   * In the case of an array type, type qualifiers are folded into the array
   * element type, so no type qualifiers remain.
   * </p>
   * 
   * @return the possibly empty type qualifier set. The object returned is
   *         never the object stored in any type, so it can be modified safely
   *         by the caller.
   */
  public final EnumSet<SrcTypeQualifier> expandSrcTypeQualifiers() {
    return expandTopLevelTypedefs().getSrcTypeQualifiers();
  }

  /**
   * Get this type without top-level type qualifiers and without typedef
   * expansion.
   * 
   * <p>
   * For example, {@code int const * const} becomes {@code int const *}.
   * </p>
   * 
   * <p>
   * This does not expand typedefs (if one day we actually represent typedef
   * names in this hierarchy), which might hide type qualifiers. To expand
   * top-level typedefs and then discard top-level type qualifiers, use
   * {@link #toIso(Class)} with argument {@link SrcBaldType SrcBaldType.class}
   * .
   * </p>
   * 
   * @return the type without top-level type qualifiers and without typedef
   *         expansion
   */
  public abstract SrcUnqualifiedType unqualifiedUnexpanded();

  /**
   * Get this type's top-level type qualifiers before typedef expansion.
   * 
   * <p>
   * This does not expand typedefs (if one day we actually represent typedef
   * names in this hierarchy), which might hide additional type qualifiers.
   * This also does not merge type qualifiers into array element types. To
   * expand top-level typedefs, merge in the case of array types, and then get
   * the top-level type qualifiers, use {@link #expandSrcTypeQualifiers}.
   * </p>
   * 
   * @return the type qualifier set. If this is a {@link SrcUnqualifiedType},
   *         the set is always empty. If this is a {@link SrcQualifiedType},
   *         the set is always non-empty, and a clone of the set is returned
   *         so that the set stored in this cannot be modified by the caller.
   */
  public abstract EnumSet<SrcTypeQualifier> getSrcTypeQualifiers();

  /**
   * A {@link SrcType} with top-level typedefs expanded. Meant to used only
   * by {@link #expandTopLevelTypedefs} and its callers.
   */
  public static class SrcTopLevelExpandedType {
    private final SrcBaldType srcBaldType;
    private final EnumSet<SrcTypeQualifier> srcTypeQualifiers;
    public SrcTopLevelExpandedType(
      SrcBaldType srcBaldType,
      EnumSet<SrcTypeQualifier> srcTypeQualifiers)
    {
      this.srcBaldType = srcBaldType;
      this.srcTypeQualifiers = srcTypeQualifiers;
    }
    public SrcBaldType bald() {
      return srcBaldType;
    }
    public EnumSet<SrcTypeQualifier> getSrcTypeQualifiers() {
      return srcTypeQualifiers;
    }
  }

  /**
   * Expand top-level typedefs. Thus, {@link #unqualifiedUnexpanded} called on
   * the result will always return a {@link SrcBaldType}. This method is meant
   * to be called only in the implementations of {@link SrcType} methods,
   * which other code should call instead.
   */
  protected final SrcTopLevelExpandedType expandTopLevelTypedefs() {
    return expandTopLevelTypedefs(EnumSet.noneOf(SrcTypeQualifier.class));
  }
  /**
   * This method is meant to be called only by its implementations and by
   * {@link #expandTopLevelTypedefs()}.
   */
  protected abstract SrcTopLevelExpandedType expandTopLevelTypedefs(
    EnumSet<SrcTypeQualifier> addQuals);

  /**
   * Is this an incomplete type? (ISO C99 sec. 6.2.5p1.)
   * 
   * <p>
   * This method should be overridden only by subtypes of {@link SrcBaldType}
   * as the implementation here handles typedefs and type qualifiers.
   * </p>
   */
  public boolean isIncompleteType() {
    return expandTopLevelTypedefs().bald().isIncompleteType();
  }

  /**
   * Is this an object type? (ISO C99 sec. 6.2.5p1.)
   */
  public final boolean isObjectType() {
    return !isIncompleteType() && !iso(SrcFunctionType.class);
  }

  /**
   * Does this type effectively have any of the specified qualifiers?
   * 
   * <p>
   * That is, after expanding typedefs, check this type and, if it's an array
   * type, check its element type, into which an array type's type qualifiers
   * are always folded.
   * </p>
   * 
   * <p>
   * This method should be overridden only by subtypes of {@link SrcBaldType}
   * as the implementation here handles top-level typedefs and type
   * qualifiers.
   * </p>
   */
  public boolean hasEffectiveQualifier(SrcTypeQualifier... quals) {
    final SrcTopLevelExpandedType expandedType = expandTopLevelTypedefs();
    for (final SrcTypeQualifier qual : quals) {
      if (expandedType.srcTypeQualifiers.contains(qual))
        return true;
    }
    return expandedType.bald().hasEffectiveQualifier(quals);
  }

  /**
   * Return this type but without any of the specified qualifiers as
   * effective qualifiers.
   * 
   * <p>
   * That is, after expanding typedefs, remove the specified qualifiers from
   * this type and, if it's an array type, remove them from its element
   * type, into which an array type's type qualifiers are always folded.
   * </p>
   * 
   * <p>
   * If none of the specified qualifiers are effective qualifiers, this type
   * is returned possibly with some typedefs expanded.
   * </p>
   * 
   * <p>
   * This method should be overridden only by subtypes of
   * {@link SrcBaldType} as the implementation here handles top-level
   * typedefs and type qualifiers.
   * </p>
   */
  public SrcType withoutEffectiveQualifier(SrcTypeQualifier... quals) {
    final SrcTopLevelExpandedType expandedType = expandTopLevelTypedefs();
    final EnumSet<SrcTypeQualifier> newQuals
      = expandedType.getSrcTypeQualifiers().clone();
    for (final SrcTypeQualifier qual : quals)
      newQuals.remove(qual);
    return SrcQualifiedType.get(
      expandedType.bald().withoutEffectiveQualifier(quals),
      newQuals);
  }

  /**
   * Wrapper that passes this type to
   * {@link SrcTypeAndComponentsIterator#SrcTypeAndComponentsIterator(SrcType, boolean)}.
   */
  public final SrcTypeIterator iterator(boolean storageOnly) {
    return new SrcTypeAndComponentsIterator(this, storageOnly);
  }

  /**
   * Wrapper that passes this type to
   * {@link SrcTypeAndComponentsIterator#SrcTypeAndComponentsIterator(SrcType, boolean, Set)}.
   */
  public final SrcTypeIterator iterator(boolean storageOnly,
                                        Set<SrcType> skipTypes)
  {
    return new SrcTypeAndComponentsIterator(this, storageOnly, skipTypes);
  }

  /**
   * When an object of this type is allocated, is any part of that object's
   * storage qualified with the specified type qualifier?
   * 
   * <p>
   * This type, member types in the case of a struct or union, the element
   * type in the case of an array, and member and element types found
   * recursively are searched. Pointer target types, return types, and
   * parameter types are not searched because they do not require storage when
   * this object is allocated.
   * </p>
   * 
   * <p>
   * This type must be a complete type.
   * </p>
   */
  public final boolean storageHasQualifier(SrcTypeQualifier qual) {
    for (final SrcTypeIterator i = iterator(true); i.hasNext();) {
      try {
        if (i.next().expandSrcTypeQualifiers().contains(qual))
          return true;
      }
      catch (SrcCompletableTypeException e) {
        throw new IllegalStateException(
          "storage has incomplete "+e.getType().getKeyword()+" type");
      }
    }
    return false;
  }

  /**
   * When an object of this type is allocated, is any part of that object's
   * storage a pointer to a type for which {@link #hasEffectiveQualifier}
   * returns true for any of the specified type qualifiers?
   * 
   * <p>
   * This type, member types in the case of a struct or union, the element
   * type in the case of an array, and member and element types found
   * recursively are searched. Return types and parameter types are not
   * searched because they do not require storage when this object is
   * allocated. A pointer's target type is checked by calling
   * {@link #hasEffectiveQualifier} on it.
   * ({@link #storageHasPointerToQualifier} is not called recursively on the
   * pointer's target type even if it is a struct, union, or array).
   * </p>
   * 
   * <p>
   * This type must be a complete type.
   * </p>
   */
  public final boolean storageHasPointerToQualifier(SrcTypeQualifier... quals)
  {
    for (final SrcTypeIterator i = iterator(true); i.hasNext();) {
      try {
        final SrcPointerType p = i.next().toIso(SrcPointerType.class);
        if (p != null && p.getTargetType().hasEffectiveQualifier(quals))
          return true;
      }
      catch (SrcCompletableTypeException e) {
        throw new IllegalStateException(
          "storage has incomplete "+e.getType().getKeyword()+" type");
      }
    }
    return false;
  }

  /**
   * Check constraints when an object of this type is allocated without an
   * explicit initializer.
   * 
   * <p>
   * This method must not be called until the object is allocated without an
   * explicit initializer. That is, for a tentative definition, it must not be
   * called until the end of the translation unit.
   * </p>
   * 
   * <p>
   * The object must have no explicit initializer at all. That is, if the
   * object has a compound initializer, then this method must instead be
   * called for the type of each sub-object that has no explicit initializer.
   * </p>
   * 
   * @param hasImplicitInit
   *          whether the allocation has an implicit initializer (an object
   *          with static storage duration has an implicit zero initializer)
   * @throws SrcRuntimeException
   *           if a constraint is violated
   */
  public final void checkAllocWithoutExplicitInit(boolean hasImplicitInit) {
    if (storageHasPointerToQualifier(SrcTypeQualifier.NVL,
                                     SrcTypeQualifier.NVL_WP))
      throw new SrcRuntimeException(
        "pointer to NVM allocated without explicit initializer");
  }

  /**
   * Check constraints when an object of this type is specified with static or
   * automatic storage duration.
   * 
   * <p>
   * This method must be called for global variables, local variables,
   * compound literals (if one day we support them), function parameter types,
   * and function return types except that it must not be called for a void
   * return type.
   * </p>
   * 
   * <p>
   * This method need not be called for sub-objects (such as the members of a
   * variable of struct type).
   * </p>
   * 
   * @param what
   *          a description of the object
   * @param storageAllocated
   *           whether this object is being checked for a declaration that
   *           allocates storage (for example, not an extern declaration)
   * @throws SrcRuntimeException
   *           if a constraint is violated
   */
  public final void checkStaticOrAutoObject(String what,
                                            boolean storageAllocated)
  {
    if (iso(SrcVoidType))
      throw new SrcRuntimeException(what+" has void type");
    if (storageAllocated && isIncompleteType())
      throw new SrcRuntimeException(what+" has incomplete type");
    if (hasEffectiveQualifier(SrcTypeQualifier.NVL, SrcTypeQualifier.NVL_WP))
      throw new SrcRuntimeException(what+" is NVM-stored");
  }

  /**
   * Check if two types from the same translation unit are compatible. (ISO
   * C99 sec. 6.2.7p1.)
   * 
   * <p>
   * Not all cases of incompatibility specified by ISO C99 are guaranteed to
   * be detected. Specifically, we assume both types are from the same
   * translation unit, so ISO C99 sec. 6.2.7p1's discussion of compatibility
   * between structs or unions from separate translation units is not
   * implemented.
   * </p>
   * 
   * <p>
   * Warnings or errors (depending on the {@code warn} and
   * {@code warningAsErrors} arguments) about type incompatibilities might
   * not be detectable or reported until an incomplete type is completed. In
   * that case, the incomplete type is used as is in the type compatibility
   * check now.
   * </p>
   * 
   * <p>
   * An easy way to check whether another compiler (clang, gcc, etc.)
   * considers two types, T1 and T2, to be compatible is to assign p1 = p2
   * where p1 is a pointer to T1 and p2 is a pointer to T2. That check
   * doesn't work so well in the case of qualified or void target types.
   * Another way is to declare the same function twice with its first
   * argument as type T1 and then as type T2.
   * </p>
   * 
   * @param other
   *          the other type
   * @param warn
   *          whether type incompatibilities should be reported as warnings
   *          instead of errors
   * @param warningsAsErrors
   *          whether to treat warnings as errors. This is irrelevant if
   *          {@code warn} is false. (Type incompatibilities are errors if
   *          either {@code warn} is false or both {@code warn} and
   *          {@code warningsAsErrors} are true, but the message in the
   *          latter case points out that a warning is being treated as an
   *          error.)
   * @param msgPrefix
   *          message prefix for any error or warning reported
   * @return true, or false if the types are determined to be incompatible
   *         and that case is being treated as a warning
   * @throws SrcRuntimeException
   *           if the types are determined to be incompatible and that case
   *           is being treated as an error
   */
  public final boolean checkCompatibility(
    SrcType other, boolean warn, boolean warningsAsErrors, String msgPrefix)
  {
    return null != buildComposite(other, warn, warningsAsErrors, msgPrefix);
  }

  /**
   * Build the composite of two types. (ISO C99 sec. 6.2.7p3.)
   * 
   * <p>
   * Warnings or errors (depending on the {@code warn} and
   * {@code warningAsErrors} arguments) about type incompatibilities might
   * not be detectable or reported until an incomplete type is completed. In
   * that case, the incomplete type is used as is in the composite type
   * computation now.
   * </p>
   * 
   * <p>
   * It's important that implementations observe the following ISO C99 sec.
   * 6.2.7p3 rule: "These rules [for computing a composite type from two
   * types] apply recursively to the types from which the two types are
   * derived." 6.2.5p20 specifies exactly what types are derived from what
   * types. Of particular importance are that a function type is derived
   * from its return type, an array type is derived from its element type,
   * and a pointer type is derived from its target type. 6.2.7p5 gives two
   * examples of computing a composite pointer type where this rule proves
   * important.
   * </p>
   * 
   * <p>
   * Implementations should maintain as much of the typedef structure as
   * possible for the sake of diagnostics. That is, throughout the type,
   * wherever a resulting composite type is eqv to one of the existing
   * types, the existing type should be returned.
   * </p>
   * 
   * <p>
   * This method is used to implement {@link #checkCompatibility}, so it
   * must not produce diagnostics beyond type incompatibility diagnostics.
   * </p>
   * 
   * @param other
   *          the other type
   * @param warn
   *          whether type incompatibilities should be reported as warnings
   *          instead of errors
   * @param warningsAsErrors
   *          whether to treat warnings as errors. This is irrelevant if
   *          {@code warn} is false. (Type incompatibilities are errors if
   *          either {@code warn} is false or both {@code warn} and
   *          {@code warningsAsErrors} are true, but the message in the
   *          latter case points out that a warning is being treated as an
   *          error.)
   * @param msgPrefix
   *          message prefix for any warning or error reported. This prefix
   *          should describe the purpose of the composite type as the
   *          exception message otherwise describes incompatible types but
   *          not a composite type.
   * @return the composite type, or null if the types are incompatible and
   *         that case is being treated as a warning
   * @throws SrcRuntimeException
   *           if the types are determined to be incompatible and that case
   *           is being treated as an error
   */
  public final SrcType buildComposite(
    SrcType other, boolean warn, boolean warningsAsErrors, String msgPrefix)
  {
    // ISO C99 6.7.5.2p6.
    if (this == other)
      return this;
    // ISO C99 6.7.3p9.
    final SrcTopLevelExpandedType
      thisExpanded = expandTopLevelTypedefs(),
      otherExpanded = other.expandTopLevelTypedefs();
    final SrcBaldType baldComposite = thisExpanded.bald().buildCompositeBald(
        otherExpanded.bald(), warn, warningsAsErrors, msgPrefix);
    if (baldComposite == null)
      return null;
    if (!thisExpanded.getSrcTypeQualifiers()
        .equals(otherExpanded.getSrcTypeQualifiers()))
    {
      final StringBuilder diff = new StringBuilder();
      for (SrcTypeQualifier qual : SrcTypeQualifier.values()) {
        if (thisExpanded.getSrcTypeQualifiers().contains(qual)
            != otherExpanded.getSrcTypeQualifiers().contains(qual))
        {
          if (diff.length() > 0)
            diff.append(", ");
          diff.append(qual.name().toLowerCase());
        }
      }
      BuildLLVM.warnOrError(warn, warningsAsErrors,
        msgPrefix+": types are incompatible because only one of them has"
        +" each of the following type qualifiers: "+diff.toString());
      return null;
    }
    // We can use either type qualifier set because they must be the same.
    final SrcType res = SrcQualifiedType.get(
      baldComposite, thisExpanded.getSrcTypeQualifiers());
    return res.eqv(this) ? this : res.eqv(other) ? other : res;
  }

  /**
   * Same as {@link #checkCompatibility(SrcType, String, boolean, boolean)}
   * except {@code warn} is always false.
   */
  public final boolean checkCompatibility(SrcType other, String msgPrefix) {
    return checkCompatibility(other, false, false, msgPrefix);
  }

  /**
   * Same as {@link #buildComposite(SrcType, String, boolean, boolean)}
   * except {@code warn} is always false.
   */
  public final SrcType buildComposite(SrcType other, String msgPrefix) {
    return buildComposite(other, false, false, msgPrefix);
  }

  /**
   * Get the LLVM type with which this source type is represented except when
   * it is used as a pointer target. When used as a pointer target, call
   * {@link #getLLVMTypeAsPointerTarget} instead.
   * 
   * <p>
   * For bit-fields, this returns the type of the bit-field's storage unit, as
   * described in ISO C99 sec. 6.7.2.1p10.
   * </p>
   * 
   * <p>
   * This method should be overridden only by subtypes of {@link SrcBaldType}
   * as the implementation here handles typedefs and type qualifiers.
   * </p>
   */
  public LLVMType getLLVMType(LLVMContext context) {
    final SrcTopLevelExpandedType expandedType = expandTopLevelTypedefs();
    return expandedType.bald().getLLVMType(
      expandedType.getSrcTypeQualifiers(), context);
  }

  /**
   * Get the LLVM type with which this source type is represented when it is
   * used as a pointer target. For other purposes call {@link #getLLVMType}
   * instead.
   * 
   * <p>
   * This method should be overridden only by subtypes of {@link SrcBaldType}
   * as the implementation here handles typedefs and type qualifiers.
   * </p>
   */
  public LLVMType getLLVMTypeAsPointerTarget(LLVMContext context) {
    final SrcTopLevelExpandedType expandedType = expandTopLevelTypedefs();
    return expandedType.bald().getLLVMTypeAsPointerTarget(
      expandedType.getSrcTypeQualifiers(), context);
  }

  /**
   * Same as {@link ValueAndType#prepareForOp} except only compute the new
   * type.
   */
  public final SrcType prepareForOp() {
    final SrcTopLevelExpandedType expandedType = expandTopLevelTypedefs();
    return expandedType.bald()
           .prepareForOp(expandedType.getSrcTypeQualifiers());
  }

  /**
   * This is meant to be called only by {@link ValueAndType#prepareForOp}.
   */
  public final ValueAndType prepareForOp(LLVMValue value,
                                         boolean lvalueOrFnDesignator,
                                         LLVMModule module,
                                         LLVMInstructionBuilder builder)
  {
    final SrcTopLevelExpandedType expandedType = expandTopLevelTypedefs();
    return expandedType.bald()
           .prepareForOp(expandedType.getSrcTypeQualifiers(), value,
                         lvalueOrFnDesignator, module, builder);
  }

  /**
   * Same as {@link ValueAndType#defaultArgumentPromote} except only compute
   * the new type.
   */
  public final SrcType defaultArgumentPromote(
    boolean warn, boolean warningsAsErrors, String msgPrefix)
  {
    final SrcType prep = prepareForOp();
    prep.checkDefaultArgumentPromote(warn, warningsAsErrors, msgPrefix);
    return prep.defaultArgumentPromoteNoPrep();
  }

  /**
   * This method is meant to be called only by
   * {@link SrcType#defaultArgumentPromote}. This method should be overridden
   * only by subtypes of {@link SrcBaldType} as the implementation here
   * handles typedefs and type qualifiers.
   */
  public SrcType defaultArgumentPromoteNoPrep() {
    final SrcTopLevelExpandedType expandedType = expandTopLevelTypedefs();
    return expandedType.bald().defaultArgumentPromoteNoPrep();
  }

  /**
   * This method is meant to be called only by
   * {@link ValueAndType#defaultArgumentPromote}. This method should be
   * overridden only by subtypes of {@link SrcBaldType} as the implementation
   * here handles typedefs and type qualifiers.
   */
  public LLVMValue defaultArgumentPromoteNoPrep(
    LLVMValue value, LLVMModule module, LLVMInstructionBuilder builder)
  {
    final SrcTopLevelExpandedType expandedType = expandTopLevelTypedefs();
    return expandedType.bald()
           .defaultArgumentPromoteNoPrep(value, module, builder);
  }

  /**
   * This method is meant to be called only by
   * {@link ValueAndType#defaultArgumentPromote} and
   * {@link SrcType#defaultArgumentPromote}.
   */
  public final void checkDefaultArgumentPromote(
    boolean warn, boolean warningsAsErrors, String msg)
  {
    new SrcTypeInvolvesNoNVMCheck(this, warn, warningsAsErrors,
      msg+": default argument promotions applied to type that involves NVM"
      +" storage")
    .run();
  }

  /**
   * Same as {@link ValueAndType#load} except the {@link LLVMValue} appears at
   * the start of the parameter list.
   */
  public final ValueAndType load(LLVMValue lvalue, LLVMModule module,
                                 LLVMInstructionBuilder builder)
  {
    final SrcTopLevelExpandedType expandedType = expandTopLevelTypedefs();
    return expandedType.bald().load(expandedType.getSrcTypeQualifiers(),
                                           lvalue, module, builder);
  }

  /**
   * Same as {@link ValueAndType#store} except the {@link LLVMValue} for the
   * lvalue appears at the start of the parameter list.
   */
  public final void store(boolean forInit, LLVMValue lvalue, LLVMValue rvalue,
                          LLVMModule module, LLVMInstructionBuilder builder)
  {
    final SrcTopLevelExpandedType expandedType = expandTopLevelTypedefs();
    expandedType.bald().store(expandedType.getSrcTypeQualifiers(), forInit,
                              lvalue, rvalue, module, builder);
  }

  /**
   * Convert an {@link LLVMValue} to this type if the conversion (whether
   * implicitly or with an explicit cast) is ever valid in C. Callers are
   * responsible for constraining which conversions are permitted for their
   * use cases.
   * 
   * <p>
   * This method should be overridden only by subtypes of {@link SrcBaldType}
   * as the implementation here handles typedefs and type qualifiers.
   * </p>
   * 
   * @param from
   *          the value and type to be converted. It should be the result of a
   *          {@link ValueAndType#prepareForOp} call except when initializing
   *          (not just assigning) an array from a string literal.
   * @param operation
   *          description of what requires this conversion, appropriate in an
   *          error message before the phrase
   *          " requires conversion from X to Y". For example,
   *          "explicit cast".
   * @param module
   *          the LLVM module into which any required declarations should be
   *          generated for the conversion
   * @param builder
   *          the IR builder with which any LLVM instructions should be
   *          generated
   * @return the new value, which might be exactly {@code from.llvmValue} if
   *         no LLVM operations were required to perform the conversion, or
   *         which might be null for a void expression (see
   *         {@link ValueAndType#ValueAndType} documentation).
   * @throws SrcRuntimeException
   *           if the conversion is never valid in C
   */
  public LLVMValue convertFromNoPrep(ValueAndType from, String operation,
                                     LLVMModule module,
                                     LLVMInstructionBuilder builder)
  {
    return expandTopLevelTypedefs().bald().convertFromNoPrep(from, operation,
                                                             module, builder);
  }

  /**
   * Call {@link #convertFromNoPrep} on two operands. {@code operation} should
   * be appropriate in an error message before the phrase
   * " operand N requires conversion from X to Y".
   */
  public final LLVMValue[] binaryConvertFromNoPrep(
    ValueAndType from1, ValueAndType from2, String operation,
    LLVMModule module, LLVMInstructionBuilder builder)
  {
    return new LLVMValue[]{
      convertFromNoPrep(from1, operation + " operand 1", module, builder),
      convertFromNoPrep(from2, operation + " operand 2", module, builder)
    };
  }

  /**
   * This method is meant to be called only by {@link ValueAndType#evalAsCond}.
   * This method should be overridden only by subtypes of {@link SrcBaldType}
   * as the implementation here handles typedefs and type qualifiers.
   */
  public LLVMValue evalAsCondNoPrep(String operand, LLVMValue value,
                                    LLVMModule module,
                                    LLVMInstructionBuilder builder)
  {
    final SrcTopLevelExpandedType expandedType = expandTopLevelTypedefs();
    return expandedType.bald().evalAsCondNoPrep(operand, value, module,
                                                       builder);
  }

  /**
   * This is meant to be called only by {@link ValueAndType#address}.
   */
  public final ValueAndType address(
    LLVMValue value, boolean lvalueOrFnDesignator, LLVMModule module,
    LLVMInstructionBuilder builder)
  {
    if (!lvalueOrFnDesignator)
      throw new SrcRuntimeException(
        "operand to unary \"&\" is not a function designator and not an"
        +" lvalue");
    if (iso(SrcBitFieldType.class))
      throw new SrcRuntimeException("operand to unary \"&\" is a bit-field");
    return new ValueAndType(value, SrcPointerType.get(this), false);
  }

  /**
   * This method is meant to be called only by
   * {@link ValueAndType#indirection}. This method should be overridden only
   * by subtypes of {@link SrcBaldType} as the implementation here handles
   * typedefs and type qualifiers.
   */
  public ValueAndType indirectionNoPrep(String operand, LLVMValue value,
                                        LLVMModule module,
                                        LLVMInstructionBuilder builder)
  {
    final SrcTopLevelExpandedType expandedType = expandTopLevelTypedefs();
    return expandedType.bald().indirectionNoPrep(operand, value, module,
                                                        builder);
  }

  /**
   * This method is meant to be called only by {@link ValueAndType#unaryPlus}.
   * This method should be overridden only by subtypes of {@link SrcBaldType}
   * as the implementation here handles typedefs and type qualifiers.
   */
  public ValueAndType unaryPlusNoPrep(LLVMValue value, LLVMModule module,
                                      LLVMInstructionBuilder builder)
  {
    final SrcTopLevelExpandedType expandedType = expandTopLevelTypedefs();
    return expandedType.bald().unaryPlusNoPrep(value, module, builder);
  }

  /**
   * This method is meant to be called only by {@link ValueAndType#unaryMinus}.
   * This method should be overridden only by subtypes of
   * {@link SrcBaldType} as the implementation here handles typedefs and type
   * qualifiers.
   */
  public ValueAndType unaryMinusNoPrep(LLVMValue value, LLVMModule module,
                                       LLVMInstructionBuilder builder)
  {
    final SrcTopLevelExpandedType expandedType = expandTopLevelTypedefs();
    return expandedType.bald().unaryMinusNoPrep(value, module, builder);
  }

  /**
   * This method is meant to be called only by
   * {@link ValueAndType#unaryBitwiseComplement}. This method should be
   * overridden only by subtypes of {@link SrcBaldType} as the implementation
   * here handles typedefs and type qualifiers.
   */
  public ValueAndType unaryBitwiseComplementNoPrep(
    LLVMValue value, LLVMModule module, LLVMInstructionBuilder builder)
  {
    final SrcTopLevelExpandedType expandedType = expandTopLevelTypedefs();
    return expandedType.bald()
           .unaryBitwiseComplementNoPrep(value, module, builder);
  }

  /**
   * This method is meant to be called only by {@link ValueAndType#unaryNot}.
   * This method should be overridden only by subtypes of {@link SrcBaldType}
   * as the implementation here handles typedefs and type qualifiers.
   */
  public ValueAndType unaryNotNoPrep(LLVMValue value, LLVMModule module,
                                     LLVMInstructionBuilder builder)
  {
    final SrcTopLevelExpandedType expandedType = expandTopLevelTypedefs();
    return expandedType.bald().unaryNotNoPrep(value, module, builder);
  }

  /**
   * This is meant to be called only by {@link ValueAndType#multiply}.
   */
  public static ValueAndType multiplyNoPrep(
    ValueAndType op1, ValueAndType op2, LLVMModule module,
    LLVMInstructionBuilder builder)
  {
    final SrcArithmeticType resultType
      = SrcArithmeticType.getTypeFromUsualArithmeticConversionsNoPrep(
          "binary \"*\"", op1.getSrcType(), op2.getSrcType());
    final LLVMValue[] opsConv
      = resultType.binaryConvertFromNoPrep(op1, op2, "for binary \"*\",",
                                           module, builder);
    return new ValueAndType(resultType.multiplyNoPrep(opsConv[0], opsConv[1],
                                                      module, builder),
                            resultType, false);
  }

  /**
   * This is meant to be called only by {@link ValueAndType#divide}.
   */
  public static ValueAndType divideNoPrep(ValueAndType op1, ValueAndType op2,
                                          LLVMModule module,
                                          LLVMInstructionBuilder builder)
  {
    final SrcArithmeticType resultType
      = SrcArithmeticType.getTypeFromUsualArithmeticConversionsNoPrep(
          "binary \"/\"", op1.getSrcType(), op2.getSrcType());
    final LLVMValue[] opsConv
      = resultType.binaryConvertFromNoPrep(op1, op2, "for binary \"/\",",
                                           module, builder);
    return new ValueAndType(resultType.divideNoPrep(opsConv[0], opsConv[1],
                                                    module, builder),
                            resultType, false);
  }

  /**
   * This is meant to be called only by {@link ValueAndType#remainder}.
   */
  public static ValueAndType remainderNoPrep(
    ValueAndType op1, ValueAndType op2, LLVMModule module,
    LLVMInstructionBuilder builder)
  {
    if (!op1.getSrcType().iso(SrcIntegerType.class))
      throw new SrcRuntimeException(
        "first operand to binary \"%\" is not of integer type");
    if (!op2.getSrcType().iso(SrcIntegerType.class))
      throw new SrcRuntimeException(
        "second operand to binary \"%\" is not of integer type");
    final SrcArithmeticType resultType
      = SrcArithmeticType.getTypeFromUsualArithmeticConversionsNoPrep(
          "binary \"%\"", op1.getSrcType(), op2.getSrcType());
    final LLVMValue[] opsConv
      = resultType.binaryConvertFromNoPrep(op1, op2, "for binary \"%\",",
                                           module, builder);
    return new ValueAndType(resultType.remainderNoPrep(opsConv[0], opsConv[1],
                                                       module, builder),
                            resultType, false);
  }

  /**
   * This is meant to be called only by {@link ValueAndType#add}.
   */
  public static ValueAndType addNoPrep(String operator, ValueAndType op1,
                                       ValueAndType op2, LLVMModule module,
                                       LLVMInstructionBuilder builder)
  {
    // Check the operand types and compute the result type.
    final SrcType op1Type = op1.getSrcType();
    final SrcType op2Type = op2.getSrcType();
    final SrcPointerType op1PtrType = op1Type.toIso(SrcPointerType.class);
    final SrcPointerType op2PtrType = op2Type.toIso(SrcPointerType.class);
    final SrcScalarType resultType;
    if (op1Type.iso(SrcArithmeticType.class)
        && op2Type.iso(SrcArithmeticType.class))
      resultType
        = SrcArithmeticType.getTypeFromUsualArithmeticConversionsNoPrep(
            operator, op1Type, op2Type);
    else if (op1PtrType != null || op2PtrType != null) {
      final SrcPointerType ptrType = op1PtrType != null ? op1PtrType
                                                        : op2PtrType;
      final SrcType otherType = op1PtrType != null ? op2Type : op1Type;
      final String ptrNumber = op1PtrType != null ? "first" : "second";
      final String otherNumber = op1PtrType != null ? "second" : "first";
      if (!ptrType.getTargetType().isObjectType())
        throw new SrcRuntimeException(
          ptrNumber + " operand to " + operator + " is a pointer to a"
          + " non-object type");
      if (!otherType.iso(SrcIntegerType.class))
        throw new SrcRuntimeException(
          ptrNumber + " operand to " + operator + " is of pointer type but "
          + otherNumber + " operand is not of integer type");
      resultType = ptrType;
    }
    else if (!op1Type.iso(SrcScalarType.class))
      throw new SrcRuntimeException("first operand to " + operator
                                    + " is not of scalar type");
    else {
      assert (!op2Type.iso(SrcScalarType.class));
      throw new SrcRuntimeException(
        "second operand to " + operator + " is not of scalar type");
    }

    // Compute the result.
    final LLVMValue[] opsConv;
    if (resultType.iso(SrcArithmeticType.class))
      opsConv = resultType.binaryConvertFromNoPrep(
                  op1, op2, "for " + operator + ",", module, builder);
    else {
      assert (resultType.iso(SrcPointerType.class));
      opsConv = new LLVMValue[] { op1.getLLVMValue(), op2.getLLVMValue() };
    }
    return new ValueAndType(resultType.addNoPrep(opsConv[0], opsConv[1],
                                                 module, builder),
                            resultType, false);
  }

  /**
   * This is meant to be called only by {@link ValueAndType#subtract}.
   */
  public static ValueAndType subtractNoPrep(
    String operator, ValueAndType op1, ValueAndType op2,
    LLVMTargetData targetData, LLVMModule module,
    LLVMInstructionBuilder builder)
  {
    // Check the operand types and compute the result type.
    final SrcType op1Type = op1.getSrcType();
    final SrcType op2Type = op2.getSrcType();
    final SrcPointerType op1PtrType = op1Type.toIso(SrcPointerType.class);
    final SrcPointerType op2PtrType = op2Type.toIso(SrcPointerType.class);
    final SrcScalarType resultType;
    if (op1Type.iso(SrcArithmeticType.class)
        && op2Type.iso(SrcArithmeticType.class))
      resultType
        = SrcArithmeticType.getTypeFromUsualArithmeticConversionsNoPrep(
            operator, op1Type, op2Type);
    else if (op1PtrType != null && op2PtrType != null) {
      op1PtrType.getTargetType().toIso(SrcBaldType.class)
      .checkCompatibility(
         op2PtrType.getTargetType().toIso(SrcBaldType.class),
         "operands of "+operator+" have pointer types with incompatible"
         +" target types");
      if (!op1PtrType.getTargetType().isObjectType())
        throw new SrcRuntimeException(
          "first operand to "+operator+" is a pointer to a non-object type");
      if (!op2PtrType.getTargetType().isObjectType())
        throw new SrcRuntimeException(
          "second operand to "+operator+" is a pointer to a non-object type");
      resultType = SRC_PTRDIFF_T_TYPE;
    }
    else if (op1PtrType != null && op2Type.iso(SrcIntegerType.class)) {
      if (!op1PtrType.getTargetType().isObjectType())
        throw new SrcRuntimeException(
          "first operand to "+operator+" is a pointer to a non-object type");
      resultType = op1PtrType;
    }
    else
      throw new SrcRuntimeException("invalid operands to "+operator);

    // Compute the result.
    final LLVMValue[] opsConv;
    final SrcScalarType dispatchType;
    if (resultType.iso(SrcArithmeticType.class) && op1PtrType == null) {
      dispatchType = resultType;
      opsConv = resultType.binaryConvertFromNoPrep(
                  op1, op2, "for "+operator+",", module, builder);
    }
    else {
      assert (op1PtrType != null);
      dispatchType = op1PtrType;
      opsConv = new LLVMValue[]{op1.getLLVMValue(), op2.getLLVMValue()};
    }
    return new ValueAndType(
      dispatchType.subtractNoPrep(opsConv[0], opsConv[1], targetData, module,
                                  builder),
      resultType, false);
  }

  /**
   * This is meant to be called only by {@link ValueAndType#shift}.
   */
  public static ValueAndType shiftNoPrep(ValueAndType op1, ValueAndType op2,
                                         boolean right, LLVMModule module,
                                         LLVMInstructionBuilder builder)
  {
    // Check the operand types.
    final SrcIntegerType op1IntType
      = op1.getSrcType().toIso(SrcIntegerType.class);
    final SrcIntegerType op2IntType
      = op2.getSrcType().toIso(SrcIntegerType.class);
    if (op1IntType == null)
      throw new SrcRuntimeException(
        "first operand to binary \"<<\" is not of integer type");
    if (op2IntType == null)
      throw new SrcRuntimeException(
        "second operand to binary \"<<\" is not of integer type");
    final ValueAndType op1Promoted
      = op1IntType.integerPromoteNoPrep(op1.getLLVMValue(), module, builder);
    final SrcIntegerType resultType
      = op1Promoted.getSrcType().toEqv(SrcIntegerType.class);
    final ValueAndType op2Promoted
      = op2IntType.integerPromoteNoPrep(op2.getLLVMValue(), module, builder)
        .convertToNoPrep(resultType, "second operand to shift operator",
                         module, builder);
    // ISO C99 sec. 6.5.7p3 says the behavior is undefined if the right operand
    // is negative. LLVM's shift instructions treat the right operand as
    // unsigned, so we go with that.
    final LLVMValue resultValue;
    if (!right)
      resultValue = LLVMShiftInstruction.create(
        builder, ".shl", ShiftType.SHL, op1Promoted.getLLVMValue(),
        op2Promoted.getLLVMValue());
    else if (!resultType.isSigned())
      resultValue = LLVMShiftInstruction.create(
        builder, ".shr", ShiftType.LOGICAL_SHR, op1Promoted.getLLVMValue(),
        op2Promoted.getLLVMValue());
    else
      // ISO C99 sec. 6.5.7p5 says the case of signed type and negative values
      // is implementation-defined. We choose to mimic clang (3.5.1) an gcc
      // (4.2.1).
      resultValue = LLVMShiftInstruction.create(
        builder, ".shr", ShiftType.ARITHMETIC_SHR, op1Promoted.getLLVMValue(),
        op2Promoted.getLLVMValue());
    return new ValueAndType(resultValue, resultType, false);
  }

  /**
   * This is meant to be called only by {@link ValueAndType#relational}.
   */
  public static ValueAndType relationalNoPrep(
    ValueAndType op1, ValueAndType op2, boolean greater, boolean equals,
    LLVMModule module, LLVMInstructionBuilder builder)
  {
    final String oper = (greater ? ">" : "<") + (equals ? "=" : "");
    final String operName = (greater ? "g" : "l") + (equals ? "e" : "t");

    // Check the operand types.
    final SrcType op1Type = op1.getSrcType();
    final SrcType op2Type = op2.getSrcType();
    final SrcPointerType op1PtrType = op1Type.toIso(SrcPointerType.class);
    final SrcPointerType op2PtrType = op2Type.toIso(SrcPointerType.class);
    if (op1PtrType != null && op2PtrType != null) {
      final SrcType op1TargetType = op1PtrType.getTargetType();
      final SrcType op2TargetType = op2PtrType.getTargetType();
      if (op1TargetType.iso(SrcFunctionType.class))
        throw new SrcRuntimeException(
          "first operand to binary \"" + oper
          + "\" is a pointer to function type");
      if (op2TargetType.iso(SrcFunctionType.class))
        throw new SrcRuntimeException(
           "second operand to binary \"" + oper
           + "\" is a pointer to function type");
      op1TargetType.toIso(SrcBaldType.class).checkCompatibility(
        op2TargetType.toIso(SrcBaldType.class),
        "operands of binary \"" + oper + "\" have pointer types with"
        +" incompatible target types");
      if (op1TargetType.isIncompleteType() != op2TargetType.isIncompleteType())
        throw new SrcRuntimeException(
          "operands of binary \"" + oper + "\" have pointer types, but only"
          + " one of the target types is complete");
    }
    else if ((op1PtrType != null) != (op2PtrType != null)) {
      throw new SrcRuntimeException("only one operand to binary \"" + oper
                                    + "\" is of pointer type");
    }
    else if (!op1.getSrcType().iso(SrcArithmeticType.class))
      throw new SrcRuntimeException("first operand to binary \"" + oper
                                    + "\" is not of scalar type");
    else if (!op2.getSrcType().iso(SrcArithmeticType.class))
      throw new SrcRuntimeException("second operand to binary \"" + oper
                                    + "\" is not of scalar type");

    // Compute the result.
    final LLVMValue[] opsConv;
    final SrcScalarType dispatchType;
    if (op1.getSrcType().iso(SrcArithmeticType.class)) {
      assert (op2.getSrcType().iso(SrcArithmeticType.class));
      dispatchType
        = SrcArithmeticType.getTypeFromUsualArithmeticConversionsNoPrep(
            "binary \"" + oper + "\"", op1.getSrcType(), op2.getSrcType());
      opsConv
        = dispatchType.binaryConvertFromNoPrep(
            op1, op2, "for binary \"" + oper + "\",", module, builder);
    }
    else {
      dispatchType = op1PtrType;
      assert (op2PtrType != null);
      opsConv = new LLVMValue[]{op1.getLLVMValue(), op2.getLLVMValue()};
    }
    final LLVMValue resultI1Value
      = dispatchType.relationalI1NoPrep(opsConv[0], opsConv[1], greater,
                                        equals, module, builder);
    final SrcType resultType = SrcIntType;
    final LLVMValue resultValue = LLVMExtendCast.create(
      builder, "." + operName, resultI1Value,
      resultType.getLLVMType(module.getContext()), ExtendType.ZERO);
    return new ValueAndType(resultValue, resultType, false);
  }

  /**
   * This is meant to be called only by {@link ValueAndType#equality}.
   */
  public static ValueAndType equalityNoPrep(
    ValueAndType op1, ValueAndType op2, boolean equals, LLVMModule module,
    LLVMInstructionBuilder builder)
  {
    final String oper = equals ? "==" : "!=";
    final String operName = equals ? "eq" : "ne";

    // Check the operand types.
    final SrcType op1Type = op1.getSrcType();
    final SrcType op2Type = op2.getSrcType();
    final SrcPointerType op1PtrType = op1Type.toIso(SrcPointerType.class);
    final SrcPointerType op2PtrType = op2Type.toIso(SrcPointerType.class);
    if (op1PtrType != null && op2PtrType != null) {
      final SrcType op1TargetType = op1PtrType.getTargetType();
      final SrcType op2TargetType = op2PtrType.getTargetType();
      final boolean op1IsVoidPtr = op1TargetType.iso(SrcVoidType);
      if (op1IsVoidPtr || op2TargetType.iso(SrcVoidType)) {
        // This permits the case where at least one operand is a null pointer
        // constant that includes a void* cast.
        // It also permits the case where one operand is a pointer to an
        // object or incomplete type and the other is a pointer to void.
        final ValueAndType voidPtrOp = op1IsVoidPtr ? op1 : op2;
        final SrcType otherPtrTargetType = op1IsVoidPtr ? op2TargetType
                                                        : op1TargetType;
        if (otherPtrTargetType.iso(SrcFunctionType.class)
            && !voidPtrOp.isNullPointerConstant())
          throw new SrcRuntimeException(
            "operands to binary \"" + oper + "\" are a function pointer and a"
            + " void pointer that is not a null pointer constant");
      }
      else
        op1TargetType.toIso(SrcBaldType.class).checkCompatibility(
          op2TargetType.toIso(SrcBaldType.class),
          "operands of binary \"" + oper + "\" have pointer types with"
          + " incompatible target types, neither of which is void");
    }
    else if ((op1PtrType != null) != (op2PtrType != null)) {
      final ValueAndType nonPtrOp = op1PtrType != null ? op2 : op1;
      if (!nonPtrOp.getSrcType().iso(SrcIntegerType.class))
        throw new SrcRuntimeException(
          "only one operand to binary \"" + oper + "\" is of pointer type");
      else if (!nonPtrOp.isNullPointerConstant())
        throw new SrcRuntimeException(
          "operands to binary \"" + oper + "\" are a pointer and an integer"
          + " expression that is not a null pointer constant");
      // This permits the case where one operand is a pointer and the other
      // is a null pointer constant that does not include a void* cast,
    }
    else if (!op1Type.iso(SrcArithmeticType.class))
      throw new SrcRuntimeException("first operand to binary \"" + oper
                                    + "\" is not of scalar type");
    else if (!op2Type.iso(SrcArithmeticType.class))
      throw new SrcRuntimeException("second operand to binary \"" + oper
                                    + "\" is not of scalar type");
    final SrcIntegerType resultType = SrcIntType;

    // Convert the operands and perform the operation.
    final LLVMValue[] opsConv;
    final SrcScalarType dispatchType;
    if (op1Type.iso(SrcArithmeticType.class)
        && op2Type.iso(SrcArithmeticType.class))
    {
      dispatchType
        = SrcArithmeticType.getTypeFromUsualArithmeticConversionsNoPrep(
            "binary \"" + oper + "\"", op1.getSrcType(), op2.getSrcType());
      opsConv
        = dispatchType.binaryConvertFromNoPrep(
            op1, op2, "for binary \"" + oper + "\",", module, builder);
    }
    else if (op1PtrType != null && op2.isNullPointerConstant()
             || op2PtrType != null && op1.isNullPointerConstant())
    {
      // Convert the null pointer constant to the other pointer type. If
      // they're both null pointer constants, then convert the one that has
      // integer type (if either) to the one that has pointer type.
      final SrcPointerType ptrType = op1PtrType != null ? op1PtrType
                                                        : op2PtrType;
      final ValueAndType opPtr = op1PtrType != null ? op1 : op2;
      ValueAndType opNull = op1PtrType != null ? op2 : op1;
      opNull = opNull.convertToNoPrep(
        opPtr.getSrcType(),
        "null pointer constant operand to binary \"" + oper + "\" operator",
        module, builder);
      dispatchType = ptrType;
      opsConv = new LLVMValue[]{opPtr.getLLVMValue(), opNull.getLLVMValue()};
    }
    else {
      assert(op1PtrType != null);
      assert(op2PtrType != null);
      final SrcType op1TargetType = op1PtrType.getTargetType();
      // There might not actually be a void pointer, so we then just
      // arbitrarily choose to convert op2 to op1's type.
      final boolean op1IsVoidPtr = op1TargetType.iso(SrcVoidType);
      final ValueAndType opDominate = op1IsVoidPtr ? op2 : op1;
      dispatchType = op1IsVoidPtr ? op2PtrType : op1PtrType;
      ValueAndType opNonDominate = op1IsVoidPtr ? op1 : op2;
      opNonDominate = opNonDominate.convertToNoPrep(
        opDominate.getSrcType(),
        "pointer operand to binary \"" + oper + "\" operator",
        module, builder);
      opsConv = new LLVMValue[]{opDominate.getLLVMValue(),
                                opNonDominate.getLLVMValue()};
    }
    final LLVMValue resultI1Value
      = dispatchType.equalityI1NoPrep(opsConv[0], opsConv[1], equals, module,
                                      builder);
    final LLVMValue resultValue
      = LLVMExtendCast.create(builder, "." + operName, resultI1Value,
                              resultType.getLLVMType(module.getContext()),
                              ExtendType.ZERO);
    return new ValueAndType(resultValue, resultType, false);
  }

  public enum AndXorOr {
    AND("&", "bitwiseAnd"), XOR("^", "bitwiseXor"), OR("|", "bitwiseOr");
    private AndXorOr(String oper, String operName) {
      this.oper = oper;
      this.operName = operName;
    }
    final String oper;
    final String operName;
  }

  /**
   * This is meant to be called only by {@link ValueAndType#andXorOr}.
   */
  public static ValueAndType andXorOrNoPrep(
    ValueAndType op1, ValueAndType op2, AndXorOr kind, LLVMModule module,
    LLVMInstructionBuilder builder)
  {
    final String oper = "binary \"" + kind.oper + "\"";
    if (!op1.getSrcType().iso(SrcIntegerType.class))
      throw new SrcRuntimeException("first operand to " + oper
                                    + " is not of integer type");
    if (!op2.getSrcType().iso(SrcIntegerType.class))
      throw new SrcRuntimeException("second operand to " + oper
                                    + " is not of integer type");
    final SrcIntegerType resultType
      = SrcArithmeticType.getTypeFromUsualArithmeticConversionsNoPrep(
          oper, op1.getSrcType(), op2.getSrcType())
        .toEqv(SrcIntegerType.class);
    assert(resultType != null);
    final LLVMValue[] opsConv = resultType.binaryConvertFromNoPrep(
      op1, op2, "for binary \"" + kind.oper + "\",", module, builder);
    final LLVMValue resultValue;
    switch (kind) {
    case AND:
      resultValue = LLVMAndInstruction.create(builder, "." + kind.operName,
                                              opsConv[0], opsConv[1]);
      break;
    case XOR:
      resultValue = LLVMXorInstruction.create(builder, "." + kind.operName,
                                              opsConv[0], opsConv[1]);
      break;
    case OR:
      resultValue = LLVMOrInstruction.create(builder, "." + kind.operName,
                                              opsConv[0], opsConv[1]);
      break;
    default:
      throw new IllegalStateException();
    }
    return new ValueAndType(resultValue, resultType, false);
  }

  /**
   * This is meant to be called only by {@link ValueAndType#arraySubscript}.
   */
  public static ValueAndType arraySubscriptNoPrep(
    ValueAndType op1, ValueAndType op2, LLVMModule module,
    LLVMInstructionBuilder builder)
  {
    final String operator = "subscript operator";
    final boolean op1IsPtr = op1.getSrcType().iso(SrcPointerType.class);
    if (!op1IsPtr && !op2.getSrcType().iso(SrcPointerType.class))
      throw new SrcRuntimeException(
        "neither operand to " + operator + " is of pointer type");
    final SrcType otherType = op1IsPtr ? op2.getSrcType() : op1.getSrcType();
    if (!otherType.iso(SrcIntegerType.class))
      throw new SrcRuntimeException(
        "neither operand to " + operator + " is of integer type");
    // The above validates operands further than the calls below.
    return addNoPrep(operator, op1, op2, module, builder)
           .indirection("subscripted expression", module, builder);
  }

  /**
   * This is meant to be called only by {@link ValueAndType#simpleAssign}.
   */
  public static ValueAndType simpleAssign(
    String operator, boolean binary,
    ValueAndType op1, ValueAndType op2, LLVMModule module,
    LLVMInstructionBuilder builder, boolean warningsAsErrors)
  {
    final String op1Name = (binary ? "first " : "") + "operand";
    // ISO C99 sec. 6.3.2.1p1 defines modifiable lvalue, and 6.5.16p2 says
    // that's what's required here.
    if (!op1.isModifiableLvalue())
      throw new SrcRuntimeException(op1Name+" of "+operator+" is not a"
                                    +" modifiable lvalue");
    final SrcType op1Type = op1.getSrcType();
    final LLVMValue resultValue
      = op2.convertForAssign(op1.getSrcType(), AssignKind.ASSIGNMENT_OPERATOR,
                             module, builder, warningsAsErrors);
    op1Type.store(false, op1.getLLVMValue(), resultValue, module, builder);
    return new ValueAndType(resultValue, op1Type, false);
  }

  /**
   * Same as {@link #toString(String, Set)} but with an abstract nested
   * declarator and an initially empty skip set.
   */
  @Override
  public final String toString() {
    return toString(
      "", Collections.newSetFromMap(new IdentityHashMap<SrcType, Boolean>()));
  }

  /**
   * Build a string representation of this type in C syntax. This is meant for
   * debugging not for generating legal C source. Some LLVM constant
   * expressions might be enclosed in angle brackets.
   * 
   * @param nestedDecl
   *          a nested declarator, such as an identifier or function type,
   *          that has/returns this type; or an empty string to indicate an
   *          abstract declarator
   * @param skipTypes
   *          a set of types to skip as necessary to avoid infinite recursion.
   *          The contents of this set are modified during the generation of
   *          the string.
   * @return the string
   */
  public abstract String toString(String nestedDecl, Set<SrcType> skipTypes);

  /**
   * Same as {@link #toCompatibilityString(Map, Integer, LLVMContext)} but
   * with an initially empty struct set.
   */
  public final String toCompatibilityString(LLVMContext ctxt) {
    return toCompatibilityString(new HashSet<SrcStructOrUnionType>(), ctxt);
  }

  /**
   * Compute a string representation of this type for checking both API and
   * ABI compatibility with another type.
   * 
   * <p>
   * The strings returned by this method (or, more practically, checksums of
   * those strings because the strings can be long) can be compared at run
   * time to check whether types are compatible at both the API and ABI
   * level. C-level struct and enum tags are encoded in the returned string
   * because multiple structs or enums with identical bodies but different
   * semantics and thus different tags could appear in a single C program,
   * so retrieving data as one such type even though it was written as
   * another such type could permit manipulating the data with code that was
   * not designed for that data. Moreover, the following are included in the
   * returned string: type qualifiers, both sizes and signedness of integer
   * types, a struct's exact sequence of field names and field type
   * compatibility strings, an enum's set of member names and values, and
   * the compatibiliy string for an enum's compatible integer type. Each of
   * these impacts how data is accessed either at the API or ABI level.
   * TODO: More information might ought to be included, such as alignment
   * and struct padding in case it changes across targets or compiler
   * versions.
   * </p>
   * 
   * @param structSet
   *          a set of all struct types already visited. This set enables us
   *          to handle recursive struct types without entering an infinite
   *          recursion. That is, we expand the struct definition into the
   *          result string only the first time we see the struct
   *          referenced. The contents of this set are modified during the
   *          generation of the result string.
   * @param ctxt
   *          the LLVM context for building LLVM types with which this
   *          source type is represented
   * @return the type compatibility string
   * @throws IllegalStateException
   *           if this type (or some type it references, perhaps indirectly)
   *           is an incomplete struct type. In that case, the type
   *           compatibility string would not fully represent the type.
   * @throws IllegalStateException
   *           if this method is called on a type, such as a function or
   *           union, that is invalid for NVM storage. This is not an
   *           inherent limitation of this method, but we're currently only
   *           using it for NVL-C, and we didn't bother to implement it for
   *           unnecessary types.
   */
  public abstract String toCompatibilityString(
    Set<SrcStructOrUnionType> structSet, LLVMContext ctxt);
}
