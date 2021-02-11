package openacc.codegen.llvmBackend;

import java.util.EnumSet;

import openacc.codegen.llvmBackend.SrcQualifiedType.SrcTypeQualifier;

/**
 * The LLVM backend's class for all types from the C source that have no
 * top-level type qualifiers before typedef expansion. That is,
 * {@link #getSrcTypeQualifiers} always returns an empty set.
 * 
 * <p>
 * These types might have type qualifiers on target types, on element types,
 * on member types, or at the top level after typedef expansion (if one day we
 * actually represent typedef names in this hierarchy). Typedef expansion is
 * important to keep in mind because a {@link SrcUnqualifiedType} can become a
 * {@link SrcQualifiedType} when typedefs are expanded, so don't assume it's
 * safe to use {@link #eqv} instead of {@link #iso} just because you are
 * working with {@link SrcUnqualifiedType} objects. See
 * {@link SrcBaldType} for a safer alternative.
 * </p>
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public abstract class SrcUnqualifiedType extends SrcType {
  @Override
  public final SrcUnqualifiedType unqualifiedUnexpanded() {
    return this;
  }
  @Override
  public final EnumSet<SrcTypeQualifier> getSrcTypeQualifiers() {
    return EnumSet.noneOf(SrcTypeQualifier.class);
  }
}
