package openacc.codegen.llvmBackend;

import java.util.EnumSet;
import java.util.Set;

import org.jllvm.LLVMContext;

import openacc.codegen.llvmBackend.SrcQualifiedType.SrcTypeQualifier;

/**
 * The LLVM backend's class for all typedefs from the C source.
 * 
 * <p>
 * TODO: This class is currently unused. In the future, it will be used to
 * record the structure of typedefs for the sake of improved diagnostics in
 * the LLVM backend. For now, all typedefs are fully expanded in the LLVM
 * backend's type system, but we are trying to use methods like {@link #eqv}
 * and {@link #isa} everywhere to help us prepare for using typedefs. This
 * class is currently just a placeholder to make the planned type hierarchy
 * easier to discuss. See {@link SrcType}'s header comments for further
 * details.
 * </p>
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public final class SrcTypedefType extends SrcUnqualifiedType {
  private String name;
  private SrcType targetType;

  @Override
  protected SrcTopLevelExpandedType expandTopLevelTypedefs(
    EnumSet<SrcTypeQualifier> addQuals)
  {
    return targetType.expandTopLevelTypedefs(addQuals);
  }

  @Override
  public String toString(String nestedDecl, Set<SrcType> skipTypes) {
    StringBuilder str = new StringBuilder(name);
    if (!nestedDecl.isEmpty()) {
      str.append(" ");
      str.append(nestedDecl);
    }
    return str.toString();
  }

  @Override
  public String toCompatibilityString(Set<SrcStructOrUnionType> structSet,
                                      LLVMContext ctxt)
  {
    return targetType.toCompatibilityString(structSet, ctxt);
  }
}
