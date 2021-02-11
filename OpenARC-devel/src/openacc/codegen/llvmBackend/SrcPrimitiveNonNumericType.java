package openacc.codegen.llvmBackend;

import java.util.Set;

import openacc.codegen.llvmBackend.SrcQualifiedType.SrcTypeQualifier;

import org.jllvm.LLVMContext;
import org.jllvm.LLVMInstructionBuilder;
import org.jllvm.LLVMIntegerType;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMType;
import org.jllvm.LLVMValue;
import org.jllvm.LLVMVoidType;

/**
 * The LLVM backend's class for all primitive non-numeric types from the C
 * source.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class SrcPrimitiveNonNumericType extends SrcBaldType {
  /**
   * LLVM IR does not permit pointers to void. LLVM documentation recommends
   * pointers to {@link LLVMIntegerType#i8} instead.
   */
  public static final SrcPrimitiveNonNumericType SrcVoidType
    = new SrcPrimitiveNonNumericType(0, 8, "void");

  private final int llvmIntWidth; // 0 means void
  private final int llvmIntWidthAsPointerTarget; // 0 means void
  private final String cSyntax;

  private SrcPrimitiveNonNumericType(int llvmIntWidth,
                                     int llvmIntWidthAsPointerTarget,
                                     String cSyntax)
  {
    this.llvmIntWidth = llvmIntWidth;
    this.llvmIntWidthAsPointerTarget = llvmIntWidthAsPointerTarget;
    this.cSyntax = cSyntax;
  }

  @Override
  public boolean eqvBald(SrcBaldType other) {
    return this == other;
  }

  @Override
  public boolean isIncompleteType() {
    return true;
  }

  @Override
  public boolean hasEffectiveQualifier(SrcTypeQualifier... quals) {
    return false;
  }

  @Override
  public SrcPrimitiveNonNumericType withoutEffectiveQualifier(
    SrcTypeQualifier... quals)
  {
    return this;
  }

  @Override
  public SrcTypeIterator componentIterator(boolean storageOnly,
                                           Set<SrcType> skipTypes)
  {
    if (storageOnly)
      throw new IllegalStateException("storage-only iteration of void type");
    return new SrcTypeComponentIterator(storageOnly, skipTypes);
  }

  @Override
  public SrcPrimitiveNonNumericType buildCompositeBald(
    SrcBaldType other, boolean warn, boolean warningsAsErrors,
    String msgPrefix)
  {
    if (eqvBald(other))
      return this;
    BuildLLVM.warnOrError(warn, warningsAsErrors,
      msgPrefix+": void type is incompatible with non-void type: "+other);
    return null;
  }

  @Override
  public LLVMType getLLVMType(LLVMContext context) {
    return llvmIntWidth == 0 ? LLVMVoidType.get(context)
                             : LLVMIntegerType.get(context, llvmIntWidth);
  }
  @Override
  public LLVMType getLLVMTypeAsPointerTarget(LLVMContext context) {
    return llvmIntWidthAsPointerTarget == 0
           ? LLVMVoidType.get(context)
           : LLVMIntegerType.get(context, llvmIntWidthAsPointerTarget);
  }

  @Override
  public SrcType defaultArgumentPromoteNoPrep() {
    return this;
  }
  @Override
  public LLVMValue defaultArgumentPromoteNoPrep(LLVMValue value,
                                                LLVMModule module,
                                                LLVMInstructionBuilder builder)
  {
    return value;
  }

  @Override
  public LLVMValue convertFromNoPrep(ValueAndType from, String operation,
                                     LLVMModule module,
                                     LLVMInstructionBuilder builder)
  {
    // The preceding call to prepareForOp might have generated a (useless but
    // potentially segfault-producing) load if called on an lvalue. At least
    // clang (clang-600.0.56) also does so for an explicit cast to void even
    // though its later compiler passes apparently optimize the load away. See
    // ExpressionStatement visitor in BuildLLVM as well.
    return null;
  }

  @Override
  public String toString(String nestedDecl, Set<SrcType> skipTypes) {
    StringBuilder res = new StringBuilder(cSyntax);
    if (!nestedDecl.isEmpty()) {
      res.append(" ");
      res.append(nestedDecl);
    }
    return res.toString();
  }

  @Override
  public String toCompatibilityString(Set<SrcStructOrUnionType> structSet,
                                      LLVMContext ctxt)
  {
    return cSyntax;
  }
}
