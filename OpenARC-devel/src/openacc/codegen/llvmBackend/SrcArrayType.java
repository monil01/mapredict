package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SRC_WCHAR_TYPE;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcCharType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcSignedCharType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedCharType;

import java.lang.ref.WeakReference;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Set;

import openacc.codegen.llvmBackend.SrcQualifiedType.SrcTypeQualifier;

import org.jllvm.LLVMArrayType;
import org.jllvm.LLVMConstant;
import org.jllvm.LLVMConstantArray;
import org.jllvm.LLVMConstantInteger;
import org.jllvm.LLVMContext;
import org.jllvm.LLVMGetElementPointerInstruction;
import org.jllvm.LLVMGlobalVariable;
import org.jllvm.LLVMInstructionBuilder;
import org.jllvm.LLVMIntegerType;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMValue;
import org.jllvm.bindings.LLVMLinkage;

/**
 * The LLVM backend's class for all array types from the C source.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class SrcArrayType extends SrcBaldType {
  private static final HashMap<SrcArrayType, WeakReference<SrcArrayType>>
    cache = new HashMap<>();

  private final SrcType elementType;
  // -1 if unspecified in source, becomes 0 if flexible array member
  private final long numElements;

  private SrcArrayType(SrcType elementType, long numElements) {
    assert(elementType != null); // otherwise, hashCode needs updating
    if (elementType.isIncompleteType())
      throw new SrcRuntimeException(
        "array type has incomplete element type");
    if (elementType.iso(SrcFunctionType.class))
      throw new SrcRuntimeException(
        "array type has function element type");
    assert(numElements >= -1); // so that unspecified num is cached properly
    this.elementType = elementType;
    this.numElements = numElements;
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + elementType.hashCode();
    result = prime * result + (int)(numElements^(numElements>>>32));
    return result;
  }
  @Override
  public boolean equals(Object obj) {
    if (this == obj) return true;
    if (obj == null) return false;
    if (getClass() != obj.getClass()) return false;
    SrcArrayType other = (SrcArrayType)obj;
    if (elementType != other.elementType) return false;
    if (numElements != other.numElements) return false;
    return true;
  }

  /**
   * Get the specified array type.
   * 
   * <p>
   * After this method has been called once for a particular type of array, it
   * is guaranteed to return the same object for the same type of array until
   * that object can be garbage-collected because it is no longer referenced
   * outside this class's internal cache.
   * </p>
   * 
   * @param elementType
   *          the array type's element type
   * @param numElements
   *          the number of elements in the array type, or 0 if flexible array
   *          member. Call {@link #get(SrcType)} if number of elements is left
   *          unspecified.
   * @return the specified array type
   * @throws SrcRuntimeException
   *           if {@code elementType} is an incomplete type or function type
   *           (ISO C99 sec. 6.2.5p20 with footnote 36, and 6.7.5.2p1)
   */
  public static SrcArrayType get(SrcType elementType, long numElements) {
    final SrcArrayType cacheKey = new SrcArrayType(elementType, numElements);
    WeakReference<SrcArrayType> ref;
    synchronized (cache) {
      ref = cache.get(cacheKey);
    }
    SrcArrayType type;
    if (ref == null || (type = ref.get()) == null) {
      type = new SrcArrayType(elementType, numElements);
      ref = new WeakReference<>(type);
      synchronized (cache) {
        cache.put(cacheKey, ref);
      }
    }
    return type;
  }
  /**
   * Same as {@link #get(SrcType, long)} except the number of elements was
   * left unspecified in the source.
   */
  public static SrcArrayType get(SrcType elementType) {
    return get(elementType, -1);
  }
  @Override
  protected void finalize() {
    synchronized (cache) {
      if (cache.get(this).get() == null)
        cache.remove(this);
    }
  }

  /** Get the array type's element type. */
  public SrcType getElementType() {
    return elementType;
  }
  /** Is the number of elements specified? */
  public boolean numElementsIsSpecified() {
    return numElements > -1;
  }
  /**
   * Get number of elements in the array type. Fails an assertion if
   * {@link #numElementsIsSpecified} returns false.
   */
  public long getNumElements() {
    assert(numElementsIsSpecified());
    return numElements;
  }

  @Override
  public boolean eqvBald(SrcBaldType other) {
    // Optimize for trivial eqv relation.
    if (this == other)
      return true;
    if (!(other instanceof SrcArrayType))
      return false;
    final SrcArrayType otherArrayType = (SrcArrayType)other;
    if (numElements != otherArrayType.numElements)
      return false;
    return elementType.eqv(otherArrayType.elementType);
  }

  @Override
  public boolean isIncompleteType() {
    return !numElementsIsSpecified();
  }

  @Override
  public boolean hasEffectiveQualifier(SrcTypeQualifier... quals) {
    return elementType.hasEffectiveQualifier(quals);
  }

  @Override
  public SrcArrayType withoutEffectiveQualifier(SrcTypeQualifier... quals)
  {
    final SrcType eleTy = elementType.withoutEffectiveQualifier(quals);
    if (numElementsIsSpecified())
      return SrcArrayType.get(eleTy, numElements);
    return SrcArrayType.get(eleTy);
  }

  @Override
  public SrcTypeIterator componentIterator(boolean storageOnly,
                                           Set<SrcType> skipTypes)
  {
    return new SrcTypeComponentIterator(storageOnly, skipTypes,
                                        new SrcType[]{elementType});
  }

  @Override
  public SrcArrayType buildCompositeBald(
    SrcBaldType other, boolean warn, boolean warningsAsErrors,
    String msgPrefix)
  {
    // ISO C99 6.7.5.2p6.
    if (this == other)
      return this;
    final SrcArrayType otherArray = other.toEqv(SrcArrayType.class);
    if (otherArray == null) {
      BuildLLVM.warnOrError(warn, warningsAsErrors,
        msgPrefix+": array type is incompatible with non-array type: "
        +other);
      return null;
    }
    final SrcType compositeElementType = elementType.buildComposite(
      otherArray.elementType, warn, warningsAsErrors,
      msgPrefix+": array types are incompatible because their element"
      +" types are incompatible");
    if (compositeElementType == null)
      return null;
    if (numElementsIsSpecified() && otherArray.numElementsIsSpecified()
        && getNumElements() != otherArray.getNumElements())
    {
      BuildLLVM.warnOrError(warn, warningsAsErrors,
        msgPrefix+": array types are incompatible because they have"
        +" different sizes");
      return null;
    }
    final SrcArrayType res
      = SrcArrayType.get(compositeElementType,
                         numElementsIsSpecified() ? getNumElements()
                                                  : otherArray.numElements);
    return res.eqv(this) ? this : res.eqv(otherArray) ? otherArray : res;
  }

  @Override
  public LLVMArrayType getLLVMType(LLVMContext context) {
    // It's possible to have a pointer to an incomplete array type (clang 3.5.1
    // and gcc 4.2.1 support it), so we use size 0 in that case.
    return LLVMArrayType.get(elementType.getLLVMType(context),
                             numElementsIsSpecified() ? getNumElements() : 0);
  }
  @Override
  public LLVMArrayType getLLVMTypeAsPointerTarget(LLVMContext context) {
    return getLLVMType(context);
  }

  @Override
  public SrcType prepareForOp(EnumSet<SrcTypeQualifier> srcTypeQualifiers) {
    // Type qualifiers on array type should have been folded into element type.
    assert(srcTypeQualifiers.isEmpty());
    return SrcPointerType.get(getElementType());
  }
  @Override
  public ValueAndType prepareForOp(
    EnumSet<SrcTypeQualifier> srcTypeQualifiers, LLVMValue value,
    boolean lvalueOrFnDesignator, LLVMModule module,
    LLVMInstructionBuilder builder)
  {
    // Type qualifiers on array type should have been folded into element type.
    assert(srcTypeQualifiers.isEmpty());
    final LLVMContext context = module.getContext();
    // ISO C99 sec. 6.3.2.1p3.
    if (lvalueOrFnDesignator) {
      final LLVMConstant zero
        = LLVMConstant.constNull(LLVMIntegerType.get(context, 32));
      final LLVMValue resValue
        = LLVMGetElementPointerInstruction.create(builder, ".arr2ptr", value,
                                                  zero, zero);
      return new ValueAndType(resValue, prepareForOp(), false);
    }
    // According to preconditions on the ValueAndType constructor, an rvalue
    // array is always a string literal unless it's a constant-expression
    // compound initializer, which should never be processed here.
    final LLVMConstantArray str = (LLVMConstantArray)value;
    final LLVMGlobalVariable var = module.addGlobal(getLLVMType(context),
                                                    ".str");
    // These three properties mimic clang (clang-600.0.56):
    // private unnamed_addr constant
    var.setLinkage(LLVMLinkage.LLVMPrivateLinkage);
    var.setConstant(true);
    // TODO: When we have an LLVM whose C API supports it, extend jllvm
    // accordingly, uncomment this, and add check for it in the test
    // suite. It's available by at least LLVM 3.5.0 as
    // LLVMSetUnnamedAddr.
    // See: http://lists.cs.uiuc.edu/pipermail/llvm-commits/Week-of-Mon-20140310/208071.html
    // See similar todo for __FUNCTION__ in BuildLLVM.
    //var.setUnnamedAddr(true);
    var.setInitializer(str);
    return prepareForOp(var, true, module, builder);
  }

  @Override
  public SrcType defaultArgumentPromoteNoPrep() {
    // prepareForOp has been called and it never returns an array.
    throw new IllegalStateException();
  }
  @Override
  public LLVMValue defaultArgumentPromoteNoPrep(LLVMValue value,
                                                LLVMModule module,
                                                LLVMInstructionBuilder builder)
  {
    // prepareForOp has been called and it never returns an array.
    throw new IllegalStateException();
  }

  @Override
  public LLVMValue convertFromNoPrep(ValueAndType from, String operation,
                                     LLVMModule module,
                                     LLVMInstructionBuilder builder)
  {
    final LLVMContext context = module.getContext();
    // The only way an array can be assigned to an array in ISO C is when a
    // string literal initializes an array as described in ISO C99 sec.
    // 6.3.2.1p3 and 6.7.8p14-15. If this weren't an initialization, then
    // ValueAndType#prepareforOp should have been called and converted any
    // string literal to a pointer.
    if (from.isStringLiteral()) {
      final SrcArrayType fromArrayType
        = from.getSrcType().toIso(SrcArrayType.class);
      if (elementType.iso(SRC_WCHAR_TYPE)) {
        if (fromArrayType.elementType.eqv(SrcCharType))
          throw new SrcRuntimeException(
            "wide character array initialized from non-wide string literal");
        assert(fromArrayType.elementType.eqv(SRC_WCHAR_TYPE));
      }
      else if (elementType.iso(SrcCharType)
               || elementType.iso(SrcUnsignedCharType)
               || elementType.iso(SrcSignedCharType))
      {
        if (fromArrayType.elementType.eqv(SRC_WCHAR_TYPE))
          throw new SrcRuntimeException(
            "non-wide character array initialized from wide string literal");
        assert(fromArrayType.elementType.eqv(SrcCharType));
      }
      else
        throw new SrcRuntimeException(
          "non-character array initialized from string literal");
      final LLVMConstantArray str = (LLVMConstantArray)from.getLLVMValue();
      final long numElements = numElementsIsSpecified()
                               ? getNumElements()
                               : fromArrayType.getNumElements();
      final LLVMConstant[] elements = new LLVMConstant[(int)numElements];
      for (int i = 0; i < numElements; ++i) {
        if (i < fromArrayType.getNumElements())
          elements[i]
            = (LLVMConstantInteger)str.extractValue(new long[]{i});
        else
          elements[i]
            = LLVMConstant.constNull(elementType.getLLVMType(context));
      }
      return LLVMConstantArray.get(elementType.getLLVMType(context),
                                   elements);
    }
    throw new SrcRuntimeException(operation + " requires conversion to array"
                                  + " type");
  }

  @Override
  public String toString(String nestedDecl, Set<SrcType> skipTypes) {
    StringBuilder str = new StringBuilder(nestedDecl);
    str.append("[");
    if (numElementsIsSpecified())
      str.append(getNumElements());
    str.append("]");
    return elementType.toString(str.toString(), skipTypes);
  }

  @Override
  public String toCompatibilityString(Set<SrcStructOrUnionType> structSet,
                                      LLVMContext ctxt)
  {
    StringBuilder str = new StringBuilder(
      elementType.toCompatibilityString(structSet, ctxt));
    str.append("[");
    if (numElementsIsSpecified())
      str.append(getNumElements());
    str.append("]");
    return str.toString();
  }
}
