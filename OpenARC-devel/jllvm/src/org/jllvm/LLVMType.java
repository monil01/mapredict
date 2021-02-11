package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.LLVMTypeKind;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueType;

/**
 * Implements all methods from Core.h dealing with the base class Type.
 * 
 * <p>
 * All jllvm objects of type {@link LLVMType} are stored in a cache (in the
 * associated {@link LLVMContext} object). Moreover, for each such object, the
 * {@link LLVMType} subtype consistently corresponds to LLVM object's runtime
 * type. This approach is the same as the approach for the {@link LLVMValue}
 * hierarchy. See the {@link LLVMValue} documentation for a full explanation.
 * {@link #getType} here corresponds to {@link LLVMValue#getValue} there for
 * cache lookup. {@link LLVMIdentifiedStructType} is an {@link LLVMType} whose
 * objects are not uniqued and that thus has a constructor instead of a
 * factory method.
 * </p>
 */
public abstract class LLVMType {
	/**
	 * Same as {@link SWIGTYPE_p_LLVMOpaqueType} except instances compare equal
	 * when and only when the underlying LLVM object addresses are equal.
	 * {@link #hashCode} is adjusted accordingly.
	 */
	public static class Hashable_LLVMOpaqueType
		extends SWIGTYPE_p_LLVMOpaqueType
	{
		/**
		 * Construct a new {@link Hashable_LLVMOpaqueType}.
		 * 
		 * @param o
		 *          contains the non-null address to store, or is null
		 * @return a new {@link Hashable_LLVMOpaqueType} containing a non-null
		 *         address, or null if {@code o} is null
		 */
		public static Hashable_LLVMOpaqueType make(SWIGTYPE_p_LLVMOpaqueType o) {
			if (o == null)
				return null;
			assert(getCPtr(o) != 0);
		  return new Hashable_LLVMOpaqueType(o);
		}

		/** Always use {@link make} instead.  */
		private Hashable_LLVMOpaqueType(SWIGTYPE_p_LLVMOpaqueType o) {
			super(getCPtr(o), false);
		}

		@Override
		public int hashCode() {
			long cPtr = getCPtr(this);
			// This is the hashCode documented for Long.hashCode at
			// http://docs.oracle.com/javase/7/docs/api/java/lang/Long.html.
			// We reproduce the formula here instead of calling Long.hashCode so
			// we don't have to waste time constructing a Long object.
			return (int) (cPtr ^ (cPtr >>> 32));
		}
	
		@Override
		public boolean equals(java.lang.Object obj) {
			if (!(obj instanceof Hashable_LLVMOpaqueType)) {
				return false;
			}
			return getCPtr(this) == getCPtr((Hashable_LLVMOpaqueType) obj);
		}
	}
	
	/**
	 * Contains the address of the LLVM object that this wraps.  When the address
	 * is null, {@link #instance} is null instead of containing null.
	 */
	protected final Hashable_LLVMOpaqueType instance;
	
	public LLVMTypeKind getTypeKind() {
		return Core.LLVMGetTypeKind(instance);
	}
	
	public boolean isSized() {
		return Core.LLVMTypeIsSized(instance) != 0;
	}
	
	/**
	 * Modeled after llvm::Type::getPrimitiveSizeInBits. See LLVM documentation
	 * for that function for more info. Returns zero if the type does not have a
	 * size or is not a primitive type.
	 */
	public long getPrimitiveSizeInBits() { return 0; }
	
	public LLVMContext getContext() {
		return LLVMContext.getContext(Core.LLVMGetTypeContext(instance));
	}
	
	public SWIGTYPE_p_LLVMOpaqueType getInstance() {
		return instance;
	}
	
	public boolean equals(Object obj) {
		if(obj instanceof LLVMType)
			return ((LLVMType)obj).instance == instance;
		else
			return false;
	}
	
	public LLVMType(SWIGTYPE_p_LLVMOpaqueType tr) {
		instance = Hashable_LLVMOpaqueType.make(tr);
		// In the case of null, we cannot use getContext, and we don't want to
		// cache the value anyway, as documented in LLVMValue's header comments,
		// referenced in the header comments here.
		if (tr != null)
			getContext().getTypeCache().put(instance,this);
	}
	
	/**
	 * Get a cached {@link LLVMType} object.
	 * 
	 * @param tr
	 *          contains the address of the LLVM object for which a cached
	 *          {@link LLVMType} object should be sought; must not be null
	 * @return the cached {@link LLVMType} object, or a new {@link LLVMType}
	 *         object for {@code tr} if none has previously been cached for
	 *         {@code tr}
	 */
	public static LLVMType getType(SWIGTYPE_p_LLVMOpaqueType tr) {
		assert(tr != null);
		final LLVMContext ctxt = LLVMContext.getContext(Core.LLVMGetTypeContext(tr));
		final LLVMType cached = ctxt.getTypeCache().get(Hashable_LLVMOpaqueType.make(tr));
		final LLVMTypeKind kind = Core.LLVMGetTypeKind(tr);
		LLVMType result = null;
		if(kind == LLVMTypeKind.LLVMVoidTypeKind)
			result = cached instanceof LLVMVoidType ? cached : new LLVMVoidType(tr);
		else if(kind == LLVMTypeKind.LLVMFloatTypeKind)
			result = cached instanceof LLVMFloatType ? cached : new LLVMFloatType(tr);
		else if(kind == LLVMTypeKind.LLVMDoubleTypeKind)
			result = cached instanceof LLVMDoubleType ? cached : new LLVMDoubleType(tr);
		else if(kind == LLVMTypeKind.LLVMX86_FP80TypeKind)
			result = cached instanceof LLVMX86FP80Type ? cached : new LLVMX86FP80Type(tr);
		else if(kind == LLVMTypeKind.LLVMFP128TypeKind)
			result = cached instanceof LLVMFP128Type ? cached : new LLVMFP128Type(tr);
		else if(kind == LLVMTypeKind.LLVMPPC_FP128TypeKind)
			result = cached instanceof LLVMPPCFP128Type ? cached : new LLVMPPCFP128Type(tr);
		else if(kind == LLVMTypeKind.LLVMLabelTypeKind)
			result = cached instanceof LLVMLabelType ? cached : new LLVMLabelType(tr);
		else if(kind == LLVMTypeKind.LLVMIntegerTypeKind)
			result = cached instanceof LLVMIntegerType ? cached : new LLVMIntegerType(tr);
		else if(kind == LLVMTypeKind.LLVMFunctionTypeKind)
			result = cached instanceof LLVMFunctionType ? cached : new LLVMFunctionType(tr);
		else if(kind == LLVMTypeKind.LLVMStructTypeKind) {
			if (null == Core.LLVMGetStructName(tr))
				result = cached instanceof LLVMStructType ? cached : new LLVMStructType(tr);
			else
				result = cached instanceof LLVMIdentifiedStructType ? cached : new LLVMIdentifiedStructType(tr);
		}
		else if(kind == LLVMTypeKind.LLVMArrayTypeKind)
			result = cached instanceof LLVMArrayType ? cached : new LLVMArrayType(tr);
		else if(kind == LLVMTypeKind.LLVMPointerTypeKind)
			result = cached instanceof LLVMPointerType ? cached : new LLVMPointerType(tr);
		else if(kind == LLVMTypeKind.LLVMVectorTypeKind)
			result = cached instanceof LLVMVectorType ? cached : new LLVMVectorType(tr);
		else if(kind == LLVMTypeKind.LLVMMetadataTypeKind)
			result = cached instanceof LLVMMetadataType ? cached : new LLVMMetadataType(tr);
		assert(result != null);
		return result;
	}
	
	/** Convert to normal LLVM IR assembly representation. */
	public abstract String toString();
	
	/**
	 * Convert to textual representation used to indicate parameter and return
	 * types in names of overloaded LLVM intrinsic functions.
	 * 
	 * @throws IllegalStateException
	 *           if no such representation is known (for void type, for example).
	 */
	public abstract String toStringForIntrinsic();
}
