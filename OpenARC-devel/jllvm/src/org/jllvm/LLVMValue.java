package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.LLVMOpcode;
import org.jllvm.bindings.LLVMTypeKind;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueType;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;

/**
 * Implements all methods from Core.h dealing with the base class Value.
 * 
 * <p>
 * All jllvm objects of type {@link LLVMValue} are stored in a cache (in the
 * associated {@link LLVMContext} object), keyed on the underlying LLVM
 * objects they wrap. Moreover, for each such object, the {@link LLVMValue}
 * subtype consistently corresponds to LLVM object's runtime type. This
 * approach has many advantages. It reduces the number of {@link LLVMValue}
 * objects in memory. It makes comparing {@link LLVMValue} objects by
 * reference reliably equivalent to comparing their LLVM objects by reference.
 * It enables reliably using {@code instanceof} with an {@link LLVMValue}
 * object to determine the runtime type of its LLVM object. It enables
 * reliably casting an {@link LLVMValue} object reference to its runtime type
 * instead of re-wrapping its LLVM object in a new {@link LLVMValue} object of
 * the runtime type.
 * </p>
 * 
 * <p>
 * In order to achieve these advantages, {@link LLVMValue} objects must be
 * handled consistently as follows. Wherever an object is retrieved from LLVM,
 * unless it can be guaranteed that object has not yet been seen in jllvm,
 * {@link #getValue} must be called to search the {@link LLVMValue} cache for
 * an existing {@link LLVMValue} object before creating a new wrapper. If
 * {@link #getValue} doesn't find an existing {@link LLVMValue} object, it
 * wraps the LLVM object in an {@link LLVMalue} object of the runtime type. If
 * the {@link LLVMValue} type hierarchy is extended, {@link #getValue} must be
 * extended to detect the new types at runtime. Every {@link LLVMValue}
 * subtype must provide a constructor for wrapping an existing LLVM object,
 * but this constructor should only be called by {@link #getValue} and by the
 * type's other constructors (except when wrapping a null -- see below).
 * {@link LLVMValue} subtypes like {@link LLVMGlobalVariable}, for which new
 * objects are always requested via some other LLVM type's method, such as the
 * LLVM method called by {@link LLVMModule#addGlobal}, need not provide any
 * additional constructor or factory method. {@link LLVMValue} subtypes like
 * {@link LLVMConstantInteger}, for which new objects can be requested
 * directly via that subtype but whose LLVM objects are uniqued in LLVM, must
 * provide a factory method, normally named {@code get}, to perform cache
 * lookup rather than blindly constructing a new object. {@link LLVMValue}
 * subtypes like {@link LLVMFunction}, for which new objects can be requested
 * directly via that subtype but whose LLVM objects are not uniqued in LLVM,
 * instead provide a constructor to make it clearer that a new object is
 * constructed every time.
 * </p>
 * 
 * <p>
 * For example, constant integers are uniqued in LLVM, so
 * {@link LLVMConstantInteger} provides the factory method
 * {@link LLVMConstantInteger#get} rather than a constructor. Also,
 * {@link LLVMGlobalVariable#getInitializer} calls {@link #getValue} to
 * retrieve or construct the {@link LLVMValue} object for its return value
 * rather than blindly wrapping it in a new {@link LLVMConstant}, which is its
 * return type. As a result, if {@link LLVMGlobalVariable#getInitializer} is
 * about to return a constant integer, the return is wrapped in a
 * {@link LLVMConstantInteger} object, which is guaranteed to be the same
 * object previously used for that constant integer, if any.
 * </p>
 * 
 * <p>
 * Special attention must given to the case where LLVM might return null
 * instead of an object. Generally, jllvm wraps any null from LLVM in a jllvm
 * object, which is never cached because there are many types that might need
 * to wrap null. For example, when there are no more basic blocks, instead of
 * returning null, {@link LLVMBasicBlock#getNextBasicBlock} returns a
 * {@link LLVMBasicBlock} wrapping a null. The apparent advantage of this
 * approach is that, when jllvm must pass a value of type {@link LLVMValue} to
 * SWIG-generated code, it can simply call {@link #getInstance} on the value
 * and pass its results without checking whether the value itself was null.
 * The disadvantage is that, when the user wants to check a value for null,
 * such as the return of {@link LLVMBasicBlock#getNextBasicBlock}, it is
 * unintuitive that he must check the result of {@link #getInstance} rather
 * than the value itself, and it is unintuitive that instanceof can return
 * true for a null.
 * </p>
 * 
 * <p>
 * {@link LLVMBasicBlock#getNextBasicBlock} does not handle null directly.
 * Instead, it passes the result of LLVM's {@code LLVMGetNextBasicBlock} to
 * {@link LLVMBasicBlock#getBasicBlock}. That, in turn, passes the result to
 * {@link #getValue} if the result is not null, or it wraps the result if the
 * result is null. All {@link LLVMValue} subtypes for which some LLVM method
 * can return a null result should provide a method like
 * {@link LLVMBasicBlock#getBasicBlock}.
 * </p>
 * 
 * <p>
 * TODO: Any place where the above rules are disobeyed is a bug. In
 * particular, there might exist methods that do not properly handle a null
 * returned by LLVM, and so they will produce an assertion failure in
 * {@link #getValue}.
 * </p>
 * 
 * <p>
 * For any LLVM object, LLVM eventually frees it, and then LLVM might
 * eventually reuse it's address for another LLVM object, but then the cache
 * might have a stale jllvm object for that address. Thus, {@link #getValue}
 * reuses a cached jllvm object if and only if it has the appropriate type and
 * fields for the LLVM object at the specified address. To make this check
 * easy, {@link LLVMValue} subtypes usually do not have fields other than the
 * LLVM object address. If they do, {@link #getValue} must retrieve those
 * fields from the new LLVM object and update the retrieved jllvm object
 * accordingly. Thus, the jllvm object must not have fields that cannot be be
 * retrieved from the LLVM object, and fields that can be retrieved from the
 * LLVM object are redundant anyway.
 * </p>
 */
public abstract class LLVMValue {
	/**
	 * Same as {@link SWIGTYPE_p_LLVMOpaqueValue} except instances compare equal
	 * when and only when the underlying LLVM object addresses are equal.
	 * {@link #hashCode} is adjusted accordingly.
	 */
	static class Hashable_LLVMOpaqueValue extends SWIGTYPE_p_LLVMOpaqueValue {
		/**
		 * Construct a new {@link Hashable_LLVMOpaqueValue}.
		 * 
		 * @param o
		 *          contains the non-null address to store, or is null
		 * @return a new {@link Hashable_LLVMOpaqueValue} containing a non-null
		 *         address, or null if {@code o} is null
		 */
		public static Hashable_LLVMOpaqueValue make(SWIGTYPE_p_LLVMOpaqueValue o) {
			if (o == null)
				return null;
			assert(getCPtr(o) != 0);
			return new Hashable_LLVMOpaqueValue(o);
		}

		/** Always use {@link make} instead.  */
		private Hashable_LLVMOpaqueValue(SWIGTYPE_p_LLVMOpaqueValue o) {
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
			if (!(obj instanceof Hashable_LLVMOpaqueValue)) {
				return false;
			}
			return getCPtr(this) == getCPtr((Hashable_LLVMOpaqueValue) obj);
		}
		
		/**
		 * When calling {@link org.jllvm.bindings.CoreJNI} methods, it's sometimes
		 * necessary to retrieve the underlying LLVM object address.
		 * 
		 * @return the underlying LLVM object address
		 */
		public long getCPtr() {
			return getCPtr(this);
		}
	}
	
	/**
	 * Contains the address of the LLVM object that this wraps.  When the address
	 * is null, {@link #instance} is null instead of containing null.
	 */
	protected Hashable_LLVMOpaqueValue instance;
	
	public LLVMValue(SWIGTYPE_p_LLVMOpaqueValue val) {
		instance = Hashable_LLVMOpaqueValue.make(val);
		// In the case of null, we cannot use typeOf, and we don't want to cache
		// the value anyway, as documented in the header comments.
		if (val != null)
			typeOf().getContext().getValueCache().put(instance,this);
	}
	
	public LLVMType typeOf() {
		return LLVMType.getType(Core.LLVMTypeOf(instance));
	}
	
	public String getValueName() {
		return Core.LLVMGetValueName(instance);
	}
	
	public void setValueName(String name) {
		Core.LLVMSetValueName(instance,name);
	}
	
	public void replaceAllUsesWith(LLVMValue newVal) {
		Core.LLVMReplaceAllUsesWith(instance,newVal.instance);
	}
	
	public LLVMUse getFirstUse() {
		return new LLVMUse(Core.LLVMGetFirstUse(instance));
	}
	
	public void dump() {
		Core.LLVMDumpValue(instance);
	}
	
	public SWIGTYPE_p_LLVMOpaqueValue getInstance() {
		return instance;
	}
	
	public boolean equals(Object obj) {
		if(obj instanceof LLVMValue)
			return ((LLVMValue)obj).instance == instance;
		else
			return false;
	}
	
	/**
	 * Get a cached {@link LLVMValue} object.
	 * 
	 * @param val
	 *          contains the address of the LLVM object for which a cached
	 *          {@link LLVMValue} object should be sought; must not be null
	 * @return the cached {@link LLVMValue} object, or a new {@link LLVMValue}
	 *         object for {@code val} if none has previously been cached for
	 *         {@code val}
	 */
	public static LLVMValue getValue(SWIGTYPE_p_LLVMOpaqueValue val) {
		assert(val != null);
		final LLVMContext ctxt = LLVMType.getType(Core.LLVMTypeOf(val)).getContext();
		final LLVMValue cached = ctxt.getValueCache().get(Hashable_LLVMOpaqueValue.make(val));
		// The LLVMIsA* functions are listed in LLVM 3.2's
		// include/llvm-c/Core.h's LLVM_FOR_EACH_VALUE_SUBCLASS macro definition.
		final LLVMValue result;
		if(Core.LLVMIsAConstant(val) != null) {
			// Sometimes LLVMIsAConstantArray or LLVMIsAConstantStruct returns null
			// for a constant whose type is indeed an array or struct, so we check
			// the type directly instead. For example, the constant initializer built
			// for "int a[1] = {0};" or "struct {int i;} s = {0};" has this problem.
			// In the case of the struct, the zero value is significant because 1,
			// for example, does not have the problem.
			if(Core.LLVMGetTypeKind(Core.LLVMTypeOf(val)) == LLVMTypeKind.LLVMArrayTypeKind)
				result = cached instanceof LLVMConstantArray ? cached : new LLVMConstantArray(val);
			else if(Core.LLVMGetTypeKind(Core.LLVMTypeOf(val)) == LLVMTypeKind.LLVMStructTypeKind)
				result = cached instanceof LLVMConstantStruct ? cached : new LLVMConstantStruct(val);
			else if(Core.LLVMIsAConstantExpr(val) != null)
				result = cached instanceof LLVMConstantExpression ? cached : new LLVMConstantExpression(val);
			else if(Core.LLVMIsAConstantInt(val) != null) {
				final SWIGTYPE_p_LLVMOpaqueType type = Core.LLVMTypeOf(val);
				if(Core.LLVMGetTypeKind(type) == LLVMTypeKind.LLVMIntegerTypeKind && Core.LLVMGetIntTypeWidth(type) == 1)
					result = cached instanceof LLVMConstantBoolean ? cached : new LLVMConstantBoolean(val);
				else
					result = cached instanceof LLVMConstantInteger ? cached : new LLVMConstantInteger(val);
			}
			else if(Core.LLVMIsAConstantPointerNull(val) != null)
				result = cached instanceof LLVMConstantPointer ? cached : new LLVMConstantPointer(val);
			else if(Core.LLVMIsAConstantFP(val) != null)
				result = cached instanceof LLVMConstantReal ? cached : new LLVMConstantReal(val);
			else if(Core.LLVMIsAConstantVector(val) != null)
				result = cached instanceof LLVMConstantVector ? cached : new LLVMConstantVector(val);
			else if(Core.LLVMIsAGlobalValue(val) != null) {
				if(Core.LLVMIsAFunction(val) != null)
					result = cached instanceof LLVMFunction ? cached : new LLVMFunction(val);
				else if(Core.LLVMIsAGlobalAlias(val) != null)
					result = cached instanceof LLVMGlobalAlias ? cached : new LLVMGlobalAlias(val);
				else if(Core.LLVMIsAGlobalVariable(val) != null)
					result = cached instanceof LLVMGlobalVariable ? cached : new LLVMGlobalVariable(val);
				else
					throw new IllegalStateException();
			}
			else if (Core.LLVMIsAUndefValue(val) != null)
			  result = cached instanceof LLVMUndefinedValue ? cached : new LLVMUndefinedValue(val);
			else
				throw new IllegalStateException();
		}
		else if(Core.LLVMIsAInstruction(val) != null) {
			// An LLVMHeapAllocation, LLVMFreeInstruction, or
			// LLVMUnaryBitwiseInstruction is never retrieved from the cache. See
			// the documentation for their constructors for details.
			LLVMOpcode opcode = Core.LLVMGetInstructionOpcode(val);
			if (opcode == LLVMOpcode.LLVMAlloca)
				result = cached instanceof LLVMStackAllocation ? cached : new LLVMStackAllocation(val);
			else if (opcode == LLVMOpcode.LLVMAdd)
				result = cached instanceof LLVMAddInstruction ? cached : new LLVMAddInstruction(val); 
			else if (opcode == LLVMOpcode.LLVMFAdd)
				result = cached instanceof LLVMAddInstruction ? cached : new LLVMAddInstruction(val);
			else if (opcode == LLVMOpcode.LLVMUDiv)
				result = cached instanceof LLVMDivideInstruction ? cached : new LLVMDivideInstruction(val);
			else if (opcode == LLVMOpcode.LLVMSDiv)
				result = cached instanceof LLVMDivideInstruction ? cached : new LLVMDivideInstruction(val);
			else if (opcode == LLVMOpcode.LLVMFDiv)
				result = cached instanceof LLVMDivideInstruction ? cached : new LLVMDivideInstruction(val);
			else if (opcode == LLVMOpcode.LLVMMul)
				result = cached instanceof LLVMMultiplyInstruction ? cached : new LLVMMultiplyInstruction(val);
			else if (opcode == LLVMOpcode.LLVMFMul)
				result = cached instanceof LLVMMultiplyInstruction ? cached : new LLVMMultiplyInstruction(val);
			else if (opcode == LLVMOpcode.LLVMURem)
				result = cached instanceof LLVMRemainderInstruction ? cached : new LLVMRemainderInstruction(val);
			else if (opcode == LLVMOpcode.LLVMSRem)
				result = cached instanceof LLVMRemainderInstruction ? cached : new LLVMRemainderInstruction(val);
			else if (opcode == LLVMOpcode.LLVMFRem)
				result = cached instanceof LLVMRemainderInstruction ? cached : new LLVMRemainderInstruction(val);
			else if (opcode == LLVMOpcode.LLVMSub)
				result = cached instanceof LLVMSubtractInstruction ? cached : new LLVMSubtractInstruction(val);
			else if (opcode == LLVMOpcode.LLVMFSub)
				result = cached instanceof LLVMSubtractInstruction ? cached : new LLVMSubtractInstruction(val);
			else if (opcode == LLVMOpcode.LLVMAnd)
				result = cached instanceof LLVMAndInstruction ? cached : new LLVMAndInstruction(val);
			else if (opcode == LLVMOpcode.LLVMOr)
				result = cached instanceof LLVMOrInstruction ? cached : new LLVMOrInstruction(val);
			else if (opcode == LLVMOpcode.LLVMXor)
				result = cached instanceof LLVMXorInstruction ? cached : new LLVMXorInstruction(val);
			else if (opcode == LLVMOpcode.LLVMCall)
				result = cached instanceof LLVMCallInstruction ? cached : new LLVMCallInstruction(val);
			else if (opcode == LLVMOpcode.LLVMBitCast)
				result = cached instanceof LLVMBitCast ? cached : new LLVMBitCast(val);
			else if (opcode == LLVMOpcode.LLVMZExt)
				result = cached instanceof LLVMExtendCast
				         && ((LLVMExtendCast)cached).getInstructionType() == LLVMExtendCast.ExtendType.ZERO
				         ? cached : new LLVMExtendCast(val, LLVMExtendCast.ExtendType.ZERO);
			else if (opcode == LLVMOpcode.LLVMSExt)
				result = cached instanceof LLVMExtendCast
				         && ((LLVMExtendCast)cached).getInstructionType() == LLVMExtendCast.ExtendType.SIGN
				         ? cached : new LLVMExtendCast(val, LLVMExtendCast.ExtendType.SIGN);
			else if (opcode == LLVMOpcode.LLVMFPExt)
				result = cached instanceof LLVMExtendCast
				         && ((LLVMExtendCast)cached).getInstructionType() == LLVMExtendCast.ExtendType.FLOAT
				         ? cached : new LLVMExtendCast(val, LLVMExtendCast.ExtendType.FLOAT);
			else if (opcode == LLVMOpcode.LLVMFPToUI)
				result = cached instanceof LLVMFloatToIntegerCast
				         && ((LLVMFloatToIntegerCast)cached).getCastType() == LLVMFloatToIntegerCast.FPToIntCastType.UNSIGNED
				         ? cached : new LLVMFloatToIntegerCast(val, LLVMFloatToIntegerCast.FPToIntCastType.UNSIGNED);
			else if (opcode == LLVMOpcode.LLVMFPToSI)
				result = cached instanceof LLVMFloatToIntegerCast
				         && ((LLVMFloatToIntegerCast)cached).getCastType() == LLVMFloatToIntegerCast.FPToIntCastType.SIGNED
				         ? cached : new LLVMFloatToIntegerCast(val, LLVMFloatToIntegerCast.FPToIntCastType.SIGNED);
			else if (opcode == LLVMOpcode.LLVMUIToFP)
				result = cached instanceof LLVMIntegerToFloatCast
				         && ((LLVMIntegerToFloatCast)cached).getCastType() == LLVMIntegerToFloatCast.IntCastType.UNSIGNED
				         ? cached : new LLVMIntegerToFloatCast(val, LLVMIntegerToFloatCast.IntCastType.UNSIGNED);
			else if (opcode == LLVMOpcode.LLVMSIToFP)
				result = cached instanceof LLVMIntegerToFloatCast
				         && ((LLVMIntegerToFloatCast)cached).getCastType() == LLVMIntegerToFloatCast.IntCastType.SIGNED
				         ? cached : new LLVMIntegerToFloatCast(val, LLVMIntegerToFloatCast.IntCastType.SIGNED);
			else if (opcode == LLVMOpcode.LLVMPtrToInt)
				result = cached instanceof LLVMPtrIntCast
				         && ((LLVMPtrIntCast)cached).getCastType() == LLVMPtrIntCast.PtrIntCastType.PTR_TO_INT
				         ? cached : new LLVMPtrIntCast(val, LLVMPtrIntCast.PtrIntCastType.PTR_TO_INT);
			else if (opcode == LLVMOpcode.LLVMIntToPtr)
				result = cached instanceof LLVMPtrIntCast
				         && ((LLVMPtrIntCast)cached).getCastType() == LLVMPtrIntCast.PtrIntCastType.INT_TO_PTR
				         ? cached : new LLVMPtrIntCast(val, LLVMPtrIntCast.PtrIntCastType.INT_TO_PTR);
			else if (opcode == LLVMOpcode.LLVMTrunc)
				result = cached instanceof LLVMTruncateCast
				         && ((LLVMTruncateCast)cached).getInstructionType() == LLVMTruncateCast.TruncateType.INTEGER
				         ? cached : new LLVMTruncateCast(val, LLVMTruncateCast.TruncateType.INTEGER);
			else if (opcode == LLVMOpcode.LLVMFPTrunc)
				result = cached instanceof LLVMTruncateCast
				         && ((LLVMTruncateCast)cached).getInstructionType() == LLVMTruncateCast.TruncateType.FLOAT
				         ? cached : new LLVMTruncateCast(val, LLVMTruncateCast.TruncateType.FLOAT);
			else if (opcode == LLVMOpcode.LLVMFCmp)
				result = cached instanceof LLVMFloatComparison ? cached : new LLVMFloatComparison(val);
			else if (opcode == LLVMOpcode.LLVMICmp)
				result = cached instanceof LLVMIntegerComparison ? cached : new LLVMIntegerComparison(val);
			else if (opcode == LLVMOpcode.LLVMExtractElement)
				result = cached instanceof LLVMExtractElementInstruction ? cached : new LLVMExtractElementInstruction(val);
			else if (opcode == LLVMOpcode.LLVMExtractValue)
				result = cached instanceof LLVMExtractValueInstruction ? cached : new LLVMExtractValueInstruction(val);
			else if (opcode == LLVMOpcode.LLVMGetElementPtr)
				result = cached instanceof LLVMGetElementPointerInstruction ? cached : new LLVMGetElementPointerInstruction(val);
			else if (opcode == LLVMOpcode.LLVMInsertElement)
				result = cached instanceof LLVMInsertElementInstruction ? cached : new LLVMInsertElementInstruction(val);
			else if (opcode == LLVMOpcode.LLVMInsertValue)
				result = cached instanceof LLVMInsertValueInstruction ? cached : new LLVMInsertValueInstruction(val);
			else if (opcode == LLVMOpcode.LLVMLoad)
				result = cached instanceof LLVMLoadInstruction ? cached : new LLVMLoadInstruction(val);
			else if (opcode == LLVMOpcode.LLVMPHI)
				result = cached instanceof LLVMPhiNode ? cached : new LLVMPhiNode(val);
			else if (opcode == LLVMOpcode.LLVMSelect)
				result = cached instanceof LLVMSelectInstruction ? cached : new LLVMSelectInstruction(val);
			else if (opcode == LLVMOpcode.LLVMShl)
				result = cached instanceof LLVMShiftInstruction
				         && ((LLVMShiftInstruction)cached).getShiftType() == LLVMShiftInstruction.ShiftType.SHL
				         ? cached : new LLVMShiftInstruction(val, LLVMShiftInstruction.ShiftType.SHL);
			else if (opcode == LLVMOpcode.LLVMLShr)
				result = cached instanceof LLVMShiftInstruction
				         && ((LLVMShiftInstruction)cached).getShiftType() == LLVMShiftInstruction.ShiftType.LOGICAL_SHR
				         ? cached : new LLVMShiftInstruction(val, LLVMShiftInstruction.ShiftType.LOGICAL_SHR);
			else if (opcode == LLVMOpcode.LLVMAShr)
				result = cached instanceof LLVMShiftInstruction
				         && ((LLVMShiftInstruction)cached).getShiftType() == LLVMShiftInstruction.ShiftType.ARITHMETIC_SHR
				         ? cached : new LLVMShiftInstruction(val, LLVMShiftInstruction.ShiftType.ARITHMETIC_SHR);
			else if (opcode == LLVMOpcode.LLVMShuffleVector)
				result = cached instanceof LLVMShuffleVectorInstruction ? cached : new LLVMShuffleVectorInstruction(val);
			else if (opcode == LLVMOpcode.LLVMStore)
				result = cached instanceof LLVMStoreInstruction ? cached : new LLVMStoreInstruction(val);
			else if (opcode == LLVMOpcode.LLVMBr)
				result = cached instanceof LLVMBranchInstruction ? cached : new LLVMBranchInstruction(val);
			else if (opcode == LLVMOpcode.LLVMInvoke)
				result = cached instanceof LLVMInvokeInstruction ? cached : new LLVMInvokeInstruction(val);
			else if (opcode == LLVMOpcode.LLVMRet)
				result = cached instanceof LLVMReturnInstruction ? cached : new LLVMReturnInstruction(val);
			else if (opcode == LLVMOpcode.LLVMSwitch)
				result = cached instanceof LLVMSwitchInstruction ? cached : new LLVMSwitchInstruction(val);
			else if (opcode == LLVMOpcode.LLVMUnreachable)
				result = cached instanceof LLVMUnreachableInstruction ? cached : new LLVMUnreachableInstruction(val);
			else if (opcode == LLVMOpcode.LLVMVAArg)
				result = cached instanceof LLVMVariableArgumentInstruction ? cached : new LLVMVariableArgumentInstruction(val);
			else
				throw new IllegalStateException();
		}
		else if (Core.LLVMIsAArgument(val) != null)
			result = cached instanceof LLVMArgument ? cached : new LLVMArgument(val);
		else if (Core.LLVMIsABasicBlock(val) != null)
			result = cached instanceof LLVMBasicBlock ? cached : new LLVMBasicBlock(val);
		else if (Core.LLVMIsAInlineAsm(val) != null)
			result = cached instanceof LLVMConstantInlineASM ? cached : new LLVMConstantInlineASM(val);
		else if (Core.LLVMIsAMDNode(val) != null)
			result = cached instanceof LLVMMDNode ? cached : new LLVMMDNode(val);
		else if (Core.LLVMIsAMDString(val) != null)
			result = cached instanceof LLVMMDString ? cached : new LLVMMDString(val);
		else
			throw new IllegalStateException();
		assert(result != null);
		return result;
	}
}
