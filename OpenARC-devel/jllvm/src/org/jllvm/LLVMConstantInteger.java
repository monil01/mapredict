package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;
import org.jllvm.LLVMIntegerType;
import java.math.BigInteger;

public class LLVMConstantInteger extends LLVMConstant {
	public static LLVMConstantInteger get(LLVMIntegerType intType,long N,boolean signExtend) {
		if(intType.getWidth() == 1)
			return LLVMConstantBoolean.get(N != 0, intType.getContext());
		// This BigInteger is destined for the unsigned long long parameter of the
		// LLVM C API's LLVMConstInt, so we must sign-extend any negative N
		// properly for unsigned long long, especially given that BigInteger
		// stores its value in the least number of bits possible. (We assume
		// unsigned long long is 64 bits, which is usually true, and the LLVM C++
		// API uses uint64_t for this parameter anyway.) Doing so mimics C type
		// conversion rules when passing a negative value to the unsigned long
		// long parameter of the LLVM C API's LLVMConstInt. All of this happens
		// before signExtend is applied to convert the resulting value to intType.
		BigInteger val = BigInteger.valueOf(N);
		if(N < 0)
			val = BigInteger.ONE.shiftLeft(64).add(val);
		return (LLVMConstantInteger)LLVMValue.getValue(Core.LLVMConstInt(intType.getInstance(),val,signExtend ? 1 : 0));
	}
	
	public LLVMConstantInteger(SWIGTYPE_p_LLVMOpaqueValue c) {
		super(c);
		assert(typeOf() instanceof LLVMIntegerType);
	}
	
	public static LLVMConstantInteger allOnes(LLVMIntegerType type) {
		return (LLVMConstantInteger)LLVMValue.getValue(Core.LLVMConstAllOnes(type.getInstance()));
	}

	public LLVMConstant truncate(LLVMIntegerType targetType) {
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstTrunc(instance,targetType.getInstance()));
	}
	
	public LLVMConstant signExtend(LLVMIntegerType targetType) {
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstSExt(instance,targetType.getInstance()));
	}
	
	public LLVMConstant zeroExtend(LLVMIntegerType targetType) {
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstZExt(instance,targetType.getInstance()));
	}
	
	public LLVMConstant toFloatingPoint(LLVMRealType targetType,boolean signed) {
		if(signed)
			return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstSIToFP(instance,targetType.getInstance()));
		else
			return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstUIToFP(instance,targetType.getInstance()));
	}
	
	public LLVMConstant toPointer(LLVMPointerType targetType) {
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstIntToPtr(instance,targetType.getInstance()));
	}
	
	public java.math.BigInteger getZExtValue() {
		return Core.LLVMConstIntGetZExtValue(instance);
	}
	
	public long getSExtValue() {
		return Core.LLVMConstIntGetSExtValue(instance);
	}
	
	public String toString() {
		return typeOf() + "<" + getZExtValue().toString() + ">";
	}
}
