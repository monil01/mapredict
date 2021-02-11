package org.jllvm;

import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueValue;
import org.jllvm.bindings.LLVMIntPredicate;
import org.jllvm.bindings.LLVMRealPredicate;
import org.jllvm.bindings.SWIGTYPE_p_unsigned_int;
import org.jllvm.bindings.SWIGTYPE_p_p_LLVMOpaqueValue;
import org.jllvm.LLVMConstant;

public class LLVMConstantExpression extends LLVMConstant {
	public enum DivType {SIGNEDINT,UNSIGNEDINT,FLOAT};
	public static LLVMConstant getSizeOf(LLVMType type) {
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMSizeOf(type.getInstance()));
	}
	
	public static LLVMConstant neg(LLVMConstant op) {
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstNeg(op.instance));
	}
	
	public static LLVMConstant fneg(LLVMConstant op) {
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstFNeg(op.instance));
	}
	
	public static LLVMConstant add(LLVMConstant lhs,LLVMConstant rhs) {
		assert(lhs.typeOf().equals(rhs.typeOf()));
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstAdd(lhs.instance,rhs.instance));
	}
	
	public static LLVMConstant fadd(LLVMConstant lhs,LLVMConstant rhs) {
		assert(lhs.typeOf().equals(rhs.typeOf()));
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstFAdd(lhs.instance,rhs.instance));
	}
	
	public static LLVMConstant subtract(LLVMConstant lhs,LLVMConstant rhs) {
		assert(lhs.typeOf().equals(rhs.typeOf()));
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstSub(lhs.instance,rhs.instance));
	}
	
	public static LLVMConstant fsubtract(LLVMConstant lhs,LLVMConstant rhs) {
		assert(lhs.typeOf().equals(rhs.typeOf()));
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstFSub(lhs.instance,rhs.instance));
	}
	
	public static LLVMConstant multiply(LLVMConstant lhs,LLVMConstant rhs,boolean fp) {
		assert(lhs.typeOf().equals(rhs.typeOf()));
		if(fp)
			return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstFMul(lhs.instance,rhs.instance));
		else
			return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstMul(lhs.instance,rhs.instance));
	}
	
	public static LLVMConstant divide(LLVMConstant lhs,LLVMConstant rhs,DivType type) {
		assert(lhs.typeOf().equals(rhs.typeOf()));
		if(type == DivType.SIGNEDINT)
			return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstSDiv(lhs.instance,rhs.instance));
		else if(type == DivType.UNSIGNEDINT)
			return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstUDiv(lhs.instance,rhs.instance));
		else if(type == DivType.FLOAT)
			return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstFDiv(lhs.instance,rhs.instance));
		return null;
	}
	
	public static LLVMConstant remainder(LLVMConstant lhs,LLVMConstant rhs,DivType type) {
		assert(lhs.typeOf().equals(rhs.typeOf()));
		if(type == DivType.SIGNEDINT)
			return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstSRem(lhs.instance,rhs.instance));
		else if(type == DivType.UNSIGNEDINT)
			return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstURem(lhs.instance,rhs.instance));
		else if(type == DivType.FLOAT)
			return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstFRem(lhs.instance,rhs.instance));
		return null;
	}
	
	public static LLVMConstant or(LLVMConstant lhs,LLVMConstant rhs) {
		assert(lhs.typeOf().equals(rhs.typeOf()));
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstOr(lhs.instance,rhs.instance));
	}
	
	public static LLVMConstant and(LLVMConstant lhs,LLVMConstant rhs) {
		assert(lhs.typeOf().equals(rhs.typeOf()));
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstAnd(lhs.instance,rhs.instance));
	}
	
	public static LLVMConstant xor(LLVMConstant lhs,LLVMConstant rhs) {
		assert(lhs.typeOf().equals(rhs.typeOf()));
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstXor(lhs.instance,rhs.instance));
	}
	
	public static LLVMConstant intComparison(LLVMConstant lhs,LLVMConstant rhs,LLVMIntPredicate pred) {
		assert(lhs.typeOf().equals(rhs.typeOf()));
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstICmp(pred,lhs.instance,rhs.instance));
	}
	
	public static LLVMConstant realComparison(LLVMConstant lhs,LLVMConstant rhs,LLVMRealPredicate pred) {
		assert(lhs.typeOf().equals(rhs.typeOf()));
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstFCmp(pred,lhs.instance,rhs.instance));
	}
	
	public static LLVMConstant shiftLeft(LLVMConstant lhs,LLVMConstant rhs) {
		assert(lhs.typeOf().equals(rhs.typeOf()));
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstShl(lhs.instance,rhs.instance));
	}
	
	/* Pass in false for a logical shift-right (zero-fill) and true for an arithmetic shift-right (sign-extension). */
	public static LLVMConstant shiftRight(LLVMConstant lhs,LLVMConstant rhs,boolean arithmetic) {
		assert(lhs.typeOf().equals(rhs.typeOf()));
		if(arithmetic)
			return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstAShr(lhs.instance,rhs.instance));
		else
			return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstLShr(lhs.instance,rhs.instance));
	}
	
	public static LLVMConstant gep(LLVMConstant op, LLVMConstant... indices) {
		SWIGTYPE_p_p_LLVMOpaqueValue values = Core.new_LLVMValueRefArray(indices.length);
		for(int i=0;i<indices.length;i++)
			Core.LLVMValueRefArray_setitem(values,i,indices[i].getInstance());
		SWIGTYPE_p_LLVMOpaqueValue result = Core.LLVMConstGEP(op.instance,values,indices.length);
		Core.delete_LLVMValueRefArray(values);
		return (LLVMConstant)LLVMValue.getValue(result);
	}
	
	public static LLVMConstant truncate(LLVMConstant op, LLVMIntegerType targetType) {
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstTrunc(op.instance,targetType.getInstance()));
	}
	
	public static LLVMConstant signExtend(LLVMConstant op, LLVMIntegerType targetType) {
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstSExt(op.instance,targetType.getInstance()));
	}
	
	public static LLVMConstant zeroExtend(LLVMConstant op, LLVMIntegerType targetType) {
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstZExt(op.instance,targetType.getInstance()));
	}
	
	public static LLVMConstant fpTruncate(LLVMConstant op, LLVMRealType targetType) {
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstFPTrunc(op.instance,targetType.getInstance()));
	}
	
	public static LLVMConstant fpExtend(LLVMConstant op, LLVMRealType targetType) {
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstFPExt(op.instance,targetType.getInstance()));
	}
	
	public static LLVMConstant uiToFP(LLVMConstant op, LLVMRealType targetType) {
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstUIToFP(op.instance, targetType.getInstance()));
	}
	
	public static LLVMConstant siToFP(LLVMConstant op, LLVMRealType targetType) {
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstSIToFP(op.instance, targetType.getInstance()));
	}
	
	public static LLVMConstant fpToUI(LLVMConstant op, LLVMIntegerType targetType) {
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstFPToUI(op.instance, targetType.getInstance()));
	}
	
	public static LLVMConstant fpToSI(LLVMConstant op, LLVMIntegerType targetType) {
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstFPToSI(op.instance, targetType.getInstance()));
	}
	
	public static LLVMConstant ptrToInt(LLVMConstant op, LLVMIntegerType targetType) {
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstPtrToInt(op.instance, targetType.getInstance()));
	}
	
	public static LLVMConstant intToPtr(LLVMConstant op, LLVMPointerType targetType) {
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstIntToPtr(op.instance, targetType.getInstance()));
	}
	
	public static LLVMConstant bitCast(LLVMConstant op, LLVMType targetType) {
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstBitCast(op.instance, targetType.getInstance()));
	}
	
	public static LLVMConstant select(LLVMConstant condition,LLVMConstant True,LLVMConstant False) {
		assert(condition instanceof LLVMConstantVector || condition instanceof LLVMConstantBoolean);
		assert(True.typeOf().equals(False.typeOf()));
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstSelect(condition.instance,True.instance,False.instance));
	}
	
	public static LLVMConstant shuffleVector(LLVMConstantVector a,LLVMConstantVector b,LLVMConstantVector mask) {
		assert(mask.typeOf() instanceof LLVMVectorType);
		assert(((LLVMVectorType)mask.typeOf()).getElementType() instanceof LLVMIntegerType);
		assert(((LLVMIntegerType)((LLVMVectorType)mask.typeOf()).getElementType()).getWidth() == 32);
		return (LLVMConstant)LLVMValue.getValue(Core.LLVMConstShuffleVector(a.instance,b.instance,mask.instance));
	}
	
	public static LLVMConstant extractValue(LLVMConstant op, long[] indices) {
		SWIGTYPE_p_unsigned_int params = Core.new_UnsignedIntArray(indices.length);
		for(int i=0;i<indices.length;i++)
			Core.UnsignedIntArray_setitem(params,i,indices[i]);
		LLVMConstant result = (LLVMConstant)LLVMValue.getValue(Core.LLVMConstExtractValue(op.instance,params,indices.length));
		Core.delete_UnsignedIntArray(params);
		return result;
	}
	
	public LLVMConstantExpression(SWIGTYPE_p_LLVMOpaqueValue val) {
		super(val);
	}
}
