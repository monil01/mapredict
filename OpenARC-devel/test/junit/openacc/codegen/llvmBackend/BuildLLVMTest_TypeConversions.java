package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcPrimitiveFloatingType.SrcDoubleType;
import static openacc.codegen.llvmBackend.SrcPrimitiveFloatingType.SrcFloatType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcIntType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcLongLongType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcLongType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedCharType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedIntType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedLongLongType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedLongType;
import static org.junit.Assert.assertNotEquals;

import java.math.BigInteger;

import org.jllvm.LLVMContext;
import org.jllvm.LLVMExecutionEngine;
import org.jllvm.LLVMGenericInt;
import org.jllvm.LLVMGenericPointer;
import org.jllvm.LLVMGenericValue;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMTargetData;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Checks the ability to build correct LLVM IR for C type conversions.
 * 
 * <p>
 * Variable initializers were mostly implemented at the time these tests were
 * written. Thus, we use local variable initializers to check type conversions
 * of non-constant expressions. We use global variable initializers to check
 * type conversions of constant expressions because LLVM will fail if we
 * accidentally generate non-constant expressions in that case.
 * </p>
 * 
 * <p>
 * Integer promotions were originally implemented for function arguments, but
 * in my test environment there are many cases for which they seem to have no
 * effect (I guess each function argument is allocated 32 bits minimum, so
 * promoting to 32 bits beforehand doesn't matter unless there's a need for
 * sign extension). We test them thoroughly anyway (in
 * {@link #defaultArgumentPromotions}) just to be sure the integer promotions
 * don't somehow break function arguments. TODO: Also test integer promotions
 * thoroughly for something like unary plus/minus, where we can more easily
 * see if they're having an effect just by testing the resulting sizeof value.
 * </p>
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public class BuildLLVMTest_TypeConversions extends BuildLLVMTest {
  @BeforeClass public static void setup() {
    System.loadLibrary("jllvm");
  }

  // long double is x86_fp80, which has no default alignment so LLVM will fail
  // an assertion if we try to compute its size without this.
  private static final String TARGET_DATA_LAYOUT = "f80:128";
  private static final int SIZEOF_LONG_DOUBLE = 16; // = 128/8

  @Test public void toBool() throws Exception {
    final BigInteger unsignedCharMax
      = BigInteger.ONE.shiftLeft((int)SrcUnsignedCharType.getWidth())
        .subtract(BigInteger.ONE);
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        "_Bool boolToBool(_Bool v) { _Bool b = v; return b; }",
        // Important because they're both represented as i8 even though the
        // logical width of _Bool is 1.
        "_Bool charToBool(char v) { _Bool b = v; return b; }",
        "_Bool intToBool(int v) { _Bool b = v; return b; }",
        "_Bool unsignedToBool(unsigned v) { _Bool b = v; return b; }",
        "_Bool floatToBool(float v) { _Bool b = v; return b; }",
        "_Bool doubleToBool(double v) { _Bool b = v; return b; }",
        "_Bool pointerToBool(void *v) { _Bool b = v; return b; }",

        // Important because they're both represented as i8 even though the
        // logical width of _Bool is 1.
        // Character constants have int type, so these character constants
        // need casts to char in order to exercise char-to-bool conversion.
        "_Bool cCharZeroToBool = (char)'\\0';",
        "_Bool cCharOneToBool = (char)'\\1';",
        "_Bool cCharPosToBool = (char)'\\23';",
        "_Bool cCharNegToBool = (char)'\\x" + unsignedCharMax.toString(16) + "';",

        "_Bool cIntZeroToBool = 0;",
        "_Bool cIntOneToBool = 1;",
        "_Bool cIntPosToBool = 58;",
        "_Bool cIntNegToBool = -1;",

        "_Bool cUnsignedZeroToBool = 0u;",
        "_Bool cUnsignedOneToBool = 1u;",
        "_Bool cUnsignedPosToBool = 23u;",

        "_Bool cFloatZeroToBool = 0.f;",
        "_Bool cFloatPosToBool = 1.f;",
        "_Bool cFloatNegToBool = -3.5f;",
        "_Bool cDoubleZeroToBool = 0.;",
        "_Bool cDoublePosToBool = 1.;",
        "_Bool cDoubleNegToBool = -3.5;",

        "_Bool cPointerNullToBool = (void*)0;",
        "_Bool cPointerNonNullToBool = &cPointerNullToBool;",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    checkIntFn(exec, mod, "boolToBool", getBoolGeneric(0, ctxt), 0);
    checkIntFn(exec, mod, "boolToBool", getBoolGeneric(1, ctxt), 1);

    checkIntFn(exec, mod, "charToBool", getCharGeneric(0, ctxt), 0);
    checkIntFn(exec, mod, "charToBool", getCharGeneric(1, ctxt), 1);
    checkIntFn(exec, mod, "charToBool", getCharGeneric(5, ctxt), 1);
    checkIntFn(exec, mod, "charToBool", getCharGeneric(-13, ctxt), 1);

    checkIntFn(exec, mod, "intToBool", getIntGeneric(0, ctxt), 0);
    checkIntFn(exec, mod, "intToBool", getIntGeneric(1, ctxt), 1);
    checkIntFn(exec, mod, "intToBool", getIntGeneric(2, ctxt), 1);
    checkIntFn(exec, mod, "intToBool", getIntGeneric(-100, ctxt), 1);

    checkIntFn(exec, mod, "unsignedToBool", getUnsignedIntGeneric(0, ctxt), 0);
    checkIntFn(exec, mod, "unsignedToBool", getUnsignedIntGeneric(1, ctxt), 1);
    checkIntFn(exec, mod, "unsignedToBool", getUnsignedIntGeneric(34, ctxt), 1);

    checkIntFn(exec, mod, "floatToBool", getGenericReal(SrcFloatType, 0., ctxt), 0);
    checkIntFn(exec, mod, "floatToBool", getGenericReal(SrcFloatType, .3, ctxt), 1);
    checkIntFn(exec, mod, "floatToBool", getGenericReal(SrcFloatType, -1., ctxt), 1);
    checkIntFn(exec, mod, "doubleToBool", getGenericReal(SrcDoubleType, 0., ctxt), 0);
    checkIntFn(exec, mod, "doubleToBool", getGenericReal(SrcDoubleType, 5e9, ctxt), 1);
    checkIntFn(exec, mod, "doubleToBool", getGenericReal(SrcDoubleType, -3.5e3, ctxt), 1);

    checkIntFn(exec, mod, "pointerToBool", new LLVMGenericPointer(null), 0);
    checkIntFn(exec, mod, "pointerToBool", new LLVMGenericPointer(new Ptr(2)), 1);

    checkGlobalVar(mod, "cCharZeroToBool", getConstBool(0, ctxt));
    checkGlobalVar(mod, "cCharOneToBool", getConstBool(1, ctxt));
    checkGlobalVar(mod, "cCharPosToBool", getConstBool(1, ctxt));
    checkGlobalVar(mod, "cCharNegToBool", getConstBool(1, ctxt));

    checkGlobalVar(mod, "cIntZeroToBool", getConstBool(0, ctxt));
    checkGlobalVar(mod, "cIntOneToBool", getConstBool(1, ctxt));
    checkGlobalVar(mod, "cIntPosToBool", getConstBool(1, ctxt));
    checkGlobalVar(mod, "cIntNegToBool", getConstBool(1, ctxt));

    checkGlobalVar(mod, "cUnsignedZeroToBool", getConstBool(0, ctxt));
    checkGlobalVar(mod, "cUnsignedOneToBool", getConstBool(1, ctxt));
    checkGlobalVar(mod, "cUnsignedPosToBool", getConstBool(1, ctxt));

    checkGlobalVar(mod, "cFloatZeroToBool", getConstBool(0, ctxt));
    checkGlobalVar(mod, "cFloatPosToBool", getConstBool(1, ctxt));
    checkGlobalVar(mod, "cFloatNegToBool", getConstBool(1, ctxt));
    checkGlobalVar(mod, "cDoubleZeroToBool", getConstBool(0, ctxt));
    checkGlobalVar(mod, "cDoublePosToBool", getConstBool(1, ctxt));
    checkGlobalVar(mod, "cDoubleNegToBool", getConstBool(1, ctxt));

    checkGlobalVar(mod, "cPointerNullToBool", getConstBool(0, ctxt));
    checkGlobalVar(mod, "cPointerNonNullToBool", getConstBool(1, ctxt));
    exec.dispose();
  }

  @Test public void toSignedChar() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        "signed char charToChar(signed char v) { signed char r = v; return r; }",
        // Important because, even though the logical width of _Bool is 1,
        // they're both represented as i8, so LLVM's zext to i8 would fail.
        "signed char boolToChar(_Bool v) { signed char r = v; return r; }",
        "signed char intToChar(int v) { signed char r = v; return r; }",
        "signed char unsignedToChar(unsigned v) { signed char r = v; return r; }",
        "signed char floatToChar(float v) { signed char r = v; return r; }",
        "signed char doubleToChar(double v) { signed char r = v; return r; }",
        "signed char pointerToChar(void *v) { signed char r = (signed char)v; return r; }",

        // Important because, even though the logical width of _Bool is 1,
        // they're both represented as i8, so LLVM's zext to i8 would fail.
        "signed char cBoolZeroToChar = (_Bool)0;",
        "signed char cBoolOneToChar = (_Bool)1;",

        "signed char cIntZeroToChar = 0;",
        "signed char cIntOneToChar = 1;",
        "signed char cIntPosToChar = 58;",
        "signed char cIntNegToChar = -1;",

        "signed char cUnsignedZeroToChar = 0u;",
        "signed char cUnsignedOneToChar = 1u;",
        "signed char cUnsignedPosToChar = 23u;",

        "signed char cFloatZeroToChar = 0.f;",
        "signed char cFloatPosToChar = 1.f;",
        "signed char cFloatNegToChar = -3.5f;",
        "signed char cDoubleZeroToChar = 0.;",
        "signed char cDoublePosToChar = 1.;",
        "signed char cDoubleNegToChar = -3.5;",

        "signed char cPointerNullToChar = (signed char)(void*)0;",
        "signed char cPointerNonNullToChar = (signed char)&cPointerNullToChar;",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    checkIntFn(exec, mod, "charToChar", getCharGeneric(0, ctxt), 0);
    checkIntFn(exec, mod, "charToChar", getCharGeneric(1, ctxt), 1);
    checkIntFn(exec, mod, "charToChar", getCharGeneric(3, ctxt), 3);
    checkIntFn(exec, mod, "charToChar", getCharGeneric(-5, ctxt), -5);

    checkIntFn(exec, mod, "boolToChar", getBoolGeneric(0, ctxt), 0);
    checkIntFn(exec, mod, "boolToChar", getBoolGeneric(1, ctxt), 1);

    checkIntFn(exec, mod, "intToChar", getIntGeneric(0, ctxt), 0);
    checkIntFn(exec, mod, "intToChar", getIntGeneric(1, ctxt), 1);
    checkIntFn(exec, mod, "intToChar", getIntGeneric(2, ctxt), 2);
    checkIntFn(exec, mod, "intToChar", getIntGeneric(-100, ctxt), -100);

    checkIntFn(exec, mod, "unsignedToChar", getUnsignedIntGeneric(0, ctxt), 0);
    checkIntFn(exec, mod, "unsignedToChar", getUnsignedIntGeneric(1, ctxt), 1);
    checkIntFn(exec, mod, "unsignedToChar", getUnsignedIntGeneric(34, ctxt), 34);

    checkIntFn(exec, mod, "floatToChar", getGenericReal(SrcFloatType, 0., ctxt), 0);
    checkIntFn(exec, mod, "floatToChar", getGenericReal(SrcFloatType, .3, ctxt), 0);
    checkIntFn(exec, mod, "floatToChar", getGenericReal(SrcFloatType, -1., ctxt), -1);
    checkIntFn(exec, mod, "doubleToChar", getGenericReal(SrcDoubleType, 0., ctxt), 0);
    checkIntFn(exec, mod, "doubleToChar", getGenericReal(SrcDoubleType, 5e1, ctxt), 50);
    checkIntFn(exec, mod, "doubleToChar", getGenericReal(SrcDoubleType, -3.5e1, ctxt), -35);

    checkIntFn(exec, mod, "pointerToChar", new LLVMGenericPointer(null), 0);
    checkIntFn(exec, mod, "pointerToChar", new LLVMGenericPointer(new Ptr(2)), 2);

    checkGlobalVar(mod, "cBoolZeroToChar", getConstSignedChar(0, ctxt));
    checkGlobalVar(mod, "cBoolOneToChar", getConstSignedChar(1, ctxt));

    checkGlobalVar(mod, "cIntZeroToChar", getConstSignedChar(0, ctxt));
    checkGlobalVar(mod, "cIntOneToChar", getConstSignedChar(1, ctxt));
    checkGlobalVar(mod, "cIntPosToChar", getConstSignedChar(58, ctxt));
    checkGlobalVar(mod, "cIntNegToChar", getConstSignedChar(-1, ctxt));

    checkGlobalVar(mod, "cUnsignedZeroToChar", getConstSignedChar(0, ctxt));
    checkGlobalVar(mod, "cUnsignedOneToChar", getConstSignedChar(1, ctxt));
    checkGlobalVar(mod, "cUnsignedPosToChar", getConstSignedChar(23, ctxt));

    checkGlobalVar(mod, "cFloatZeroToChar", getConstSignedChar(0, ctxt));
    checkGlobalVar(mod, "cFloatPosToChar", getConstSignedChar(1, ctxt));
    checkGlobalVar(mod, "cFloatNegToChar", getConstSignedChar(-3, ctxt));
    checkGlobalVar(mod, "cDoubleZeroToChar", getConstSignedChar(0, ctxt));
    checkGlobalVar(mod, "cDoublePosToChar", getConstSignedChar(1, ctxt));
    checkGlobalVar(mod, "cDoubleNegToChar", getConstSignedChar(-3, ctxt));

    checkGlobalVar(mod, "cPointerNullToChar", getConstSignedChar(0, ctxt));
    assertNotEquals("cPointerNullToChar initializer's value must not be zero",
                    checkGlobalVarGetInit(mod, "cPointerNonNullToChar"),
                    getConstSignedChar(0, ctxt));
    exec.dispose();
  }

  @Test public void toUnsignedInt() throws Exception {
    final BigInteger unsignedCharMax
      = BigInteger.ONE.shiftLeft((int)SrcUnsignedCharType.getWidth())
        .subtract(BigInteger.ONE);
    final BigInteger unsignedMaxPlusOne
      = BigInteger.ONE.shiftLeft((int)SrcUnsignedIntType.getWidth());
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        "unsigned boolToUnsigned(_Bool f)                 { unsigned t = f; return t; }",
        "unsigned unsignedCharToUnsigned(unsigned char f) { unsigned t = f; return t; }",
        "unsigned signedCharToUnsigned(signed char f)     { unsigned t = f; return t; }",
        "unsigned unsignedToUnsigned(unsigned f)          { unsigned t = f; return t; }",
        "unsigned intToUnsigned(int f)                    { unsigned t = f; return t; }",
        "unsigned unsignedLongToUnsigned(unsigned long f) { unsigned t = f; return t; }",
        "unsigned longToUnsigned(long f)                  { unsigned t = f; return t; }",
        "unsigned floatToUnsigned(float f)                { unsigned t = f; return t; }",
        "unsigned doubleToUnsigned(double f)              { unsigned t = f; return t; }",

        // Character constants have int type, so these character constants
        // need casts to (un)signed char in order to exercise
        // (un)signed-char-to-unsigned conversion.
        "unsigned cUnsignedCharPosToUnsigned = (unsigned char)'\\7';",
        "unsigned cUnsignedCharMaxToUnsigned = (unsigned char)'\\x" + unsignedCharMax.toString(16) + "';",
        "unsigned cSignedCharPosToUnsigned = (signed char)'\\5';",
        "unsigned cSignedCharNegToUnsigned = (signed char)'\\x" + unsignedCharMax.toString(16) + "';",

        "unsigned cUnsignedToUnsigned = 58u;",
        "unsigned cIntPosToUnsigned = 268;",
        "unsigned cIntNegToUnsigned = -987;",
        "unsigned cUnsignedLongSmallToUnsigned = 1ul;",
        "unsigned cUnsignedLongLargeToUnsigned = " + unsignedMaxPlusOne.toString() + "ul;",
        "unsigned cLongPosToUnsigned = " + unsignedMaxPlusOne.add(BigInteger.ONE)
                                           .add(BigInteger.ONE).toString() + "l;",
        "unsigned cLongNegToUnsigned = -8l;",

        "unsigned cFloatEvenToUnsigned = 1.f;",
        "unsigned cFloatFiveToUnsigned = 305e-1f;",
        "unsigned cFloatLtFiveToUnsigned = 89.001e1f;",
        "unsigned cFloatGtFiveToUnsigned = 143.78f;",
        "unsigned cFloatNegLtOneToUnsigned = -0.5f;",
        "unsigned cFloatNegGtOneToUnsigned = -3.2f;",
        "unsigned cDoubleEvenToUnsigned = 0.;",
        "unsigned cDoubleFiveToUnsigned = 6050.5;",
        "unsigned cDoubleLtFiveToUnsigned = 234.5234e1;",
        "unsigned cDoubleGtFiveToUnsigned = 987.89e-1;",
        "unsigned cDoubleNegLtOneToUnsigned = -0.873;",
        "unsigned cDoubleNegGtOneToUnsigned = -523.3;",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    checkIntFn(exec, mod, "boolToUnsigned", getBoolGeneric(0, ctxt), 0);
    checkIntFn(exec, mod, "boolToUnsigned", getBoolGeneric(1, ctxt), 1);
    checkIntFn(exec, mod, "unsignedCharToUnsigned", getUnsignedCharGeneric(53, ctxt), 53);
    checkIntFn(exec, mod, "unsignedCharToUnsigned", getUnsignedCharGeneric(-1, ctxt),
               unsignedCharMax.longValue());
    checkIntFn(exec, mod, "signedCharToUnsigned", getSignedCharGeneric(3, ctxt), 3);
    checkIntFn(exec, mod, "signedCharToUnsigned", getSignedCharGeneric(-1, ctxt), -1);
    checkIntFn(exec, mod, "unsignedToUnsigned", getUnsignedIntGeneric(81, ctxt), 81);
    checkIntFn(exec, mod, "intToUnsigned", getIntGeneric(81, ctxt), 81);
    checkIntFn(exec, mod, "intToUnsigned", getIntGeneric(-90, ctxt), -90);
    checkIntFn(exec, mod, "unsignedLongToUnsigned", getUnsignedLongGeneric(29, ctxt), 29);
    checkIntFn(exec, mod, "unsignedLongToUnsigned", getUnsignedLongGeneric(-5, ctxt), -5);
    checkIntFn(exec, mod, "longToUnsigned", getLongGeneric(23, ctxt), 23);
    checkIntFn(exec, mod, "longToUnsigned",
               new LLVMGenericInt(SrcLongType.getLLVMType(ctxt),
                                  unsignedMaxPlusOne.add(BigInteger.ONE),
                                  false),
               1);
    checkIntFn(exec, mod, "longToUnsigned", getLongGeneric(-100, ctxt), -100);

    checkIntFn(exec, mod, "floatToUnsigned", getGenericReal(SrcFloatType, 0., ctxt), 0);
    checkIntFn(exec, mod, "floatToUnsigned", getGenericReal(SrcFloatType, 1.5, ctxt), 1);
    checkIntFn(exec, mod, "floatToUnsigned", getGenericReal(SrcFloatType, 30.22e1, ctxt), 302);
    checkIntFn(exec, mod, "floatToUnsigned", getGenericReal(SrcFloatType, 66e-1, ctxt), 6);
    checkIntFn(exec, mod, "floatToUnsigned", getGenericReal(SrcFloatType, -0.9, ctxt), 0);
    checkIntFn(exec, mod, "floatToUnsigned", getGenericReal(SrcFloatType, -1.9, ctxt), -1);
    checkIntFn(exec, mod, "doubleToUnsigned", getGenericReal(SrcDoubleType, 23e1, ctxt), 230);
    checkIntFn(exec, mod, "doubleToUnsigned", getGenericReal(SrcDoubleType, 9.5, ctxt), 9);
    checkIntFn(exec, mod, "doubleToUnsigned", getGenericReal(SrcDoubleType, 3098.22e-1, ctxt), 309);
    checkIntFn(exec, mod, "doubleToUnsigned", getGenericReal(SrcDoubleType, 14321e-2, ctxt), 143);
    checkIntFn(exec, mod, "doubleToUnsigned", getGenericReal(SrcDoubleType, -0.2, ctxt), 0);
    checkIntFn(exec, mod, "doubleToUnsigned", getGenericReal(SrcDoubleType, -50.4, ctxt), -50);

    checkGlobalVar(mod, "cUnsignedCharPosToUnsigned", getConstUnsigned(7, ctxt));
    checkGlobalVar(mod, "cUnsignedCharMaxToUnsigned", getConstUnsigned(unsignedCharMax.longValue(), ctxt));
    checkGlobalVar(mod, "cSignedCharPosToUnsigned", getConstUnsigned(5, ctxt));
    checkGlobalVar(mod, "cSignedCharNegToUnsigned", getConstUnsigned(-1, ctxt));

    checkGlobalVar(mod, "cUnsignedToUnsigned", getConstUnsigned(58, ctxt));
    checkGlobalVar(mod, "cIntPosToUnsigned", getConstUnsigned(268, ctxt));
    checkGlobalVar(mod, "cIntNegToUnsigned", getConstUnsigned(-987, ctxt));
    checkGlobalVar(mod, "cUnsignedLongSmallToUnsigned", getConstUnsigned(1, ctxt));
    checkGlobalVar(mod, "cUnsignedLongLargeToUnsigned", getConstUnsigned(0, ctxt));
    checkGlobalVar(mod, "cLongPosToUnsigned", getConstUnsigned(2, ctxt));
    checkGlobalVar(mod, "cLongNegToUnsigned", getConstUnsigned(-8, ctxt));

    checkGlobalVar(mod, "cFloatEvenToUnsigned", getConstUnsigned(1, ctxt));
    checkGlobalVar(mod, "cFloatFiveToUnsigned", getConstUnsigned(30, ctxt));
    checkGlobalVar(mod, "cFloatLtFiveToUnsigned", getConstUnsigned(890, ctxt));
    checkGlobalVar(mod, "cFloatGtFiveToUnsigned", getConstUnsigned(143, ctxt));
    checkGlobalVar(mod, "cFloatNegLtOneToUnsigned", getConstUnsigned(0, ctxt));
    checkGlobalVar(mod, "cFloatNegGtOneToUnsigned", getConstUnsigned(0, ctxt));
    checkGlobalVar(mod, "cDoubleEvenToUnsigned", getConstUnsigned(0, ctxt));
    checkGlobalVar(mod, "cDoubleFiveToUnsigned", getConstUnsigned(6050, ctxt));
    checkGlobalVar(mod, "cDoubleLtFiveToUnsigned", getConstUnsigned(2345, ctxt));
    checkGlobalVar(mod, "cDoubleGtFiveToUnsigned", getConstUnsigned(98, ctxt));
    checkGlobalVar(mod, "cDoubleNegLtOneToUnsigned", getConstUnsigned(0, ctxt));
    checkGlobalVar(mod, "cDoubleNegGtOneToUnsigned", getConstUnsigned(0, ctxt));
    exec.dispose();
  }

  @Test public void toInt() throws Exception {
    final BigInteger unsignedCharMax
      = BigInteger.ONE.shiftLeft((int)SrcUnsignedCharType.getWidth())
        .subtract(BigInteger.ONE);
    final BigInteger intPosMaxPlusOne
      = BigInteger.ONE.shiftLeft((int)SrcIntType.getPosWidth());
    final BigInteger unsignedMax
      = BigInteger.ONE.shiftLeft((int)SrcUnsignedIntType.getWidth())
        .subtract(BigInteger.ONE);
    final BigInteger longNegMax
      = BigInteger.ONE.shiftLeft((int)SrcLongType.getPosWidth()).negate();
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        "int boolToInt(_Bool f)                 { int t = f; return t; }",
        "int unsignedCharToInt(unsigned char f) { int t = f; return t; }",
        "int signedCharToInt(signed char f)     { int t = f; return t; }",
        "int unsignedToInt(unsigned f)          { int t = f; return t; }",
        "int intToInt(int f)                    { int t = f; return t; }",
        "int unsignedLongToInt(unsigned long f) { int t = f; return t; }",
        "int longToInt(long f)                  { int t = f; return t; }",
        "int floatToInt(float f)                { int t = f; return t; }",
        "int doubleToInt(double f)              { int t = f; return t; }",

        // Character constants have int type, so these character constants
        // need casts to (un)signed char in order to exercise
        // (un)signed-char-to-int conversion.
        "int cUnsignedCharPosToInt = (unsigned char)'\\20';",
        "int cUnsignedCharMaxToInt = (unsigned char)'\\x" + unsignedCharMax.toString(16) + "';",
        "int cSignedCharPosToInt = (signed char)'\\5';",
        "int cSignedCharNegToInt = (signed char)'\\x" + unsignedCharMax.toString(16) + "';",

        "int cUnsignedSmallToInt = 32u;",
        "int cUnsignedMaxToInt = 0x" + unsignedMax.toString(16) + "u;",
        "int cIntPosToInt = 195;",
        "int cIntNegToInt = -23;",
        "int cUnsignedLongSmallToInt = 89ul;",
        "int cUnsignedLongLargeToNegInt = " + unsignedMax.toString() + "ul;",
        "int cUnsignedLongLargeToPosInt = " + unsignedMax.add(BigInteger.ONE)
                                              .add(BigInteger.ONE).toString()
                                              + "ul;",
        "int cLongPosSmallToInt = 78l;",
        "int cLongPosLargeToInt = 0x" + unsignedMax.toString(16) + "l;",
        "int cLongNegSmallToInt = -29l;",
        "int cLongNegLargeToInt = " + longNegMax.add(BigInteger.ONE).toString()
                                    + "l;",

        "int cFloatEvenToInt = 1.f;",
        "int cFloatFiveToInt = 305e-1f;",
        "int cFloatLtFiveToInt = 89.001e1f;",
        "int cFloatGtFiveToInt = 143.78f;",
        "int cFloatNegLtOneToInt = -0.5f;",
        "int cFloatNegGtOneToInt = -3.2f;",
        "int cDoubleEvenToInt = 0.;",
        "int cDoubleFiveToInt = 6050.5;",
        "int cDoubleLtFiveToInt = 234.5234e1;",
        "int cDoubleGtFiveToInt = 987.89e-1;",
        "int cDoubleNegLtOneToInt = -0.873;",
        "int cDoubleNegGtOneToInt = -523.3;",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    checkIntFn(exec, mod, "boolToInt", getBoolGeneric(0, ctxt), 0);
    checkIntFn(exec, mod, "boolToInt", getBoolGeneric(1, ctxt), 1);
    checkIntFn(exec, mod, "unsignedCharToInt", getUnsignedCharGeneric(53, ctxt), 53);
    checkIntFn(exec, mod, "unsignedCharToInt",
               new LLVMGenericInt(SrcUnsignedCharType.getLLVMType(ctxt),
                                  unsignedCharMax, false),
               unsignedCharMax.longValue());
    checkIntFn(exec, mod, "signedCharToInt", getSignedCharGeneric(3, ctxt), 3);
    checkIntFn(exec, mod, "signedCharToInt", getSignedCharGeneric(-35, ctxt), -35);
    checkIntFn(exec, mod, "unsignedToInt", getUnsignedIntGeneric(81, ctxt), 81);
    checkIntFn(exec, mod, "unsignedToInt", getUnsignedIntGeneric(-3, ctxt), -3);
    checkIntFn(exec, mod, "intToInt", getIntGeneric(97, ctxt), 97);
    checkIntFn(exec, mod, "intToInt", getIntGeneric(-235, ctxt), -235);
    checkIntFn(exec, mod, "unsignedLongToInt", getUnsignedLongGeneric(29, ctxt), 29);
    checkIntFn(exec, mod, "unsignedLongToInt", getUnsignedLongGeneric(-5, ctxt), -5);
    checkIntFn(exec, mod, "unsignedLongToInt",
               new LLVMGenericInt(SrcUnsignedLongType.getLLVMType(ctxt),
                                  unsignedMax.add(BigInteger.ONE), false),
               0);
    checkIntFn(exec, mod, "longToInt", getLongGeneric(23, ctxt), 23);
    checkIntFn(exec, mod, "longToInt",
               new LLVMGenericInt(SrcLongType.getLLVMType(ctxt),
                                  intPosMaxPlusOne, false),
               intPosMaxPlusOne.negate().longValue());
    checkIntFn(exec, mod, "longToInt", getLongGeneric(-100, ctxt), -100);
    checkIntFn(exec, mod, "longToInt",
               new LLVMGenericInt(SrcLongType.getLLVMType(ctxt),
                                  intPosMaxPlusOne.add(BigInteger.ONE).negate(),
                                  true),
               intPosMaxPlusOne.subtract(BigInteger.ONE).longValue());

    checkIntFn(exec, mod, "floatToInt", getGenericReal(SrcFloatType, 0., ctxt), 0);
    checkIntFn(exec, mod, "floatToInt", getGenericReal(SrcFloatType, 1.5, ctxt), 1);
    checkIntFn(exec, mod, "floatToInt", getGenericReal(SrcFloatType, 30.22e1, ctxt), 302);
    checkIntFn(exec, mod, "floatToInt", getGenericReal(SrcFloatType, 66e-1, ctxt), 6);
    checkIntFn(exec, mod, "floatToInt", getGenericReal(SrcFloatType, -0.9, ctxt), 0);
    checkIntFn(exec, mod, "floatToInt", getGenericReal(SrcFloatType, -1.3, ctxt), -1);
    checkIntFn(exec, mod, "doubleToInt", getGenericReal(SrcDoubleType, 23e1, ctxt), 230);
    checkIntFn(exec, mod, "doubleToInt", getGenericReal(SrcDoubleType, 9.5, ctxt), 9);
    checkIntFn(exec, mod, "doubleToInt", getGenericReal(SrcDoubleType, 3098.22e-1, ctxt), 309);
    checkIntFn(exec, mod, "doubleToInt", getGenericReal(SrcDoubleType, 14321e-2, ctxt), 143);
    checkIntFn(exec, mod, "doubleToInt", getGenericReal(SrcDoubleType, -0.2, ctxt), 0);
    checkIntFn(exec, mod, "doubleToInt", getGenericReal(SrcDoubleType, -50.8, ctxt), -50);

    checkGlobalVar(mod, "cUnsignedCharPosToInt", getConstInt(16, ctxt));
    checkGlobalVar(mod, "cUnsignedCharMaxToInt", getConstInt(unsignedCharMax.longValue(), ctxt));
    checkGlobalVar(mod, "cSignedCharPosToInt", getConstInt(5, ctxt));
    checkGlobalVar(mod, "cSignedCharNegToInt", getConstInt(-1, ctxt));

    checkGlobalVar(mod, "cUnsignedSmallToInt", getConstInt(32, ctxt));
    checkGlobalVar(mod, "cUnsignedMaxToInt", getConstInt(-1, ctxt));
    checkGlobalVar(mod, "cIntPosToInt", getConstInt(195, ctxt));
    checkGlobalVar(mod, "cIntNegToInt", getConstInt(-23, ctxt));
    checkGlobalVar(mod, "cUnsignedLongSmallToInt", getConstInt(89, ctxt));
    checkGlobalVar(mod, "cUnsignedLongLargeToNegInt", getConstInt(-1, ctxt));
    checkGlobalVar(mod, "cUnsignedLongLargeToPosInt", getConstInt(1, ctxt));
    checkGlobalVar(mod, "cLongPosSmallToInt", getConstInt(78, ctxt));
    checkGlobalVar(mod, "cLongPosLargeToInt", getConstInt(-1, ctxt));
    checkGlobalVar(mod, "cLongNegSmallToInt", getConstInt(-29, ctxt));
    checkGlobalVar(mod, "cLongNegLargeToInt", getConstInt(1, ctxt));

    checkGlobalVar(mod, "cFloatEvenToInt", getConstInt(1, ctxt));
    checkGlobalVar(mod, "cFloatFiveToInt", getConstInt(30, ctxt));
    checkGlobalVar(mod, "cFloatLtFiveToInt", getConstInt(890, ctxt));
    checkGlobalVar(mod, "cFloatGtFiveToInt", getConstInt(143, ctxt));
    checkGlobalVar(mod, "cFloatNegLtOneToInt", getConstInt(0, ctxt));
    checkGlobalVar(mod, "cFloatNegGtOneToInt", getConstInt(-3, ctxt));
    checkGlobalVar(mod, "cDoubleEvenToInt", getConstInt(0, ctxt));
    checkGlobalVar(mod, "cDoubleFiveToInt", getConstInt(6050, ctxt));
    checkGlobalVar(mod, "cDoubleLtFiveToInt", getConstInt(2345, ctxt));
    checkGlobalVar(mod, "cDoubleGtFiveToInt", getConstInt(98, ctxt));
    checkGlobalVar(mod, "cDoubleNegLtOneToInt", getConstInt(0, ctxt));
    checkGlobalVar(mod, "cDoubleNegGtOneToInt", getConstInt(-523, ctxt));
    exec.dispose();
  }

  @Test public void toFP() throws Exception {
    final BigInteger unsignedCharMax
      = BigInteger.ONE.shiftLeft((int)SrcUnsignedCharType.getWidth())
        .subtract(BigInteger.ONE);
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        "float unsignedCharToFloat(unsigned char f) { float t = f; return t; }",
        "float signedCharToFloat(signed char f) { float t = f; return t; }",
        "double unsignedCharToDouble(unsigned char f) { double t = f; return t; }",
        "double signedCharToDouble(signed char f) { double t = f; return t; }",
        "double floatToDouble(float f) { double t = f; return t; }",
        "float doubleToFloat(double f) { float t = f; return t; }",

        // Character constants have int type, so these character constants
        // need casts to (un)signed char in order to exercise
        // (un)signed-char-to-float conversion.
        //
        // Unary - must be applied before cast because it applies integer
        // promotions.
        "float cUnsignedCharZeroToFloat = (unsigned char)'\\0';",
        "float cUnsignedCharPosToFloat = (unsigned char)'\\5';",
        "float cUnsignedCharMaxToFloat = (unsigned char)'\\x" + unsignedCharMax.toString(16) + "';",
        "float cSignedCharZeroToFloat = (signed char)'\\0';",
        "float cSignedCharPosToFloat = (signed char)'\\x20';",
        "float cSignedCharMaxToFloat = (signed char)-'\\1';",
        "double cUnsignedCharZeroToDouble = (unsigned char)'\\0';",
        "double cUnsignedCharPosToDouble = (unsigned char)'\\5';",
        "double cUnsignedCharMaxToDouble = (unsigned char)'\\x" + unsignedCharMax.toString(16) + "';",
        "double cSignedCharZeroToDouble = (signed char)'\\0';",
        "double cSignedCharPosToDouble = (signed char)'\\x20';",
        "double cSignedCharMaxToDouble = (signed char)-'\\1';",

        "double cFloatToDouble = -3.5e8f;",
        "float cDoubleToFloat = 10e-7;",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    checkFloatFn(exec, mod, "unsignedCharToFloat", getUnsignedCharGeneric(0, ctxt), 0.);
    checkFloatFn(exec, mod, "unsignedCharToFloat", getUnsignedCharGeneric(105, ctxt), 105.);
    checkFloatFn(exec, mod, "unsignedCharToFloat",
                 new LLVMGenericInt(SrcUnsignedCharType.getLLVMType(ctxt),
                                    unsignedCharMax, false),
                 unsignedCharMax.doubleValue());
    checkFloatFn(exec, mod, "signedCharToFloat", getSignedCharGeneric(0, ctxt), 0.);
    checkFloatFn(exec, mod, "signedCharToFloat", getSignedCharGeneric(98, ctxt), 98.);
    checkFloatFn(exec, mod, "signedCharToFloat", getSignedCharGeneric(-1, ctxt), -1.);

    checkDoubleFn(exec, mod, "unsignedCharToDouble", getUnsignedCharGeneric(0, ctxt), 0.);
    checkDoubleFn(exec, mod, "unsignedCharToDouble", getUnsignedCharGeneric(105, ctxt), 105.);
    checkDoubleFn(exec, mod, "unsignedCharToDouble",
                  new LLVMGenericInt(SrcUnsignedCharType.getLLVMType(ctxt),
                                     unsignedCharMax, false),
                  unsignedCharMax.doubleValue());
    checkDoubleFn(exec, mod, "signedCharToDouble", getSignedCharGeneric(0, ctxt), 0.);
    checkDoubleFn(exec, mod, "signedCharToDouble", getSignedCharGeneric(98, ctxt), 98.);
    checkDoubleFn(exec, mod, "signedCharToDouble", getSignedCharGeneric(-1, ctxt), -1.);
    checkDoubleFn(exec, mod, "floatToDouble", getGenericReal(SrcFloatType, 5e10, ctxt), 5e10);
    checkFloatFn(exec, mod, "doubleToFloat", getGenericReal(SrcDoubleType, -3e-5, ctxt), -3e-5);

    checkGlobalVar(mod, "cUnsignedCharZeroToFloat", getConstFloat(0., ctxt));
    checkGlobalVar(mod, "cUnsignedCharPosToFloat", getConstFloat(5., ctxt));
    checkGlobalVar(mod, "cUnsignedCharMaxToFloat", getConstFloat(unsignedCharMax.floatValue(), ctxt));
    checkGlobalVar(mod, "cSignedCharZeroToFloat", getConstFloat(0., ctxt));
    checkGlobalVar(mod, "cSignedCharPosToFloat", getConstFloat(32., ctxt));
    checkGlobalVar(mod, "cSignedCharMaxToFloat", getConstFloat(-1., ctxt));

    checkGlobalVar(mod, "cUnsignedCharZeroToDouble", getConstDouble(0., ctxt));
    checkGlobalVar(mod, "cUnsignedCharPosToDouble", getConstDouble(5., ctxt));
    checkGlobalVar(mod, "cUnsignedCharMaxToDouble", getConstDouble(unsignedCharMax.doubleValue(), ctxt));
    checkGlobalVar(mod, "cSignedCharZeroToDouble", getConstDouble(0., ctxt));
    checkGlobalVar(mod, "cSignedCharPosToDouble", getConstDouble(32., ctxt));
    checkGlobalVar(mod, "cSignedCharMaxToDouble", getConstDouble(-1., ctxt));

    checkGlobalVar(mod, "cFloatToDouble", getConstDouble(-3.5e8, ctxt));
    checkGlobalVar(mod, "cDoubleToFloat", getConstFloat(10e-7, ctxt));
    exec.dispose();
  }

  @Test public void toPointer() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        // -129 so that the compatible integer type is likely int. At the time
        // this was written, BuildLLVM always chose int, but other compilers
        // (clang 3.5.1) sometimes choose unsigned when possible, so BuildLLVM
        // may do so too some day.
        "enum E {A = -129};",

        // null pointers

        "void *vNull = 0;",
        "int *iNull = 0;",
        "void (*fnNull)() = 0;",
        "void *getVNull() { return vNull; }",
        "int *getINull() { return iNull; }",
        "void (*getFnNull())() { return fnNull; }",

        // identical types

        "int init;",
        "int *getInit() { return &init; }",
        "int *i2i(int *arg) { int *res = arg; return res; }",
        "double *d2d(double *arg) { double *res = arg; return res; }",
        "enum E *e2e(enum E *arg) { enum E *res = arg; return res; }",
        "int (*au2au(int (*arg)[]))[] { int (*res)[] = arg; return res; }",
        "int (*a12a1(int (*arg)[1]))[1] { int (*res)[1] = arg; return res; }",
        "void (*fn2fn(void (*arg)()))() { void (*res)() = arg; return res; }",
        "struct S *s2s(struct S *arg) { struct S *res = arg; return res; }",
        "union S *u2u(union S *arg) { union S *res = arg; return res; }",

        "int *ci2i = &init;",
        "int *get_ci2i() { return ci2i; }",

        // void*

        "void *i2v(int *arg) { void *res = arg; return res; }",
        "int *v2i(void *arg) { int *res = arg; return res; }",
        "void *v2v(void *arg) { void *res = arg; return res; }",
        "struct S *v2s(void *arg) { struct S *res = arg; return res; }",
        "void *u2v(union U *arg) { void *res = arg; return res; }",

        "void *ci2v = &init;",
        "int *cv2i = (void*)&init;",
        "void *cv2v = (void*)&init;",
        "void *get_ci2v() { return ci2v; }",
        "int *get_cv2i() { return cv2i; }",
        "void *get_cv2v() { return cv2v; }",

        // compatible types

        "int *e2i(enum E *arg) { int *res = arg; return res; }",
        "enum E *i2e(int *arg) { enum E *res = arg; return res; }",

        "int (*eau2iau(enum E (*arg)[]))[] { int (*res)[] = arg; return res; }",
        "enum E (*iau2ea5(int (*arg)[]))[5] { enum E (*res)[5] = arg; return res; }",
        "int (*ia32iau(int (*arg)[3]))[] { int (*res)[] = arg; return res; }",
        "int (*ea72ia7(enum E (*arg)[7]))[7] { int (*res)[7] = arg; return res; }",

        "enum E (*efn02ifn0(int (*arg)()))() { enum E (*res)() = arg; return res; }",
        "enum E (*efnv2ifnv(int (*arg)(void)))(void) { enum E (*res)(void) = arg; return res; }",
        "enum E (*efn12ifn1(int (*arg)(enum E)))(int) { enum E (*res)(int) = arg; return res; }",
        "enum E (*efn1var2ifn1var(int (*arg)(enum E, ...)))(int, ...) { enum E (*res)(int, ...) = arg; return res; }",
        "enum E (*efn22ifn2(int (*arg)(enum E, int)))(int, enum E) { enum E (*res)(int, enum E) = arg; return res; }",
        "enum E (*efn2var2ifn2var(int (*arg)(enum E, int, ...)))(int, enum E, ...) { enum E (*res)(int, enum E, ...) = arg; return res; }",
        "int (*ifn02ifn1(int (*arg)()))(int) { int (*res)(int) = arg; return res; }",
        "int (*ifn12ifn0(int (*arg)(int)))() { int (*res)() = arg; return res; }",
        "int (*ifn02ifn2(int (*arg)()))(double, void*) { int (*res)(double, void*) = arg; return res; }",
        "int (*ifn22ifn0(int (*arg)(long double, unsigned long)))() { int (*res)() = arg; return res; }",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);

    // null pointers

    checkPointerFn(exec, mod, "getVNull",
                   new LLVMGenericPointer(new Ptr(0)));
    checkPointerFn(exec, mod, "getINull",
                   new LLVMGenericPointer(new Ptr(0)));
    checkPointerFn(exec, mod, "getFnNull",
                   new LLVMGenericPointer(new Ptr(0)));

    // identical types

    final LLVMGenericValue init = runFn(exec, mod, "getInit",
                                        new LLVMGenericValue[]{});
    checkPointerFn(exec, mod, "i2i", init, init);
    checkPointerFn(exec, mod, "d2d", init, init);
    checkPointerFn(exec, mod, "e2e", init, init);
    checkPointerFn(exec, mod, "au2au", init, init);
    checkPointerFn(exec, mod, "a12a1", init, init);
    checkPointerFn(exec, mod, "fn2fn", init, init);
    checkPointerFn(exec, mod, "s2s", init, init);
    checkPointerFn(exec, mod, "u2u", init, init);

    checkPointerFn(exec, mod, "get_ci2i", init);

    // void*

    checkPointerFn(exec, mod, "i2v", init, init);
    checkPointerFn(exec, mod, "v2i", init, init);
    checkPointerFn(exec, mod, "v2v", init, init);
    checkPointerFn(exec, mod, "v2s", init, init);
    checkPointerFn(exec, mod, "u2v", init, init);

    checkPointerFn(exec, mod, "get_ci2v", init);
    checkPointerFn(exec, mod, "get_cv2i", init);
    checkPointerFn(exec, mod, "get_cv2v", init);

    // compatible types

    checkPointerFn(exec, mod, "e2i", init, init);
    checkPointerFn(exec, mod, "i2e", init, init);

    checkPointerFn(exec, mod, "eau2iau", init, init);
    checkPointerFn(exec, mod, "iau2ea5", init, init);
    checkPointerFn(exec, mod, "ia32iau", init, init);
    checkPointerFn(exec, mod, "ea72ia7", init, init);

    checkPointerFn(exec, mod, "efn02ifn0", init, init);
    checkPointerFn(exec, mod, "efnv2ifnv", init, init);
    checkPointerFn(exec, mod, "efn12ifn1", init, init);
    checkPointerFn(exec, mod, "efn1var2ifn1var", init, init);
    checkPointerFn(exec, mod, "efn22ifn2", init, init);
    checkPointerFn(exec, mod, "efn2var2ifn2var", init, init);
    checkPointerFn(exec, mod, "ifn02ifn1", init, init);
    checkPointerFn(exec, mod, "ifn12ifn0", init, init);
    checkPointerFn(exec, mod, "ifn02ifn2", init, init);
    checkPointerFn(exec, mod, "ifn22ifn0", init, init);

    exec.dispose();
  }

  /**
   * A few quick checks to be sure the conversion rules for an enum type look
   * like its compatible integer type.
   */
  @Test public void toFromEnum() throws Exception {
    final SrcIntegerType enumType = SrcEnumType.COMPATIBLE_INTEGER_TYPE;
    final BigInteger enumPosMaxPlusOne
      = BigInteger.ONE.shiftLeft((int)enumType.getPosWidth());
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        "enum E {",
        "  A = -0x" + enumPosMaxPlusOne.toString(16) + ",",
        "  B = 0x" + enumPosMaxPlusOne.subtract(BigInteger.ONE).toString(16),
        "};",
        "enum E signedCharToEnum(signed char f) { enum E t = f; return t; }",
        "enum E longToEnum(long f)              { enum E t = f; return t; }",
        "enum E floatToEnum(float f)            { enum E t = f; return t; }",
        "char enumToChar(enum E f)              { char t = f; return t; }",
        "long enumToLong(enum E f)              { long t = f; return t; }",
        "unsigned long enumToUnsignedLong(enum E f) { unsigned long t = f; return t; }",
        "double enumToDouble(enum E f) { double t = f; return t; }",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    checkIntFn(exec, mod, "signedCharToEnum",
               getSignedCharGeneric(-34, ctxt), -34);
    checkIntFn(exec, mod, "longToEnum",
               new LLVMGenericInt(SrcLongType.getLLVMType(ctxt),
                                  enumPosMaxPlusOne, false),
               enumPosMaxPlusOne.negate().longValue());
    checkIntFn(exec, mod, "floatToEnum",
               getGenericReal(SrcFloatType, -1.5, ctxt), -1);
    checkIntFn(exec, mod, "enumToChar",
               new LLVMGenericInt(enumType.getLLVMType(ctxt),
                                  enumPosMaxPlusOne.subtract(BigInteger.ONE),
                                  false),
               -1);
    checkIntFn(exec, mod, "enumToLong", getIntegerGeneric(enumType, -1, ctxt), -1);
    checkIntFn(exec, mod, "enumToUnsignedLong", getIntegerGeneric(enumType, -1, ctxt), -1);
    checkDoubleFn(exec, mod, "enumToDouble", getIntegerGeneric(enumType, -302, ctxt), -302.);
    exec.dispose();
  }

  @Test public void defaultArgumentPromotions() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        "#define N 100",
        "char arr[N];",
        // s*printf and ato* are provided by the native target.
        "int sprintf(char *str, char *format, ...);", // variadic args
        "int snprintf();", // unspecified args
        "int atoi(char *str);",
        "long atol(char *str);",
        "double atof(char *str);",

        // integer promotion: bool
        "int boolVargs(_Bool arg) { sprintf(arr, \"%d\", arg); return atoi(arr); }",
        "int boolNone(_Bool arg) { snprintf(arr, N, \"%d\", arg); return atoi(arr); }",

        // integer promotion: signed and unsigned with rank < int
        "int sCharVargs(signed char arg) { sprintf(arr, \"%d\", arg); return atoi(arr); }",
        "int sCharNone(signed char arg) { snprintf(arr, N, \"%d\", arg); return atoi(arr); }",
        "int uCharVargs(unsigned char arg) { sprintf(arr, \"%d\", arg); return atoi(arr); }",
        "int uCharNone(unsigned char arg) { snprintf(arr, N, \"%d\", arg); return atoi(arr); }",

        // integer promotion: signed and unsigned with rank = int
        "int intVargs(int arg) { sprintf(arr, \"%d\", arg); return atoi(arr); }",
        "int intNone(int arg) { snprintf(arr, N, \"%d\", arg); return atoi(arr); }",
        "long uIntVargs(unsigned arg) { sprintf(arr, \"%u\", arg); return atol(arr); }",
        "long uIntNone(unsigned arg) { snprintf(arr, N, \"%u\", arg); return atol(arr); }",

        // integer promotion: rank > int
        "long longVargs(long arg) { sprintf(arr, \"%ld\", arg); return atol(arr); }",
        "long longNone(long arg) { snprintf(arr, N, \"%ld\", arg); return atol(arr); }",

        // float to double
        "double floatVargs(float arg) { sprintf(arr, \"%e\", arg); return atof(arr); }",
        "double floatNone(float arg) { snprintf(arr, N, \"%e\", arg); return atof(arr); }",

        // double remains the same
        "double doubleVargs(double arg) { sprintf(arr, \"%e\", arg); return atof(arr); }",
        "double doubleNone(double arg) { snprintf(arr, N, \"%e\", arg); return atof(arr); }",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    final LLVMGenericInt boolTrue = getBoolGeneric(1, ctxt);
    final LLVMGenericInt boolFalse = getBoolGeneric(0, ctxt);
    final BigInteger uCharMax
      = BigInteger.ONE.shiftLeft((int)SrcUnsignedCharType.getWidth())
        .subtract(BigInteger.ONE);
    final BigInteger uIntMax
      = BigInteger.ONE.shiftLeft((int)SrcUnsignedIntType.getWidth())
        .subtract(BigInteger.ONE);
    final BigInteger longPosMax
      = BigInteger.ONE.shiftLeft((int)SrcLongType.getPosWidth())
        .subtract(BigInteger.ONE);
    final BigInteger longNegMax
      = BigInteger.ONE.shiftLeft((int)SrcLongType.getPosWidth());

    // integer promotion: bool
    checkIntFn(exec, mod, "boolVargs", boolTrue, 1);
    checkIntFn(exec, mod, "boolVargs", boolFalse, 0);
    checkIntFn(exec, mod, "boolNone", boolTrue, 1);
    checkIntFn(exec, mod, "boolNone", boolFalse, 0);

    // integer promotion: signed and unsigned with rank < int
    checkIntFn(exec, mod, "sCharVargs", getSignedCharGeneric(0, ctxt), 0);
    checkIntFn(exec, mod, "sCharVargs", getSignedCharGeneric(1, ctxt), 1);
    checkIntFn(exec, mod, "sCharVargs", getSignedCharGeneric(-5, ctxt), -5);
    checkIntFn(exec, mod, "sCharNone", getSignedCharGeneric(0, ctxt), 0);
    checkIntFn(exec, mod, "sCharNone", getSignedCharGeneric(50, ctxt), 50);
    checkIntFn(exec, mod, "sCharNone", getSignedCharGeneric(-98, ctxt), -98);
    checkIntFn(exec, mod, "uCharVargs", getUnsignedCharGeneric(0, ctxt), 0);
    checkIntFn(exec, mod, "uCharVargs", getUnsignedCharGeneric(uCharMax.longValue(), ctxt), uCharMax.longValue());
    checkIntFn(exec, mod, "uCharNone", getUnsignedCharGeneric(5, ctxt), 5);
    checkIntFn(exec, mod, "uCharNone", getUnsignedCharGeneric(uCharMax.longValue(), ctxt), uCharMax.longValue());

    // integer promotion: signed and unsigned with rank = int
    checkIntFn(exec, mod, "intVargs", getIntGeneric(0, ctxt), 0);
    checkIntFn(exec, mod, "intVargs", getIntGeneric(-243, ctxt), -243);
    checkIntFn(exec, mod, "intVargs", getIntGeneric(987, ctxt), 987);
    checkIntFn(exec, mod, "intNone", getIntGeneric(5, ctxt), 5);
    checkIntFn(exec, mod, "intNone", getIntGeneric(-1, ctxt), -1);
    checkIntFn(exec, mod, "uIntVargs", getUnsignedIntGeneric(0, ctxt), 0);
    checkIntFn(exec, mod, "uIntVargs", getUnsignedIntGeneric(uIntMax.longValue(), ctxt), uIntMax.longValue());
    checkIntFn(exec, mod, "uIntNone", getUnsignedIntGeneric(874, ctxt), 874);
    checkIntFn(exec, mod, "uIntNone", getUnsignedIntGeneric(uIntMax.longValue(), ctxt), uIntMax.longValue());

    // integer promotion: rank > int
    checkIntFn(exec, mod, "longVargs", getLongGeneric(0, ctxt), 0);
    checkIntFn(exec, mod, "longVargs", getLongGeneric(longPosMax.longValue(), ctxt), longPosMax.longValue());
    checkIntFn(exec, mod, "longVargs", getLongGeneric(longNegMax.longValue(), ctxt), longNegMax.longValue());
    checkIntFn(exec, mod, "longNone", getLongGeneric(0, ctxt), 0);
    checkIntFn(exec, mod, "longNone", getLongGeneric(longPosMax.longValue(), ctxt), longPosMax.longValue());
    checkIntFn(exec, mod, "longNone", getLongGeneric(longNegMax.longValue(), ctxt), longNegMax.longValue());

    // float to double
    checkDoubleFn(exec, mod, "floatVargs", getGenericReal(SrcFloatType, 0.5e10, ctxt), 0.5e10);
    checkDoubleFn(exec, mod, "floatNone", getGenericReal(SrcFloatType, -8.3e-10, ctxt), -8.3e-10);

    // double remains the same
    checkDoubleFn(exec, mod, "doubleVargs", getGenericReal(SrcDoubleType, 389e-7, ctxt), 389e-7);
    checkDoubleFn(exec, mod, "doubleNone", getGenericReal(SrcDoubleType, -39.1e7, ctxt), -39.1e7);

    exec.dispose();
  }

  /**
   * Just a quick check to be sure the conversions are happening for parameter
   * types. Thorough testing of the conversions themselves appears in other
   * tests in this class.
   */
  @Test public void argToParamType() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final BigInteger unsignedMax
      = BigInteger.ONE.shiftLeft((int)SrcUnsignedIntType.getWidth())
        .subtract(BigInteger.ONE);
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        "#include <stddef.h>",
        "#define N 100",
        // sprintf, strtok, and ato* are provided by the native target, but we
        // make the sprintf prototype more specific in order to mix default
        // argument promotions with parameter type conversions.
        "int sprintf(char *str, char *format, int i, ...);",
        "char *strtok(char *str, const char *sep);",
        "int atoi(char *str);",
        "double atof(char *str);",
        "char str[N];",
        // The first float argument to sprintf requires conversion to int,
        // which is not a default argument promotion. The second one requires
        // conversion to double, which is a default argument promotion. Neither
        // conversion is just a reinterpretation of the bits (an instruction
        // must be generated to perform the specific conversion required).
        "void write(float arg) { sprintf(str, \"%d %e\", arg, arg); }",
        "int readInt() { return atoi(strtok(str, \" \")); }",
        "double readDouble() { return atof(strtok(0, \" \")); }",
        // Some other parameter type conversion where there are no other
        // arguments requiring default argument promotions.
        "unsigned unsignedIdentity(unsigned arg) { return arg; }",
        "unsigned sextToUnsigned(signed char arg) { return unsignedIdentity(arg); }",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    runFn(exec, mod, "write", getGenericReal(SrcFloatType, 38.8e-1, ctxt));
    checkIntFn(exec, mod, "readInt", 3);
    checkDoubleFn(exec, mod, "readDouble", 38.8e-1);
    checkIntFn(exec, mod, unsignedMax.longValue(), false, "sextToUnsigned",
               getSignedCharGeneric(-1, ctxt));
    exec.dispose();
  }

  /**
   * We thoroughly check the usual arithmetic conversions only for "{@code *}"
   * because it's simple and because we had it implemented by the time of this
   * writing. Other operators are checked in
   * {@link BuildLLVMTest_OtherExpressions} but we don't repeat the thorough
   * checking of the usual arithmetic conversions.
   */
  @Test public void usualArithmeticConversions() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final BigInteger intMax
      = BigInteger.ONE.shiftLeft((int)SrcIntType.getPosWidth())
        .subtract(BigInteger.ONE);
    final BigInteger longMax
      = BigInteger.ONE.shiftLeft((int)SrcLongType.getPosWidth())
        .subtract(BigInteger.ONE);
    final BigInteger unsignedLongMaxPlusOne
      = BigInteger.ONE.shiftLeft((int)SrcUnsignedLongType.getWidth());
    final BigInteger longLongMax
      = BigInteger.ONE.shiftLeft((int)SrcLongLongType.getPosWidth())
        .subtract(BigInteger.ONE);
    final BigInteger unsignedLongLongMaxPlusOne
      = BigInteger.ONE.shiftLeft((int)SrcUnsignedLongLongType.getWidth());
    final SimpleResult simpleResult = buildLLVMSimple(
      "", TARGET_DATA_LAYOUT,
      new String[]{
        "#include <stddef.h>",

        // result is long double
        genBinaryArith("*", "ld",   "u",   "0.5l",   "3u"),
        genBinaryArith("*", "i",    "ld",  "89",     "-3.2l"),
        genBinaryArith("*", "ld",   "ll",  "5.l",    "82ll"),
        genBinaryArith("*", "ull",  "ld",  "12ull",  ".32L"),
        genBinaryArith("*", "ld",   "f",   "1e10L",  "-5.2f"),
        genBinaryArith("*", "f",    "ld",  "83.2f",  "2e-2l"),
        genBinaryArith("*", "ld",   "d",   "-1e10l", "2.98"),
        genBinaryArith("*", "d",    "ld",  "0.72",   "2e3l"),
        genBinaryArith("*", "ld",   "ld",  "20.l",   "0.3e-8"),

        // result is double
        genBinaryArith("*", "d",   "i",    "0.5",   "3"),
        genBinaryArith("*", "u",   "d",    "89u",   "-3.2"),
        genBinaryArith("*", "d",   "ull",  "5.",    "82ull"),
        genBinaryArith("*", "ll",  "d",    "-12ll", ".32"),
        genBinaryArith("*", "d",   "f",    "1e10",  "-5.2f"),
        genBinaryArith("*", "f",   "d",    "83.2f", "2e-2"),
        genBinaryArith("*", "d",   "d",    "-1e10", "2.98"),

        // result is float
        genBinaryArith("*", "f",   "i",    "0.5f",  "3"),
        genBinaryArith("*", "u",   "f",    "89u",   "-3.2f"),
        genBinaryArith("*", "f",   "ull",  "5.f",   "82ull"),
        genBinaryArith("*", "ll",  "f",    "-12ll", ".32f"),
        genBinaryArith("*", "f",   "f",    "1e10f", "-5.2f"),

        // integer promotions => both int
        genBinaryArith("*", "b",   "b",    "0",     "1"),
        genBinaryArith("*", "b",   "c",    "1",     "5"),
        genBinaryArith("*", "sc",  "b",    "-2",    "1"),
        genBinaryArith("*", "b",   "uc",   "1",     "1"),
        genBinaryArith("*", "sc",  "c",    "3",     "5"),
        genBinaryArith("*", "uc",  "sc",   "2",     "87"),
        genBinaryArith("*", "c",   "uc",   "34",    "1"),
        genBinaryArith("*", "c",   "c",    "92",    "8"),
        genBinaryArith("*", "i",   "b",    "92",    "0"),
        genBinaryArith("*", "c",   "i",    "9",     "-23"),
        genBinaryArith("*", "i",   "i",    "-3",    "2"),

        // integer promotions => both unsigned int
        // (Keep in mind that genBinaryArith casts the values given here to
        // ui=unsigned int, and integer promotions don't change the type.)
        // unsigned int appears to be the only type that will be "promoted" to
        // unsigned int via the integer promotions.
        genBinaryArith("*", "u",   "u",    "2",     intMax.toString()),

        // integer promotions => both (un)signed long (long)
        genBinaryArith("*", "l",   "l",    "5",     "-87"),
        genBinaryArith("*", "ul",  "ul",   "2",     longMax.toString()),
        genBinaryArith("*", "ll",  "ll",   "3",     "-24"),
        genBinaryArith("*", "ull",  "ull", "2",     longMax.toString()),

        // integer promotions => different unsigned types
        genBinaryArith("*", "u",   "ul",   "3",     "5"),
        genBinaryArith("*", "ul",  "u",    longMax.toString(), "2"),
        genBinaryArith("*", "u",   "ull",  "1",     "58"),
        genBinaryArith("*", "ull", "ul",   longLongMax.toString(), "2"),

        // integer promotions => different signed types
        genBinaryArith("*", "i",   "l",    "3",     "-5"),
        genBinaryArith("*", "l",   "uc",   "99",    "2"),
        genBinaryArith("*", "b",   "ll",   "1",     "58"),
        genBinaryArith("*", "ll",  "l",    "-999",  "2"),

        // integer promotions => unsigned type >= signed type
        genBinaryArith("*", "ul",  "i",    longMax.toString(), "2"),
        genBinaryArith("*", "sc",  "ul",   "-5",    "1"),
        genBinaryArith("*", "ul",  "l",    longMax.toString(), "2"),
        genBinaryArith("*", "l",   "ul",   "-2",    "1"),
        genBinaryArith("*", "ull", "i",    "1",     "-1"),
        genBinaryArith("*", "ull", "b",    "5",     "0"),
        genBinaryArith("*", "i",   "ull",  "2",     longLongMax.toString()),
        genBinaryArith("*", "ull", "l",    "1",     "-9"),
        genBinaryArith("*", "l",   "ull",  "2",     longLongMax.toString()),
        genBinaryArith("*", "ull", "ll",   longLongMax.toString(), "2"),
        genBinaryArith("*", "ll",  "ull",  "-3",    "1"),

        // integer promotions => signed type > unsigned type but signed type is enough
        genBinaryArith("*", "l",   "u",    "-2",    "5"),
        genBinaryArith("*", "u",   "ll",   "9",     "-10"),

        // integer promotions => signed type > unsigned type but signed type is not enough
        // This assumes that ll and ul have the same bit width.
        genBinaryArith("*", "ll",  "ul",   "2",     longMax.toString()),
        genBinaryArith("*", "ul",  "ll",   "2",     longLongMax.toString()),
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    final long ldSize = SIZEOF_LONG_DOUBLE;
    final long dSize = SrcDoubleType.getLLVMType(ctxt).getPrimitiveSizeInBits()/8;
    final long fSize = SrcFloatType.getLLVMType(ctxt).getPrimitiveSizeInBits()/8;
    final long iSize = SrcIntType.getLLVMWidth()/8;
    final long uiSize = SrcUnsignedIntType.getLLVMWidth()/8;
    final long lSize = SrcLongType.getLLVMWidth()/8;
    final long ulSize = SrcUnsignedLongType.getLLVMWidth()/8;
    final long llSize = SrcLongLongType.getLLVMWidth()/8;
    final long ullSize = SrcUnsignedLongLongType.getLLVMWidth()/8;

    // result is long double
    checkBinaryArith(exec, mod, "*", "ld",   "u",  ldSize,  1.5);
    checkBinaryArith(exec, mod, "*", "i",    "ld", ldSize,  -284.8);
    checkBinaryArith(exec, mod, "*", "ld",   "ll", ldSize,  410.);
    checkBinaryArith(exec, mod, "*", "ull",  "ld", ldSize,  3.84);
    checkBinaryArith(exec, mod, "*", "ld",   "f",  ldSize,  -5.2e10);
    checkBinaryArith(exec, mod, "*", "f",    "ld", ldSize,  1.664);
    checkBinaryArith(exec, mod, "*", "ld",   "d",  ldSize,  -29.8e9);
    checkBinaryArith(exec, mod, "*", "d",    "ld", ldSize,  1440);
    checkBinaryArith(exec, mod, "*", "ld",   "ld", ldSize,  6e-8);

    // result is double
    checkBinaryArith(exec, mod, "*", "d",   "i",   dSize,   1.5);
    checkBinaryArith(exec, mod, "*", "u",   "d",   dSize,   -284.8);
    checkBinaryArith(exec, mod, "*", "d",   "ull", dSize,   410);
    checkBinaryArith(exec, mod, "*", "ll",  "d",   dSize,   -3.84);
    checkBinaryArith(exec, mod, "*", "d",   "f",   dSize,   -5.2e10);
    checkBinaryArith(exec, mod, "*", "f",   "d",   dSize,   1.664);
    checkBinaryArith(exec, mod, "*", "d",   "d",   dSize,   -2.98e10);

    // result is float
    checkBinaryArith(exec, mod, "*", "f",   "i",   fSize,   1.5);
    checkBinaryArith(exec, mod, "*", "u",   "f",   fSize,   -284.8);
    checkBinaryArith(exec, mod, "*", "f",   "ull", fSize,   410);
    checkBinaryArith(exec, mod, "*", "ll",  "f",   fSize,   -3.84);
    checkBinaryArith(exec, mod, "*", "f",   "f",   fSize,   -5.2e10);

    // integer promotions => both int
    checkBinaryArith(exec, mod, "*", "b",   "b",   iSize,   0);
    checkBinaryArith(exec, mod, "*", "b",   "c",   iSize,   5);
    checkBinaryArith(exec, mod, "*", "sc",  "b",   iSize,   -2);
    checkBinaryArith(exec, mod, "*", "b",   "uc",  iSize,   1);
    checkBinaryArith(exec, mod, "*", "sc",  "c",   iSize,   15);
    checkBinaryArith(exec, mod, "*", "uc",  "sc",  iSize,   174);
    checkBinaryArith(exec, mod, "*", "c",   "uc",  iSize,   34);
    checkBinaryArith(exec, mod, "*", "c",   "c",   iSize,   736);
    checkBinaryArith(exec, mod, "*", "i",   "b",   iSize,   0);
    checkBinaryArith(exec, mod, "*", "c",   "i",   iSize,   -207);
    checkBinaryArith(exec, mod, "*", "i",   "i",   iSize,   -6);

    // integer promotions => both unsigned int
    checkBinaryArith(exec, mod, "*", "u",   "u",   uiSize,  2*intMax.doubleValue());

    // integer promotions => both (un)signed long
    checkBinaryArith(exec, mod, "*", "l",   "l",   lSize,   5*-87);
    checkBinaryArith(exec, mod, "*", "ul",  "ul",  ulSize,  2*longMax.doubleValue());
    checkBinaryArith(exec, mod, "*", "ll",  "ll",  llSize,  3*-24);
    checkBinaryArith(exec, mod, "*", "ull", "ull", ullSize, 2*longMax.doubleValue());

    // integer promotions => different unsigned types
    checkBinaryArith(exec, mod, "*", "u",   "ul",  ulSize,  15);
    checkBinaryArith(exec, mod, "*", "ul",  "u",   ulSize,  2*longMax.doubleValue());
    checkBinaryArith(exec, mod, "*", "u",   "ull", ullSize, 58);
    checkBinaryArith(exec, mod, "*", "ull", "ul",  ullSize, 2*longLongMax.doubleValue());

    // integer promotions => different signed types
    checkBinaryArith(exec, mod, "*", "i",   "l",   ulSize,  -15);
    checkBinaryArith(exec, mod, "*", "l",   "uc",  ulSize,  2*99);
    checkBinaryArith(exec, mod, "*", "b",   "ll",  ullSize,   58);
    checkBinaryArith(exec, mod, "*", "ll",  "l",   ullSize, -999*2);

    // integer promotions => unsigned type > signed type
    checkBinaryArith(exec, mod, "*", "ul",  "i",   ulSize,  2*longMax.doubleValue());
    checkBinaryArith(exec, mod, "*", "sc",  "ul",  ulSize,
                unsignedLongMaxPlusOne.subtract(BigInteger.valueOf(5))
                .doubleValue());
    checkBinaryArith(exec, mod, "*", "ul",  "l",   ulSize,  2*longMax.doubleValue());
    checkBinaryArith(exec, mod, "*", "l",   "ul",  ulSize,
                unsignedLongMaxPlusOne.subtract(BigInteger.valueOf(2))
                .doubleValue());
    checkBinaryArith(exec, mod, "*", "ull", "i",   ullSize,
                unsignedLongLongMaxPlusOne.subtract(BigInteger.valueOf(1))
                .doubleValue());
    checkBinaryArith(exec, mod, "*", "ull", "b",   ullSize, 0);
    checkBinaryArith(exec, mod, "*", "i",   "ull", ullSize, 2*longLongMax.doubleValue());
    checkBinaryArith(exec, mod, "*", "ull", "l",   ullSize,
                unsignedLongLongMaxPlusOne.subtract(BigInteger.valueOf(9))
                .doubleValue());
    checkBinaryArith(exec, mod, "*", "l",   "ull", ullSize, 2*longLongMax.doubleValue());
    checkBinaryArith(exec, mod, "*", "ull", "ll",  ullSize, 2*longLongMax.doubleValue());
    checkBinaryArith(exec, mod, "*", "ll",  "ull", ullSize,
                unsignedLongLongMaxPlusOne.subtract(BigInteger.valueOf(3))
                .doubleValue());

    // integer promotions => signed type > unsigned type but signed type is enough
    checkBinaryArith(exec, mod, "*", "l",   "u",   lSize,   -10);
    checkBinaryArith(exec, mod, "*", "u",   "ll",  llSize,  -90);

    // integer promotions => signed type > unsigned type but signed type is not enough
    checkBinaryArith(exec, mod, "*", "ll",  "ul",  ullSize, 2*longMax.doubleValue());
    checkBinaryArith(exec, mod, "*", "ul",  "ll",  ullSize, 2*longLongMax.doubleValue());

    exec.dispose();
  }
}
