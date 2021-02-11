package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcPrimitiveFloatingType.SrcDoubleType;
import static openacc.codegen.llvmBackend.SrcPrimitiveFloatingType.SrcFloatType;
import static openacc.codegen.llvmBackend.SrcPrimitiveFloatingType.SrcLongDoubleType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SRC_CHAR_CONST_TYPE;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SRC_ENUM_CONST_TYPE;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SRC_WCHAR_TYPE;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcCharType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcIntType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcLongLongType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcLongType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcSignedCharType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedCharType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedIntType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedLongLongType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedLongType;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.math.BigInteger;

import org.jllvm.LLVMConstant;
import org.jllvm.LLVMConstantArray;
import org.jllvm.LLVMConstantInteger;
import org.jllvm.LLVMConstantReal;
import org.jllvm.LLVMContext;
import org.jllvm.LLVMExecutionEngine;
import org.jllvm.LLVMGenericInt;
import org.jllvm.LLVMGenericValue;
import org.jllvm.LLVMGlobalVariable;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMPointerType;
import org.jllvm.LLVMTargetData;
import org.jllvm.bindings.LLVMLinkage;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Checks the ability to build correct LLVM IR for C primary expressions.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class BuildLLVMTest_PrimaryExpressions extends BuildLLVMTest {
  // long double is x86_fp80, which has no default alignment so LLVM will fail
  // an assertion if we try to compute its size without this.
  private static final String TARGET_DATA_LAYOUT = "f80:128";
  private static final int SIZEOF_LONG_DOUBLE = 16; // = 128/8
      
  private static class Expect {
    public Expect(String name, LLVMConstant init) {
      this.name = name;
      this.init = init;
    }
    public final String name;
    public final LLVMConstant init;
    public LLVMGlobalVariable check(LLVMModule llvmModule) {
      LLVMGlobalVariable global = llvmModule.getNamedGlobal(name);
      assertNotNull(name + " must exist as a global variable",
                    global.getInstance());
      assertEquals(name + "'s type", LLVMPointerType.get(init.typeOf(), 0),
                   global.typeOf());
      assertEquals(name + "'s initializer", init, global.getInitializer());
      return global;
    }
    public void checkStr(LLVMModule llvmModule) {
      LLVMGlobalVariable global = check(llvmModule);
      assertEquals(name + "'s linkage", LLVMLinkage.LLVMPrivateLinkage,
                   global.getLinkage());
      assertTrue(name + " must be constant", global.isConstant());
    }
  }

  private SimpleResult checkDecls(String[] decls) throws IOException {
    String[] src = new String[decls.length + 1];
    System.arraycopy(decls, 0, src, 0, decls.length);
    src[decls.length] = "int main() { return 0; }";
    return buildLLVMSimple("", TARGET_DATA_LAYOUT, src);
  }

  private void checkDecls(LLVMModule llvmModule, Expect[] expects)
    throws IOException
  {
    for (Expect expect : expects)
      expect.check(llvmModule);
    int nGlobals = 0;
    for (LLVMGlobalVariable global = llvmModule.getFirstGlobal();
         global.getInstance() != null;
         global = global.getNextGlobal())
      ++nGlobals;
    assertEquals("global declaration count", expects.length, nGlobals);
  }

  @BeforeClass public static void setup() {
    // Normally the BuildLLVM pass loads the jllvm native library. However,
    // these test methods use jllvm before running the BuildLLVM pass. In case
    // these test methods are run before any other test methods that run the
    // BuildLLVM pass, we must load the jllvm native library first.
    System.loadLibrary("jllvm");
  }

  @Test public void floatLiteral() throws IOException {
    final SimpleResult simpleResult = checkDecls(
      new String[]{
        "#include <stddef.h>",

        "#define F0 0.f",
        "float f0 = F0;",
        "size_t f0s = sizeof F0;",

        "#define F1 .5e-3f",
        "float f1 = F1;",
        "size_t f1s = sizeof F1;",

        "#define F2 30e+20f",
        "float f2 = F2;",
        "size_t f2s = sizeof F2;",

        "#define F3 1e02f",
        "float f3 = F3;",
        "size_t f3s = sizeof F3;",

        "#define F4 0.3f",
        "float f4 = F4;",
        "size_t f4s = sizeof F4;",

        "#define D0 0.e0",
        "double d0 = D0;",
        "size_t d0s = sizeof D0;",

        "#define D1 12.",
        "double d1 = D1;",
        "size_t d1s = sizeof D1;",

        "#define D2 1e200",
        "double d2 = D2;",
        "size_t d2s = sizeof D2;",

        "#define LD0 .0l",
        "long double ld0 = LD0;",
        "size_t ld0s = sizeof LD0;",

        "#define LD1 030.010e300l",
        "long double ld1 = LD1;",
        "size_t ld1s = sizeof LD1;",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMConstantInteger float_nbytes
      = getSizeTInteger(
          SrcFloatType.getLLVMType(ctxt).getPrimitiveSizeInBits()/8, ctxt);
    final LLVMConstantInteger double_nbytes
      = getSizeTInteger(
          SrcDoubleType.getLLVMType(ctxt).getPrimitiveSizeInBits()/8, ctxt);
    final LLVMConstantInteger long_double_nbytes
      = getSizeTInteger(SIZEOF_LONG_DOUBLE, ctxt);
    checkDecls(mod,
      new Expect[]{
        new Expect("f0", LLVMConstantReal.get(SrcFloatType.getLLVMType(ctxt),
                                              0.)),
        new Expect("f0s", float_nbytes),

        new Expect("f1", LLVMConstantReal.get(SrcFloatType.getLLVMType(ctxt),
                                              0.5e-3)),
        new Expect("f1s", float_nbytes),

        new Expect("f2", LLVMConstantReal.get(SrcFloatType.getLLVMType(ctxt),
                                              30e+20)),
        new Expect("f2s", float_nbytes),

        new Expect("f3", LLVMConstantReal.get(SrcFloatType.getLLVMType(ctxt),
                                              100)),
        new Expect("f3s", float_nbytes),

        new Expect("f4", LLVMConstantReal.get(SrcFloatType.getLLVMType(ctxt),
                                              0.3)),
        new Expect("f4s", float_nbytes),

        new Expect("d0", LLVMConstantReal.get(SrcDoubleType.getLLVMType(ctxt),
                                              0.)),
        new Expect("d0s", double_nbytes),

        new Expect("d1", LLVMConstantReal.get(SrcDoubleType.getLLVMType(ctxt),
                                              12)),
        new Expect("d1s", double_nbytes),

        new Expect("d2", LLVMConstantReal.get(SrcDoubleType.getLLVMType(ctxt),
                                              1e200)),
        new Expect("d2s", double_nbytes),

        new Expect("ld0", LLVMConstantReal.get(
                            SrcLongDoubleType.getLLVMType(ctxt), 0.)),
        new Expect("ld0s", long_double_nbytes),

        new Expect("ld1", LLVMConstantReal.get(
                            SrcLongDoubleType.getLLVMType(ctxt), 30.01e300)),
        new Expect("ld1s", long_double_nbytes),
      });
  }

  /**
   * TODO: Cetus and jllvm use Java's long type to hold integer constants, so
   * they don't handle integer literals larger than 63 bits. Currently, Cetus
   * mistakenly parses an integer literal that's larger than 63 bits as a
   * floating literal. Moreover, this test calls {@link BigInteger#longValue}
   * in order to pass values to jllvm. To fix all that, we'll need to extend
   * Cetus and jllvm to handle arbitrary precision (probably using
   * {@link BigInteger}) and adjust this test case to avoid any conversion to
   * Java long.
   * 
   * For now, this test attempts to avoid testing integer literals that
   * require more than 63 bits. However, to do so, it assumes that maximum
   * values for unsigned int and positive signed long require no more than 63
   * bits, but that might not be true on all platforms. Moreover, it does not
   * exercise boundary conditions among signed long, unsigned long, signed
   * long long, and unsigned long long.
   * 
   * See {@link BuildLLVM.Visitor.visit(IntegerLiteral)} for a related bug in
   * the {@link BuildLLVM} implementation.
   */
  @Test public void integerLiteral() throws IOException {
    StringBuilder strBuilder = new StringBuilder();
    for (int i = 0; i < SrcIntType.getLLVMWidth()/8; ++i)
      strBuilder.append("ff");
    final BigInteger uiMax = new BigInteger(strBuilder.toString(), 16);
    strBuilder.setCharAt(0, '7');
    final BigInteger iMax = new BigInteger(strBuilder.toString(), 16);

    strBuilder.setLength(0);
    for (int i = 0; i < SrcLongType.getLLVMWidth()/8; ++i)
      strBuilder.append("ff");
    // final BigInteger ulMax = new BigInteger(strBuilder.toString(), 16);
    strBuilder.setCharAt(0, '7');
    final BigInteger lMax = new BigInteger(strBuilder.toString(), 16);

    final SimpleResult simpleResult = checkDecls(
      new String[]{
        "#include <stddef.h>",

        // signed int constants

        "#define I_MIN_DEC 0",
        "int i_min_dec = I_MIN_DEC;",
        "size_t i_min_dec_s = sizeof I_MIN_DEC;",
        "#define I_MIN_OCT 00",
        "int i_min_oct = I_MIN_OCT;",
        "size_t i_min_oct_s = sizeof I_MIN_OCT;",
        "#define I_MIN_HEX 0x0",
        "int i_min_hex = I_MIN_HEX;",
        "size_t i_min_hex_s = sizeof I_MIN_HEX;",

        "#define I_MID_DEC 23",
        "int i_mid_dec = I_MID_DEC;",
        "size_t i_mid_dec_s = sizeof I_MID_DEC;",
        "#define I_MID_OCT 040",
        "int i_mid_oct = I_MID_OCT;",
        "size_t i_mid_oct_s = sizeof I_MID_OCT;",
        "#define I_MID_HEX 0x31",
        "int i_mid_hex = I_MID_HEX;",
        "size_t i_mid_hex_s = sizeof I_MID_HEX;",

        "#define I_MAX_DEC " + iMax.toString(10),
        "int i_max_dec = I_MAX_DEC;",
        "size_t i_max_dec_s = sizeof I_MAX_DEC;",
        "#define I_MAX_OCT 0" + iMax.toString(8),
        "int i_max_oct = I_MAX_OCT;",
        "size_t i_max_oct_s = sizeof I_MAX_OCT;",
        "#define I_MAX_HEX 0x" + iMax.toString(16),
        "int i_max_hex = I_MAX_HEX;",
        "size_t i_max_hex_s = sizeof I_MAX_HEX;",

        // unsigned int constants
        // For iMax < x <= uiMax without the "u" suffix, both octal and hex are
        // unsigned, but decimal would become long.

        "#define UI_MIN_DEC 0u",
        "unsigned int ui_min_dec = UI_MIN_DEC;",
        "size_t ui_min_dec_s = sizeof UI_MIN_DEC;",
        "#define UI_MIN_OCT 00u",
        "unsigned int ui_min_oct = UI_MIN_OCT;",
        "size_t ui_min_oct_s = sizeof UI_MIN_OCT;",
        "#define UI_MIN_HEX 0x0u",
        "unsigned int ui_min_hex = UI_MIN_HEX;",
        "size_t ui_min_hex_s = sizeof UI_MIN_HEX;",

        "#define UI_MID_DEC " + iMax.add(BigInteger.ONE).toString(10) + "u",
        "unsigned int ui_mid_dec = UI_MID_DEC;",
        "size_t ui_mid_dec_s = sizeof UI_MID_DEC;",
        "#define UI_MID_OCT 0" + iMax.add(BigInteger.ONE).toString(8) + "u",
        "unsigned int ui_mid_oct = UI_MID_OCT;",
        "size_t ui_mid_oct_s = sizeof UI_MID_OCT;",
        "#define UI_MID_HEX 0x" + iMax.add(BigInteger.ONE).toString(16),
        "unsigned int ui_mid_hex = UI_MID_HEX;",
        "size_t ui_mid_hex_s = sizeof UI_MID_HEX;",

        "#define UI_MAX_DEC " + uiMax.toString(10) + "u",
        "unsigned int ui_max_dec = UI_MAX_DEC;",
        "size_t ui_max_dec_s = sizeof UI_MAX_DEC;",
        "#define UI_MAX_OCT 0" + uiMax.toString(8),
        "unsigned int ui_max_oct = UI_MAX_OCT;",
        "size_t ui_max_oct_s = sizeof UI_MAX_OCT;",
        "#define UI_MAX_HEX 0x" + uiMax.toString(16) + "u",
        "unsigned int ui_max_hex = UI_MAX_HEX;",
        "size_t ui_max_hex_s = sizeof UI_MAX_HEX;",

        // lower bounds where int constant becomes long

        "#define L_FROM_I_DEC " + iMax.add(BigInteger.ONE).toString(10),
        "long l_from_i_dec = L_FROM_I_DEC;",
        "size_t l_from_i_dec_s = sizeof L_FROM_I_DEC;",
        "#define L_FROM_I_OCT 0" + uiMax.add(BigInteger.ONE).toString(8),
        "long l_from_i_oct = L_FROM_I_OCT;",
        "size_t l_from_i_oct_s = sizeof L_FROM_I_OCT;",
        "#define L_FROM_I_HEX 0x" + uiMax.add(BigInteger.ONE).toString(16),
        "long l_from_i_hex = L_FROM_I_HEX;",
        "size_t l_from_i_hex_s = sizeof L_FROM_I_HEX;",

        "#define L_FROM_UI_DEC " + uiMax.add(BigInteger.ONE).toString(10) + "u",
        "unsigned long l_from_ui_dec = L_FROM_UI_DEC;",
        "size_t l_from_ui_dec_s = sizeof L_FROM_UI_DEC;",
        "#define L_FROM_UI_OCT 0" + uiMax.add(BigInteger.ONE).toString(8) + "u",
        "unsigned long l_from_ui_oct = L_FROM_UI_OCT;",
        "size_t l_from_ui_oct_s = sizeof L_FROM_UI_OCT;",
        "#define L_FROM_UI_HEX 0x" + uiMax.add(BigInteger.ONE).toString(16) + "u",
        "unsigned long l_from_ui_hex = L_FROM_UI_HEX;",
        "size_t l_from_ui_hex_s = sizeof L_FROM_UI_HEX;",

        // signed long constants

        "#define L_MAX_DEC " + lMax.toString(10) + "l",
        "long l_max_dec = L_MAX_DEC;",
        "size_t l_max_dec_s = sizeof L_MAX_DEC;",
        "#define L_MAX_OCT 0" + lMax.toString(8) + "l",
        "long l_max_oct = L_MAX_OCT;",
        "size_t l_max_oct_s = sizeof L_MAX_OCT;",
        "#define L_MAX_HEX 0x" + lMax.toString(16) + "l",
        "long l_max_hex = L_MAX_HEX;",
        "size_t l_max_hex_s = sizeof L_MAX_HEX;",

        // unsigned long constants

        "#define UL_MID_DEC " + lMax.toString(10) + "ul",
        "unsigned long ul_mid_dec = UL_MID_DEC;",
        "size_t ul_mid_dec_s = sizeof UL_MID_DEC;",
        "#define UL_MID_OCT 0" + lMax.toString(8) + "ul",
        "unsigned long ul_mid_oct = UL_MID_OCT;",
        "size_t ul_mid_oct_s = sizeof UL_MID_OCT;",
        "#define UL_MID_HEX 0x" + lMax.toString(16) + "ul",
        "unsigned long ul_mid_hex = UL_MID_HEX;",
        "size_t ul_mid_hex_s = sizeof UL_MID_HEX;",

        // signed long long constants

        "#define LL_MAX_DEC " + lMax.toString(10) + "ll",
        "long long ll_max_dec = LL_MAX_DEC;",
        "size_t ll_max_dec_s = sizeof LL_MAX_DEC;",
        "#define LL_MAX_OCT 0" + lMax.toString(8) + "ll",
        "long long ll_max_oct = LL_MAX_OCT;",
        "size_t ll_max_oct_s = sizeof LL_MAX_OCT;",
        "#define LL_MAX_HEX 0x" + lMax.toString(16) + "ll",
        "long long ll_max_hex = LL_MAX_HEX;",
        "size_t ll_max_hex_s = sizeof LL_MAX_HEX;",

        // unsigned long long constants

        "#define ULL_MID_DEC " + lMax.toString(10) + "ull",
        "unsigned long long ull_mid_dec = ULL_MID_DEC;",
        "size_t ull_mid_dec_s = sizeof ULL_MID_DEC;",
        "#define ULL_MID_OCT 0" + lMax.toString(8) + "ull",
        "unsigned long long ull_mid_oct = ULL_MID_OCT;",
        "size_t ull_mid_oct_s = sizeof ULL_MID_OCT;",
        "#define ULL_MID_HEX 0x" + lMax.toString(16) + "ull",
        "unsigned long long ull_mid_hex = ULL_MID_HEX;",
        "size_t ull_mid_hex_s = sizeof ULL_MID_HEX;",
      });

    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMConstantInteger int_nbytes
      = getSizeTInteger(SrcIntType.getLLVMWidth()/8, ctxt);
    final LLVMConstantInteger long_nbytes
      = getSizeTInteger(SrcLongType.getLLVMWidth()/8, ctxt);
    final LLVMConstantInteger long_long_nbytes
      = getSizeTInteger(SrcLongLongType.getLLVMWidth()/8, ctxt);

    checkDecls(mod,
      new Expect[]{

        // signed int constants

        new Expect("i_min_dec",
                   LLVMConstantInteger.get(SrcIntType.getLLVMType(ctxt), 0,
                                           false)),
        new Expect("i_min_dec_s", int_nbytes),
        new Expect("i_min_oct",
                   LLVMConstantInteger.get(SrcIntType.getLLVMType(ctxt), 0,
                                           false)),
        new Expect("i_min_oct_s", int_nbytes),
        new Expect("i_min_hex",
                   LLVMConstantInteger.get(SrcIntType.getLLVMType(ctxt), 0,
                                           false)),
        new Expect("i_min_hex_s", int_nbytes),

        new Expect("i_mid_dec",
                   LLVMConstantInteger.get(SrcIntType.getLLVMType(ctxt), 23,
                                           false)),
        new Expect("i_mid_dec_s", int_nbytes),
        new Expect("i_mid_oct",
                   LLVMConstantInteger.get(SrcIntType.getLLVMType(ctxt), 040,
                                           false)),
        new Expect("i_mid_oct_s", int_nbytes),
        new Expect("i_mid_hex",
                   LLVMConstantInteger.get(SrcIntType.getLLVMType(ctxt), 0x31,
                                           false)),
        new Expect("i_mid_hex_s", int_nbytes),

        new Expect("i_max_dec",
                   LLVMConstantInteger.get(SrcIntType.getLLVMType(ctxt),
                                           iMax.longValue(), false)),
        new Expect("i_max_dec_s", int_nbytes),
        new Expect("i_max_oct",
                   LLVMConstantInteger.get(SrcIntType.getLLVMType(ctxt),
                                           iMax.longValue(), false)),
        new Expect("i_max_oct_s", int_nbytes),
        new Expect("i_max_hex",
                   LLVMConstantInteger.get(SrcIntType.getLLVMType(ctxt),
                                           iMax.longValue(), false)),
        new Expect("i_max_hex_s", int_nbytes),

        // unsigned int constants

        new Expect("ui_min_dec",
                   LLVMConstantInteger.get(
                     SrcUnsignedIntType.getLLVMType(ctxt), 0, false)),
        new Expect("ui_min_dec_s", int_nbytes),
        new Expect("ui_min_oct",
                   LLVMConstantInteger.get(
                     SrcUnsignedIntType.getLLVMType(ctxt), 0, false)),
        new Expect("ui_min_oct_s", int_nbytes),
        new Expect("ui_min_hex",
                   LLVMConstantInteger.get(
                     SrcUnsignedIntType.getLLVMType(ctxt), 0, false)),
        new Expect("ui_min_hex_s", int_nbytes),

        new Expect("ui_mid_dec",
                   LLVMConstantInteger.get(SrcUnsignedIntType.getLLVMType(ctxt),
                                           iMax.add(BigInteger.ONE).longValue(),
                                           false)),
        new Expect("ui_mid_dec_s", int_nbytes),
        new Expect("ui_mid_oct",
                   LLVMConstantInteger.get(SrcUnsignedIntType.getLLVMType(ctxt),
                                           iMax.add(BigInteger.ONE).longValue(),
                                           false)),
        new Expect("ui_mid_oct_s", int_nbytes),
        new Expect("ui_mid_hex",
                   LLVMConstantInteger.get(SrcUnsignedIntType.getLLVMType(ctxt),
                                           iMax.add(BigInteger.ONE).longValue(),
                                           false)),
        new Expect("ui_mid_hex_s", int_nbytes),

        new Expect("ui_max_dec",
                   LLVMConstantInteger.get(SrcUnsignedIntType.getLLVMType(ctxt),
                                           uiMax.longValue(), false)),
        new Expect("ui_max_dec_s", int_nbytes),
        new Expect("ui_max_oct",
                   LLVMConstantInteger.get(SrcUnsignedIntType.getLLVMType(ctxt),
                                           uiMax.longValue(), false)),
        new Expect("ui_max_oct_s", int_nbytes),
        new Expect("ui_max_hex",
                   LLVMConstantInteger.get(SrcUnsignedIntType.getLLVMType(ctxt),
                                           uiMax.longValue(), false)),
        new Expect("ui_max_hex_s", int_nbytes),

        // lower bounds where int constant becomes long

        new Expect("l_from_i_dec",
                   LLVMConstantInteger.get(SrcLongType.getLLVMType(ctxt),
                                           iMax.add(BigInteger.ONE).longValue(),
                                           false)),
        new Expect("l_from_i_dec_s", long_nbytes),
        new Expect("l_from_i_oct",
                   LLVMConstantInteger.get(SrcLongType.getLLVMType(ctxt),
                                           uiMax.add(BigInteger.ONE).longValue(),
                                           false)),
        new Expect("l_from_i_oct_s", long_nbytes),
        new Expect("l_from_i_hex",
                   LLVMConstantInteger.get(SrcLongType.getLLVMType(ctxt),
                                           uiMax.add(BigInteger.ONE).longValue(),
                                           false)),
        new Expect("l_from_i_hex_s", long_nbytes),

        new Expect("l_from_ui_dec",
                   LLVMConstantInteger.get(SrcLongType.getLLVMType(ctxt),
                                           uiMax.add(BigInteger.ONE).longValue(),
                                           false)),
        new Expect("l_from_ui_dec_s", long_nbytes),
        new Expect("l_from_ui_oct",
                   LLVMConstantInteger.get(SrcLongType.getLLVMType(ctxt),
                                           uiMax.add(BigInteger.ONE).longValue(),
                                           false)),
        new Expect("l_from_ui_oct_s", long_nbytes),
        new Expect("l_from_ui_hex",
                   LLVMConstantInteger.get(SrcLongType.getLLVMType(ctxt),
                                           uiMax.add(BigInteger.ONE).longValue(),
                                           false)),
        new Expect("l_from_ui_hex_s", long_nbytes),

        // signed long constants

        new Expect("l_max_dec",
                   LLVMConstantInteger.get(SrcLongType.getLLVMType(ctxt),
                                           lMax.longValue(), false)),
        new Expect("l_max_dec_s", long_nbytes),
        new Expect("l_max_oct",
                   LLVMConstantInteger.get(SrcLongType.getLLVMType(ctxt),
                                           lMax.longValue(), false)),
        new Expect("l_max_oct_s", long_nbytes),
        new Expect("l_max_hex",
                   LLVMConstantInteger.get(SrcLongType.getLLVMType(ctxt),
                                           lMax.longValue(), false)),
        new Expect("l_max_hex_s", long_nbytes),

        // unsigned long constants

        new Expect("ul_mid_dec",
                   LLVMConstantInteger.get(SrcUnsignedLongType.getLLVMType(ctxt),
                                           lMax.longValue(), false)),
        new Expect("ul_mid_dec_s", long_nbytes),
        new Expect("ul_mid_oct",
                   LLVMConstantInteger.get(SrcUnsignedLongType.getLLVMType(ctxt),
                                           lMax.longValue(), false)),
        new Expect("ul_mid_oct_s", long_nbytes),
        new Expect("ul_mid_hex",
                   LLVMConstantInteger.get(SrcUnsignedLongType.getLLVMType(ctxt),
                                           lMax.longValue(), false)),
        new Expect("ul_mid_hex_s", long_nbytes),

        // signed long long constants

        new Expect("ll_max_dec",
                   LLVMConstantInteger.get(SrcLongLongType.getLLVMType(ctxt),
                                           lMax.longValue(), false)),
        new Expect("ll_max_dec_s", long_long_nbytes),
        new Expect("ll_max_oct",
                   LLVMConstantInteger.get(SrcLongLongType.getLLVMType(ctxt),
                                           lMax.longValue(), false)),
        new Expect("ll_max_oct_s", long_long_nbytes),
        new Expect("ll_max_hex",
                   LLVMConstantInteger.get(SrcLongLongType.getLLVMType(ctxt),
                                           lMax.longValue(), false)),
        new Expect("ll_max_hex_s", long_long_nbytes),

        // unsigned long long constants

        new Expect("ull_mid_dec",
                   LLVMConstantInteger.get(
                     SrcUnsignedLongLongType.getLLVMType(ctxt),
                     lMax.longValue(), false)),
        new Expect("ull_mid_dec_s", long_long_nbytes),
        new Expect("ull_mid_oct",
                   LLVMConstantInteger.get(
                     SrcUnsignedLongLongType.getLLVMType(ctxt),
                     lMax.longValue(), false)),
        new Expect("ull_mid_oct_s", long_long_nbytes),
        new Expect("ull_mid_hex",
                   LLVMConstantInteger.get(
                     SrcUnsignedLongLongType.getLLVMType(ctxt),
                     lMax.longValue(), false)),
        new Expect("ull_mid_hex_s", long_long_nbytes),
      });
  }

  @Test public void charLiteral() throws IOException {
    StringBuilder strBuilder = new StringBuilder();
    for (int i = 0; i < SrcCharType.getLLVMWidth()/8; ++i)
      strBuilder.append("ff");
    final BigInteger max = new BigInteger(strBuilder.toString(), 16);
    strBuilder.setLength(0);
    for (int i = 0; i < SRC_WCHAR_TYPE.getWidth()/8; ++i)
      strBuilder.append("ff");
    final BigInteger wmax = new BigInteger(strBuilder.toString(), 16);

    final SimpleResult simpleResult = checkDecls(
      new String[]{
        "#include <stddef.h>",

        "#define C_MIN '\\0'",
        "int c_min = C_MIN;",
        "size_t c_min_s = sizeof C_MIN;",

        "#define C_MID '*'", // no escape
        "int c_mid = C_MID;",
        "size_t c_mid_s = sizeof C_MID;",

        "#define C_MAX '\\x" + max.toString(16) + "'",
        "int c_max = C_MAX;",
        "size_t c_max_s = sizeof C_MAX;",

        "#define C_NL '\\n'",
        "int c_nl = C_NL;",
        "size_t c_nl_s = sizeof C_NL;",

        "#define WC_MIN L'\\0'",
        "wchar_t wc_min = WC_MIN;",
        "size_t wc_min_s = sizeof WC_MIN;",

        "#define WC_MID L'5'", // no escape
        "wchar_t wc_mid = WC_MID;",
        "size_t wc_mid_s = sizeof WC_MID;",

        "#define WC_MAX L'\\x" + wmax.toString(16) + "'",
        "wchar_t wc_max = WC_MAX;",
        "size_t wc_max_s = sizeof WC_MAX;",
      });

    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMConstantInteger charConst_nbytes
      = getSizeTInteger(SRC_CHAR_CONST_TYPE.getLLVMWidth()/8, ctxt);
    final LLVMConstantInteger wchar_nbytes
      = getSizeTInteger(SRC_WCHAR_TYPE.getLLVMWidth()/8, ctxt);

    checkDecls(mod,
      new Expect[]{
        new Expect("c_min",
                   LLVMConstantInteger.get(SrcIntType.getLLVMType(ctxt), 0,
                                           false)),
        new Expect("c_min_s", charConst_nbytes),

        new Expect("c_mid",
                   LLVMConstantInteger.get(SrcIntType.getLLVMType(ctxt), '*',
                                           false)),
        new Expect("c_mid_s", charConst_nbytes),

        new Expect("c_max",
                   LLVMConstantInteger.get(SrcIntType.getLLVMType(ctxt),
                                           SrcCharType.isSigned()
                                           ? -1 : max.longValue(),
                                           true)),
        new Expect("c_max_s", charConst_nbytes),

        new Expect("c_nl",
                   LLVMConstantInteger.get(SrcIntType.getLLVMType(ctxt), '\n',
                                           false)),
        new Expect("c_nl_s", charConst_nbytes),

        new Expect("wc_min",
                   LLVMConstantInteger.get(SRC_WCHAR_TYPE.getLLVMType(ctxt), 0,
                                           false)),
        new Expect("wc_min_s", wchar_nbytes),

        new Expect("wc_mid",
                   LLVMConstantInteger.get(SRC_WCHAR_TYPE.getLLVMType(ctxt), '5',
                                           false)),
        new Expect("wc_mid_s", wchar_nbytes),

        new Expect("wc_max",
                   LLVMConstantInteger.get(SRC_WCHAR_TYPE.getLLVMType(ctxt),
                                           wmax.longValue(), false)),
        new Expect("wc_max_s", wchar_nbytes),
      });
  }

  private LLVMConstant getConstIntArray(LLVMContext ctxt,
                                        SrcIntegerType elementType,
                                        long... values)
  {
    LLVMConstantInteger[] elements = new LLVMConstantInteger[values.length];
    for (int i = 0; i < elements.length; ++i)
      elements[i] = LLVMConstantInteger.get(elementType.getLLVMType(ctxt),
                                            values[i], elementType.isSigned());
    return LLVMConstantArray.get(elementType.getLLVMType(ctxt), elements);
  }
  private LLVMConstant getConstCharArray(LLVMContext ctxt, long... values) {
    return getConstIntArray(ctxt, SrcCharType, values);
  }
  private LLVMConstant getConstWcharArray(LLVMContext ctxt, long... values) {
    return getConstIntArray(ctxt, SRC_WCHAR_TYPE, values);
  }
  @Test public void stringLiteral() throws IOException {
    final long char_nbytes = SrcCharType.getLLVMWidth()/8;
    final long wchar_nbytes = SRC_WCHAR_TYPE.getLLVMWidth()/8;

    StringBuilder strBuilder = new StringBuilder();
    for (int i = 0; i < SrcCharType.getLLVMWidth()/8; ++i)
      strBuilder.append("ff");
    final BigInteger max = new BigInteger(strBuilder.toString(), 16);
    strBuilder.setLength(0);
    for (int i = 0; i < SRC_WCHAR_TYPE.getLLVMWidth()/8; ++i)
      strBuilder.append("ff");
    final BigInteger wmax = new BigInteger(strBuilder.toString(), 16);

    final SimpleResult simpleResult = checkDecls(
      new String[]{
        "#include <stddef.h>",

        "#define C_EMPTY \"\"",
        "char c_empty[1] = C_EMPTY;",
        "size_t c_empty_s = sizeof C_EMPTY;",

        "#define C_SIMPLE \"*\"",
        "char c_simple[2] = C_SIMPLE;",
        "size_t c_simple_s = sizeof C_SIMPLE;",

        "#define C_MAX \"\\x" + max.toString(16) + "\"",
        "char c_max[2] = C_MAX;",
        "size_t c_max_s = sizeof C_MAX;",

        "#define C_ADJACENT0 \"a\" \"bc\"",
        "char c_adjacent0[4] = C_ADJACENT0;",
        "size_t c_adjacent0_s = sizeof C_ADJACENT0;",

        "#define C_ADJACENT1 \"ab\" \"c\" \"def\"",
        "char c_adjacent1[7] = C_ADJACENT1;",
        "size_t c_adjacent1_s = sizeof C_ADJACENT1;",

        "#define WC_EMPTY L\"\"",
        "wchar_t wc_empty[1] = WC_EMPTY;",
        "size_t wc_empty_s = sizeof WC_EMPTY;",

        "#define WC_SIMPLE L\"A\"",
        "wchar_t wc_simple[2] = WC_SIMPLE;",
        "size_t wc_simple_s = sizeof WC_SIMPLE;",

        "#define WC_MAX L\"\\x" + wmax.toString(16) + "\"",
        "wchar_t wc_max[2] = WC_MAX;",
        "size_t wc_max_s = sizeof WC_MAX;",

        "#define WC_ADJACENT0 L\"a\" \"b\"",
        "wchar_t wc_adjacent0[3] = WC_ADJACENT0;",
        "size_t wc_adjacent0_s = sizeof WC_ADJACENT0;",

        "#define WC_ADJACENT1 \"a\" L\"b\"",
        "wchar_t wc_adjacent1[3] = WC_ADJACENT1;",
        "size_t wc_adjacent1_s = sizeof WC_ADJACENT1;",

        "#define WC_ADJACENT2 L\"a\" L\"b\"",
        "wchar_t wc_adjacent2[3] = WC_ADJACENT2;",
        "size_t wc_adjacent2_s = sizeof WC_ADJACENT2;",

        "#define HEX_TERM1 \"\\x0\"",
        "char hex_term1[2] = HEX_TERM1;",
        "size_t hex_term1_s = sizeof HEX_TERM1;",

        "#define HEX_TERM2 \"\\x1a\"",
        "char hex_term2[2] = HEX_TERM2;",
        "size_t hex_term2_s = sizeof HEX_TERM2;",

        "#define HEX_TERM3 \"\\xbdc\"",
        "char hex_term3[2] = HEX_TERM3;",
        "size_t hex_term3_s = sizeof HEX_TERM3;",

        "#define HEX_TERM4 \"\\xf2AeBC\"",
        "char hex_term4[2] = HEX_TERM4;",
        "size_t hex_term4_s = sizeof HEX_TERM4;",

        "#define HEX_TERM5 \"\\x34*\"",
        "char hex_term5[3] = HEX_TERM5;",
        "size_t hex_term5_s = sizeof HEX_TERM5;",

        "#define HEX_TERM6 \"\\x56789DEF\\\\\"",
        "char hex_term6[3] = HEX_TERM6;",
        "size_t hex_term6_s = sizeof HEX_TERM6;",

        "#define OCTAL_TERM1 \"\\1\"",
        "char octal_term1[2] = OCTAL_TERM1;",
        "size_t octal_term1_s = sizeof OCTAL_TERM1;",

        "#define OCTAL_TERM2 \"\\02\"",
        "char octal_term2[2] = OCTAL_TERM2;",
        "size_t octal_term2_s = sizeof OCTAL_TERM2;",

        "#define OCTAL_TERM3 \"\\435\"",
        "char octal_term3[2] = OCTAL_TERM3;",
        "size_t octal_term3_s = sizeof OCTAL_TERM3;",

        "#define OCTAL_TERM4 \"\\6578\"",
        "char octal_term4[3] = OCTAL_TERM4;",
        "size_t octal_term4_s = sizeof OCTAL_TERM4;",

        "#define OCTAL_TERM5 \"\\32\\x5\"",
        "char octal_term5[3] = OCTAL_TERM5;",
        "size_t octal_term5_s = sizeof OCTAL_TERM5;",

        "#define OCTAL_TERM6 \"x\\0x\"",
        "char octal_term6[4] = OCTAL_TERM6;",
        "size_t octal_term6_s = sizeof OCTAL_TERM6;",

        "#define LONG_BY1 \"abc\"",
        "char long_by1[3] = LONG_BY1;",
        "size_t long_by1_s = sizeof LONG_BY1;",

        "#define LONG_BY2 L\"abc\"",
        "wchar_t long_by2[2] = LONG_BY2;",
        "size_t long_by2_s = sizeof LONG_BY2;",

        "#define SHORT_BY1 L\"\"",
        "wchar_t short_by1[2] = SHORT_BY1;",
        "size_t short_by1_s = sizeof SHORT_BY1;",

        "#define SHORT_BY2 \"ab\"",
        "char short_by2[5] = SHORT_BY2;",
        "size_t short_by2_s = sizeof SHORT_BY2;",

        "#define UNSPECIFIED \"abc\"",
        "char unspecified[] = UNSPECIFIED;",
        "size_t unspecified_s = sizeof UNSPECIFIED;",

        "signed char signedCharArray[2] = \"abc\";",
        "unsigned char unsignedCharArray[5] = \"abc\";",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    checkDecls(mod,
      new Expect[]{
        new Expect("c_empty", getConstCharArray(ctxt, 0)),
        new Expect("c_empty_s", getSizeTInteger(char_nbytes, ctxt)),

        new Expect("c_simple", getConstCharArray(ctxt, '*', 0)),
        new Expect("c_simple_s", getSizeTInteger(2*char_nbytes, ctxt)),

        new Expect("c_max", getConstCharArray(ctxt, max.longValue(), 0)),
        new Expect("c_max_s", getSizeTInteger(2*char_nbytes, ctxt)),

        new Expect("c_adjacent0", getConstCharArray(ctxt, 'a', 'b', 'c', 0)),
        new Expect("c_adjacent0_s", getSizeTInteger(4*char_nbytes, ctxt)),

        new Expect("c_adjacent1", getConstCharArray(ctxt, 'a', 'b', 'c', 'd',
                                                    'e', 'f', 0)),
        new Expect("c_adjacent1_s", getSizeTInteger(7*char_nbytes, ctxt)),

        new Expect("wc_empty", getConstWcharArray(ctxt, 0)),
        new Expect("wc_empty_s", getSizeTInteger(wchar_nbytes, ctxt)),

        new Expect("wc_simple", getConstWcharArray(ctxt, 'A', 0)),
        new Expect("wc_simple_s", getSizeTInteger(2*wchar_nbytes, ctxt)),

        new Expect("wc_max", getConstWcharArray(ctxt, wmax.longValue(), 0)),
        new Expect("wc_max_s", getSizeTInteger(2*wchar_nbytes, ctxt)),

        new Expect("wc_adjacent0", getConstWcharArray(ctxt, 'a', 'b', 0)),
        new Expect("wc_adjacent0_s", getSizeTInteger(3*wchar_nbytes, ctxt)),

        new Expect("wc_adjacent1", getConstWcharArray(ctxt, 'a', 'b', 0)),
        new Expect("wc_adjacent1_s", getSizeTInteger(3*wchar_nbytes, ctxt)),

        new Expect("wc_adjacent2", getConstWcharArray(ctxt, 'a', 'b', 0)),
        new Expect("wc_adjacent2_s", getSizeTInteger(3*wchar_nbytes, ctxt)),

        new Expect("hex_term1", getConstCharArray(ctxt, 0, 0)),
        new Expect("hex_term1_s", getSizeTInteger(2*char_nbytes, ctxt)),

        new Expect("hex_term2", getConstCharArray(ctxt, 0x1a, 0)),
        new Expect("hex_term2_s", getSizeTInteger(2*char_nbytes, ctxt)),

        new Expect("hex_term3", getConstCharArray(ctxt, 0xbdc, 0)),
        new Expect("hex_term3_s", getSizeTInteger(2*char_nbytes, ctxt)),

        new Expect("hex_term4", getConstCharArray(ctxt, 0xf2aebc, 0)),
        new Expect("hex_term4_s", getSizeTInteger(2*char_nbytes, ctxt)),

        new Expect("hex_term5", getConstCharArray(ctxt, 0x34, '*', 0)),
        new Expect("hex_term5_s", getSizeTInteger(3*char_nbytes, ctxt)),

        new Expect("hex_term6", getConstCharArray(ctxt, 0x56789def, '\\', 0)),
        new Expect("hex_term6_s", getSizeTInteger(3*char_nbytes, ctxt)),

        new Expect("octal_term1", getConstCharArray(ctxt, 1, 0)),
        new Expect("octal_term1_s", getSizeTInteger(2*char_nbytes, ctxt)),

        new Expect("octal_term2", getConstCharArray(ctxt, 2, 0)),
        new Expect("octal_term2_s", getSizeTInteger(2*char_nbytes, ctxt)),

        new Expect("octal_term3", getConstCharArray(ctxt, 0435, 0)),
        new Expect("octal_term3_s", getSizeTInteger(2*char_nbytes, ctxt)),

        new Expect("octal_term4", getConstCharArray(ctxt, 0657, '8', 0)),
        new Expect("octal_term4_s", getSizeTInteger(3*char_nbytes, ctxt)),

        new Expect("octal_term5", getConstCharArray(ctxt, 032, 5, 0)),
        new Expect("octal_term5_s", getSizeTInteger(3*char_nbytes, ctxt)),

        new Expect("octal_term6", getConstCharArray(ctxt, 'x', 0, 'x', 0)),
        new Expect("octal_term6_s", getSizeTInteger(4*char_nbytes, ctxt)),

        new Expect("long_by1", getConstCharArray(ctxt, 'a', 'b', 'c')),
        new Expect("long_by1_s", getSizeTInteger(4*char_nbytes, ctxt)),

        new Expect("long_by2", getConstWcharArray(ctxt, 'a', 'b')),
        new Expect("long_by2_s", getSizeTInteger(4*wchar_nbytes, ctxt)),

        new Expect("short_by1", getConstWcharArray(ctxt, 0, 0)),
        new Expect("short_by1_s", getSizeTInteger(1*wchar_nbytes, ctxt)),

        new Expect("short_by2", getConstCharArray(ctxt, 'a', 'b', 0, 0, 0)),
        new Expect("short_by2_s", getSizeTInteger(3*char_nbytes, ctxt)),

        new Expect("unspecified", getConstCharArray(ctxt, 'a', 'b', 'c', 0)),
        new Expect("unspecified_s", getSizeTInteger(4*char_nbytes, ctxt)),

        new Expect("signedCharArray",
                   getConstIntArray(ctxt, SrcSignedCharType, 'a', 'b')),
        new Expect("unsignedCharArray",
                   getConstIntArray(ctxt, SrcUnsignedCharType,
                                    'a', 'b', 'c', 0, 0)),
      });
  }
  @Test public void stringLiteralPointerInit() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        "#include <stddef.h>",
        // strlen and wcslen are provided by the native target.
        "size_t strlen(char*);",
        "size_t wcslen(wchar_t*);",
        "char *c0 = \"\";",
        "char *c3 = \"abc\";",
        "wchar_t *wc5 = L\"abcde\";",
        "size_t c0_len() { return strlen(c0); }",
        "size_t c3_len() { return strlen(c3); }",
        "size_t wc5_len() { return wcslen(wc5); }",
        "size_t wc0_len() {",
        "  wchar_t *wc0 = L\"\";",
        "  return wcslen(wc0);",
        "}",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    checkIntFn(exec, mod, "c0_len", 0);
    checkIntFn(exec, mod, "c3_len", 3);
    checkIntFn(exec, mod, "wc5_len", 5);
    checkIntFn(exec, mod, "wc0_len", 0);
    new Expect(".str", getConstCharArray(ctxt, 0)).checkStr(mod);
    new Expect(".str1", getConstCharArray(ctxt, 'a', 'b', 'c', 0)).checkStr(mod);
    new Expect(".str2", getConstWcharArray(ctxt, 'a', 'b', 'c', 'd', 'e', 0)).checkStr(mod);
    new Expect(".str3", getConstWcharArray(ctxt, 0)).checkStr(mod);
    exec.dispose();
  }

  @Test public void identifier() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = buildLLVMSimple(
      "x86_64", "",
      new String[]{
        "#include <stdarg.h>",
        "#include <stddef.h>",

        // enumerators

        "enum E {EA, EB = 3, EC} e = EC;",
        "long var = 5l;",
        "enum E get_EA() { return EA; }",
        "enum E get_EB() { return EB; }",
        "enum E get_EC() { return EC; }",
        "enum E get_e() { return e; }",
        "double get_local_var() { double var = 9.8; return var; }",
        "long get_global_var() { return var; }",
        "size_t get_EA_size() { return sizeof EA; }",
        "size_t get_EB_size() { return sizeof EB; }",
        "size_t get_EC_size() { return sizeof EC; }",
        "size_t get_e_size() { return sizeof e; }",
        "size_t get_local_var_size() { double var; return sizeof var; }",
        "size_t get_global_var_size() { return sizeof var; }",

        // functions and function pointers

        // sin and cos are provided by the native target.
        "int fnDef() { return 103; }",
        "double sin(double);",
        "typedef double FnT(double);",
        "FnT cos;",
        "int call_fnDef() { return fnDef(); }",
        "double call_sin() { return sin(1.570796); }",
        "double call_cos() { return cos(1.570796); }",

        "double fnParam(double fn(double)) { return fn(1.570796); }",
        "double call_fnParam() { return fnParam(sin); }",
        "double fnPtrParam(double (*fn)(double)) {", // param with nested decl
        "  double (*p)(double) = fn;", // local var with nested decl
        "  return p(1.570796);",
        "}",
        "double call_fnPtrParam() { return fnPtrParam(cos); }",

        "typedef int (*FnT2)();",
        "FnT2 fnT2Ptr = fnDef;",
        "int call_fnT2Ptr() { return fnT2Ptr(); }",

        "typedef int (*FnT3())();",
        "FnT3 fnT3;",
        "int (*fnT3())() { return fnT2Ptr; }",
        "int call_fnT3() { return fnT3()(); }",

        // Make sure symbol lookup doesn't fail for function that returns
        // pointer to function (because the symbol is a NestedDeclarator).
        "int (*fn())();",
        "int (*(*foo())())() { return fn; }",

        // function identifier as pointer
        "double (*sinPtr)(double) = sin;",
        "double call_sinPtr() { return sinPtr(3.141593); }",
        "double call_cosPtr() {",
        "  double (*cosPtr)(double) = cos;",
        "  return cosPtr(3.141593);",
        "}",

        // Check that argument to Procedure (fnPtrFn) with NestedDeclarator is
        // accessible. This used to fail due to a Cetus bug.
        // sprintf and atoi are provided by the native target.
        "int sprintf(char *str, char *format, ...);",
        "int atoi(char *str);",
        "char fnPtrFn_str[] = \"25\";",
        "void (*fnPtrFn(int i))() { sprintf(fnPtrFn_str, \"%d\", i); }",
        "int get_fnPtrFn() { return atoi(fnPtrFn_str); }",

        // structs and unions

        // Just make sure this compiles for now. We don't have enough
        // implemented to do much with structs/unions yet.
        "struct S { int i; double d; } s;",
        "void initStruct() {",
        "  struct S sLocal = s;",
        "  struct S sLocal2 = sLocal;",
        "}",

        // static locals

        // Check that static local is initialized once and maintains its
        // value.
        "char staticLocal() {",
        "  static char c;", // unspecified init is 0
        "  static char next = 5;", // specified init
        "  char seven = 7;", // specified init
        "  char oldc = c;",
        "  c = next;",
        "  next = seven;",
        "  return oldc;",
        "}",

        // Make sure you can call a function after its prototype but before
        // its definition. (At one time, BuildLLVM could not resolve the
        // function call in this case.)

        "int fnCalledBeforeDef();",
        "int callFnBeforeDef() { return fnCalledBeforeDef(); }",
        "int fnCalledBeforeDef() { return 5; }",

        "static int fnStaticCalledBeforeDef();",
        "int callFnStaticBeforeDef() { return fnStaticCalledBeforeDef(); }",
        "int fnStaticCalledBeforeDef() { return 9; }",

        // Make sure you can reference a variable before its external
        // definition. (At one time, BuildLLVM could not resolve the
        // references in these cases.)

        // tentative def only, external linkage
        "int varRefBeforeDef;",
        "size_t varRefBeforeDefSize = sizeof varRefBeforeDef;",
        "int getVarRefBeforeDefValue() { return varRefBeforeDef; }",
        "size_t getVarRefBeforeDefSize() { return varRefBeforeDefSize; }",
        "int varRefBeforeDef = 2;",

        // tentative def only, internal linkage
        "static int varStaticRefBeforeDef;",
        "size_t varStaticRefBeforeDefSize = sizeof varStaticRefBeforeDef;",
        "int getVarStaticRefBeforeDefValue() { return varStaticRefBeforeDef; }",
        "size_t getVarStaticRefBeforeDefSize() { return varStaticRefBeforeDefSize; }",
        "int varStaticRefBeforeDef = 98;",

        // extern declaration only
        "extern int varExternRefBeforeDef;",
        "size_t varExternRefBeforeDefSize = sizeof varExternRefBeforeDef;",
        "int getVarExternRefBeforeDefValue() { return varExternRefBeforeDef; }",
        "size_t getVarExternRefBeforeDefSize() { return varExternRefBeforeDefSize; }",
        "int varExternRefBeforeDef = 98;",

        // Identifier lookup from nodes that have no parent in the Cetus IR.
        // BuildLLVM works around this Cetus problem. The case of a parent-less
        // ProcedureDeclarator for a Procedure is checked in
        // BuildLLVMTest_Types#functionParamType.

        // array dimensions
        "int i;",
        "char arrDimParent0[sizeof i];",
        "char arrDimParent1[sizeof s.i];",

        // type operands
        "size_t sizeofTypeParent = sizeof(int[sizeof i]);",
        // Cetus moves struct definitions outside expressions, so there
        // doesn't seem to be any need for symbol lookup within an offsetof's
        // type operand. In case Cetus changes, we exercise this case anyway.
        "size_t offsetofTypeParent() {",
        "  return offsetof(struct {char a[sizeof i]; int m;}, m);",
        "}",
        "size_t typecastParent = sizeof (*(char(*)[sizeof i])0);",
        "size_t vaArgParent(int n, ...) {",
        "  va_list args;",
        "  va_start(args, n);",
        "  size_t s = sizeof (*va_arg(args, char(*)[sizeof i]));",
        "  va_end(args);",
        "  return s;",
        "}",

        // bit-field widths
        "union BitFieldParent {",
        "  int b : sizeof i;",
        "  unsigned int i;",
        "} bitFieldParent = {-1};",
        "unsigned int getBitFieldParent() { return bitFieldParent.i; }",

        // Reference to i resolves as forward-reference to local i in Cetus's
        // symbol tables. At one time, BuildLLVM then looked i up in its own
        // local tables and didn't find i, so it then looked i up in its own
        // file-scope table and found i. Now, it never uses the file-scope
        // table until it has exhausted all other scopes.

        "int forwardRefMixup_i = 10;",
        "int forwardRefMixup() {",
        "  int forwardRefMixup_i = 9;",
        "  {",
        "    int j = forwardRefMixup_i;",
        "    int forwardRefMixup_i = 8;",
        "    return j;",
        "  }",
        "}",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    final long enumConstSize = SRC_ENUM_CONST_TYPE.getLLVMWidth()/8;
    final long enumTypeSize
      = SrcEnumType.COMPATIBLE_INTEGER_TYPE.getLLVMWidth()/8;
    final long charSize = SrcCharType.getLLVMWidth()/8;
    final long intSize = SrcIntType.getLLVMWidth()/8;
    final long doubleSize
      = SrcDoubleType.getLLVMType(ctxt).getPrimitiveSizeInBits()/8;
    final long longSize = SrcLongType.getLLVMWidth()/8;
    checkIntFn(exec, mod, "get_EA", 0);
    checkIntFn(exec, mod, "get_EB", 3);
    checkIntFn(exec, mod, "get_EC", 4);
    checkIntFn(exec, mod, "get_e", 4);
    checkDoubleFn(exec, mod, "get_local_var", 9.8);
    checkIntFn(exec, mod, "get_global_var", 5);
    checkIntFn(exec, mod, "get_EA_size", enumConstSize);
    checkIntFn(exec, mod, "get_EB_size", enumConstSize);
    checkIntFn(exec, mod, "get_EC_size", enumConstSize);
    checkIntFn(exec, mod, "get_e_size", enumTypeSize);
    checkIntFn(exec, mod, "get_local_var_size", doubleSize);
    checkIntFn(exec, mod, "get_global_var_size", longSize);
    checkIntFn(exec, mod, "call_fnDef", 103);
    checkDoubleFn(exec, mod, "call_sin", 1.);
    checkDoubleFn(exec, mod, "call_cos", 0.);
    checkDoubleFn(exec, mod, "call_fnParam", 1.);
    checkDoubleFn(exec, mod, "call_fnPtrParam", 0.);
    checkIntFn(exec, mod, "call_fnT2Ptr", 103);
    checkIntFn(exec, mod, "call_fnT3", 103);
    checkDoubleFn(exec, mod, "call_sinPtr", 0.);
    checkDoubleFn(exec, mod, "call_cosPtr", -1.);
    runFn(exec, mod, "fnPtrFn",
          new LLVMGenericValue[]{
            new LLVMGenericInt(SrcIntType.getLLVMType(ctxt),
                               BigInteger.valueOf(32), true)
          });
    checkIntFn(exec, mod, "get_fnPtrFn", 32);
    checkIntFn(exec, mod, "staticLocal", 0);
    checkIntFn(exec, mod, "staticLocal", 5);
    checkIntFn(exec, mod, "staticLocal", 7);
    checkIntFn(exec, mod, "staticLocal", 7);
    checkIntFn(exec, mod, "staticLocal", 7);
    checkIntFn(exec, mod, "callFnBeforeDef", 5);
    checkIntFn(exec, mod, "callFnStaticBeforeDef", 9);
    checkIntFn(exec, mod, "getVarRefBeforeDefValue", 2);
    checkIntFn(exec, mod, "getVarRefBeforeDefSize",
               SrcIntType.getLLVMWidth()/8);
    checkIntFn(exec, mod, "getVarStaticRefBeforeDefValue", 98);
    checkIntFn(exec, mod, "getVarStaticRefBeforeDefSize",
               SrcIntType.getLLVMWidth()/8);
    checkIntFn(exec, mod, "getVarExternRefBeforeDefValue", 98);
    checkIntFn(exec, mod, "getVarExternRefBeforeDefSize",
               SrcIntType.getLLVMWidth()/8);

    // Identifier lookup from nodes that have no parent in the Cetus IR.
    checkGlobalVar(mod, "arrDimParent0",
                   SrcArrayType.get(SrcCharType, charSize*intSize));
    checkGlobalVar(mod, "arrDimParent1",
                   SrcArrayType.get(SrcCharType, charSize*intSize));
    checkGlobalVar(mod, "sizeofTypeParent",
                   getSizeTInteger(intSize*intSize, ctxt));
    checkIntFn(exec, mod, "offsetofTypeParent", charSize*intSize);
    checkGlobalVar(mod, "typecastParent",
                   getSizeTInteger(charSize*intSize, ctxt));
    checkIntFn(exec, mod, charSize*intSize, "vaArgParent", 0);
    checkIntFn(exec, mod, "getBitFieldParent",
               BigInteger.ONE.shiftLeft((int)intSize).subtract(BigInteger.ONE)
               .shiftLeft((int)(intSize*8-intSize)).longValue());

    checkIntFn(exec, mod, "forwardRefMixup", 9);

    exec.dispose();
  }
}
