package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcPrimitiveFloatingType.SrcDoubleType;
import static openacc.codegen.llvmBackend.SrcPrimitiveFloatingType.SrcFloatType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SRC_PTRDIFF_T_TYPE;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SRC_SIZE_T_TYPE;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcCharType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcIntType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcLongLongType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcLongType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedCharType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedIntType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedLongLongType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedLongType;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.math.BigInteger;

import org.jllvm.LLVMArrayType;
import org.jllvm.LLVMConstant;
import org.jllvm.LLVMConstantInteger;
import org.jllvm.LLVMContext;
import org.jllvm.LLVMExecutionEngine;
import org.jllvm.LLVMGenericValue;
import org.jllvm.LLVMGlobalVariable;
import org.jllvm.LLVMIntegerType;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMPointerType;
import org.jllvm.LLVMTargetData;
import org.jllvm.bindings.LLVMLinkage;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Checks the ability to build correct LLVM IR for C expressions other than
 * primary expressions, which are checked in
 * {@link BuildLLVMTest_PrimaryExpressions}.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class BuildLLVMTest_OtherExpressions extends BuildLLVMTest {
  // long double is x86_fp80, which has no default alignment so LLVM will fail
  // an assertion if we try to compute its size without this.
  private static final String TARGET_DATA_LAYOUT = "f80:128";
  private static final int SIZEOF_LONG_DOUBLE = 16; // = 128/8

  @BeforeClass public static void setup() {
    System.loadLibrary("jllvm");
  }

  /**
   * Checks various forms of function calls. Default argument promotions are
   * checked in
   * {@link BuildLLVMTest_TypeConversions#defaultArgumentPromotions}, and
   * argument to parameter type conversions are checked in
   * {@link BuildLLVMTest_TypeConversions#argToParamType}.
   */
  @Test public void functionCall() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        "#include <stddef.h>",

        // Evaluate sizeof for function call.
        "int fnDef() { return 103; }",
        "double fnProto(int);",
        "typedef char FnT(int, int);",
        "FnT fnProtoTypedef;",
        "size_t fnDef_retSize() { return sizeof fnDef(); }",
        "size_t fnProto_retSize() { return sizeof fnProto(3); }",
        "size_t fnProtoTypedef_retSize() { return sizeof fnProtoTypedef(0, 1); }",
        "size_t fnPtr_retSize() { int (*p)(); return sizeof p(); }",

        // Evaluate function calls with different numbers of arguments.
        // pow is provided by the native target.
        "double pow(double, double);",
        "int fn0Arg() { return 5; }",
        "int fn1Arg(int i) { return i; }",
        "double fn2Arg(double i, double j) { return pow(i, j); }",
        "int call_fn0Arg() { return fn0Arg(); }",
        "int call_fn1Arg() { return fn1Arg(9); }",
        "double call_fn2Arg() { return fn2Arg(2., 3.); }",

        // Evaluate function call with void return type, and evaluate
        // variadic function call.
        // sprintf and atoi are provided by the native target.
        "int sprintf(char *str, char *format, ...);",
        "int atoi(char *str);",
        "char str[3] = \"53\";",
        "void fnVoid() {",
        "  char fmt[3] = \"%d\";",
        "  sprintf(str, fmt, 67);",
        "};",
        "int call_fnVoid() { fnVoid(); return atoi(str); }",

        // Try passing a string literal as an argument.
        "int call_withStringLiteral() { return atoi(\"25\"); }",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    final long intSize = SrcIntType.getLLVMWidth()/8;
    final long doubleSize = SrcDoubleType.getLLVMType(ctxt).getPrimitiveSizeInBits()/8;
    final long charSize = SrcCharType.getLLVMWidth()/8;
    checkIntFn(exec, mod, "fnDef_retSize", intSize);
    checkIntFn(exec, mod, "fnProto_retSize", doubleSize);
    checkIntFn(exec, mod, "fnProtoTypedef_retSize", charSize);
    checkIntFn(exec, mod, "fnProtoTypedef_retSize", charSize);
    checkIntFn(exec, mod, "fnPtr_retSize", intSize);
    checkIntFn(exec, mod, "call_fn0Arg", 5);
    checkIntFn(exec, mod, "call_fn1Arg", 9);
    checkDoubleFn(exec, mod, "call_fn2Arg", 8.);
    checkIntFn(exec, mod, "call_fnVoid", 67);

    checkIntFn(exec, mod, "call_withStringLiteral", 25);
    final LLVMGlobalVariable str25 = mod.getNamedGlobal(".str");
    assertNotNull(".str must exist as a global variable", str25.getInstance());
    assertEquals(
      ".str's type",
      LLVMPointerType.get(LLVMArrayType.get(SrcCharType.getLLVMType(ctxt), 3),
                          0),
      str25.typeOf());
    assertEquals(".str's linkage", LLVMLinkage.LLVMPrivateLinkage,
                 str25.getLinkage());
    assertTrue(".str must be constant", str25.isConstant());
    exec.dispose();
  }

  private static BigInteger repeatCharInUnsignedInt(int c) {
    BigInteger res = BigInteger.ZERO;
    for (int i = 0;
         i < SrcUnsignedIntType.getWidth()/SrcUnsignedCharType.getWidth();
         ++i)
      res = res.shiftLeft((int)SrcUnsignedCharType.getWidth())
            .add(BigInteger.valueOf(c));
    return res;
  }

  /**
   * Checks . and -> postfix expressions. Also minimally checks compound
   * initializers for structs and unions, including any conversions required
   * in struct initializers, whether as constant expressions or not.
   */
  @Test public void memberAccess() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final BigInteger repeat5 = repeatCharInUnsignedInt(5);
    final BigInteger repeat32 = repeatCharInUnsignedInt(32);
    final BigInteger repeat108 = repeatCharInUnsignedInt(108);
    final BigInteger repeat120 = repeatCharInUnsignedInt(120);
    final BigInteger repeat210 = repeatCharInUnsignedInt(210);
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        "#include <stddef.h>",
        "struct S {int mem0; double mem1;} structGlobal = {30, 938.4e1};",
        "struct S structGlobalPtr[1] = {{-193.6, 2931.5f}};", // conversions
        "union U {unsigned mem0; unsigned char mem1;} unionGlobal",
        "  = {0x"+repeat5.toString(16)+"};",
        "union U unionGlobalPtr[1] = {{0x"+repeat32.toString(16)+"}};",

        // constant expression, lvalue dot. TODO: Use address-of, when
        // implemented, to get the address of the dot operator result when in a
        // global variable init to check that it actually is a constant
        // expression.
        "int globalStructDotMem0()          { return structGlobal.mem0; }",
        "double globalStructDotMem1()       { return structGlobal.mem1; }",
        "unsigned globalUnionDotMem0()      { return unionGlobal.mem0; }",
        "unsigned char globalUnionDotMem1() { return unionGlobal.mem1; }",
        "size_t globalStructDotMem0Size = sizeof structGlobal.mem0;",
        "size_t globalStructDotMem1Size = sizeof structGlobal.mem1;",
        "size_t globalUnionDotMem0Size  = sizeof unionGlobal.mem0;",
        "size_t globalUnionDotMem1Size  = sizeof unionGlobal.mem1;",

        // constant expression, ->. Using a struct array here instead of
        // taking the address of a struct checks that prepareForOp conversions
        // (in this case, array-to-pointer) are run on left-hand side.
        // TODO: Use address-of, when implemented, to get the address of the
        // -> operator result when in a global variable init to check that it
        // actually is a constant expression.
        "int globalStructArrowMem0()          { return structGlobalPtr->mem0; }",
        "double globalStructArrowMem1()       { return structGlobalPtr->mem1; }",
        "unsigned globalUnionArrowMem0()      { return unionGlobalPtr->mem0; }",
        "unsigned char globalUnionArrowMem1() { return unionGlobalPtr->mem1; }",
        "size_t globalStructArrowMem0Size = sizeof structGlobalPtr->mem0;",
        "size_t globalStructArrowMem1Size = sizeof structGlobalPtr->mem1;",
        "size_t globalUnionArrowMem0Size = sizeof unionGlobalPtr->mem0;",
        "size_t globalUnionArrowMem1Size = sizeof unionGlobalPtr->mem1;",

        // constant expression, rvalue dot appears impossible. See comments in
        // SrcStructType.accessMember and SrcUnionType.accessMember.

        // non-constant expression, lvalue dot.
        "int localStructDotMem0()    { struct S s = {'\\5', 2}; return s.mem0; }", // conversions
        "double localStructDotMem1() { struct S s = {5, 2}; return s.mem1; }",
        "unsigned localUnionDotMem0() {",
        "  union U u = {0x"+repeat108.toString(16)+"};",
        "  return u.mem0;",
        "}",
        "unsigned char localUnionDotMem1() {",
        "  union U u = {0x"+repeat108.toString(16)+"};",
        "  return u.mem1;",
        "}",
        "size_t localStructDotMem0Size() { struct S s; return sizeof s.mem0; }",
        "size_t localStructDotMem1Size() { struct S s; return sizeof s.mem1; }",
        "size_t localUnionDotMem0Size() { union U u; return sizeof u.mem0; }",
        "size_t localUnionDotMem1Size() { union U u; return sizeof u.mem1; }",

        // non-constant expression, ->.
        "int localStructArrowMem0()        { struct S s[1] = {{0, -300.8e0}}; return s->mem0; }",
        "double localStructArrowMem1()     { struct S s[1] = {{0, -300.8e0}}; return s->mem1; }",
        "unsigned localUnionArrowMem0() {",
        "  union U u[1] = {{0x"+repeat120.toString(16)+"}};",
        "  return u->mem0;",
        "}",
        "unsigned char localUnionArrowMem1() {",
        "  union U u[1] = {{0x"+repeat120.toString(16)+"}};",
        "  return u->mem1;",
        "}",
        "size_t localStructArrowMem0Size() { struct S s[1]; return sizeof s->mem0; }",
        "size_t localStructArrowMem1Size() { struct S s[1]; return sizeof s->mem1; }",
        "size_t localUnionArrowMem0Size() { union U u[1]; return sizeof u->mem0; }",
        "size_t localUnionArrowMem1Size() { union U u[1]; return sizeof u->mem1; }",

        // non-constant expression, rvalue dot.
        "struct S getStruct() { struct S s = {873, 23.7e-1}; return s; }",
        "union U getUnion() {",
        "  union U u = {0x"+repeat210.toString(16)+"};",
        "  return u;",
        "}",
        "int valueStructDotMem0()          { return getStruct().mem0; }",
        "double valueStructDotMem1()       { return getStruct().mem1; }",
        "unsigned valueUnionDotMem0()      { return getUnion().mem0; }",
        "unsigned char valueUnionDotMem1() { return getUnion().mem1; }",
        "size_t valueStructDotMem0Size = sizeof getStruct().mem0;",
        "size_t valueStructDotMem1Size = sizeof getStruct().mem1;",
        "size_t valueUnionDotMem0Size = sizeof getUnion().mem0;",
        "size_t valueUnionDotMem1Size = sizeof getUnion().mem1;",

        // bit-fields. sizeof and & are not permitted for this case, but we
        // can check sizeof for bit-fields after integer promotions due to
        // some operator.

        "struct SB { int i:4; int :1; unsigned j:3; int k:5; int :0; int l:1; };",
        "struct SB sb = { 15, -1, 15 };",
        "int globalStructDotBitField0() { return sb.i; }",
        "int globalStructDotBitField1() { return sb.j; }",
        "int globalStructDotBitField2() { return sb.k; }",
        "int globalStructDotBitField3() { return sb.l; }",
        "int localStructDotBitField0() { struct SB s={-1,7,-16,1}; return s.i; }",
        "int localStructDotBitField1() { struct SB s={-1,7,-16,1}; return s.j; }",
        "int localStructDotBitField2() { struct SB s={-1,7,-16,1}; return s.k; }",
        "int localStructDotBitField3() { struct SB s={-1,7,-16,1}; return s.l; }",
        "struct SB getSB() { struct SB sb = {15, -1, -1, -1}; return sb; }",
        "int rvalueStructDotBitField0() { return getSB().i; }",
        "int rvalueStructDotBitField1() { return getSB().j; }",
        "int rvalueStructDotBitField2() { return getSB().k; }",
        "int rvalueStructDotBitField3() { return getSB().l; }",

        "union UB { int i:5; int :1; unsigned j:3; int k:4; int :0; int l:1; };",
        "union UB ub = { 0x15 };",
        "int globalUnionDotBitField0() { return ub.i; }",
        "int globalUnionDotBitField1() { return ub.j; }",
        "int globalUnionDotBitField2() { return ub.k; }",
        "int globalUnionDotBitField3() { return ub.l; }",
        "int localUnionDotBitField0() { union UB u={0x1B}; return u.i; }",
        "int localUnionDotBitField1() { union UB u={0x1B}; return u.j; }",
        "int localUnionDotBitField2() { union UB u={0x1B}; return u.k; }",
        "int localUnionDotBitField3() { union UB u={0x1B}; return u.l; }",
        "union UB getUB() { union UB ub = {0xA}; return ub; }",
        "int rvalueUnionDotBitField0() { return getUB().i; }",
        "int rvalueUnionDotBitField1() { return getUB().j; }",
        "int rvalueUnionDotBitField2() { return getUB().k; }",
        "int rvalueUnionDotBitField3() { return getUB().l; }",

        // ISO C99 says how to promote bit-fields of type _Bool, int, signed
        // int, or unsigned int. The target type is always an int or unsigned
        // int. It doesn't say how to promote other bit-field types that a
        // compiler chooses to support, but we should certainly not convert a
        // 33-bit bit-field to an int or unsigned int.
        "struct SB2 {long i:33;} sb2;",
        "size_t bitFieldIntPromote() { return sizeof +sb.i; }",
        "size_t bitFieldLongPromote() { return sizeof +sb2.i; }",

        // Make sure signedness isn't lost when packing signed and unsigned
        // together.
        "struct SBSigned {",
        "  int            i : 16;",
        "  unsigned       u : 16;",
        "  unsigned char uc : 4;",
        "  signed   char sc : 4;",
        "} sbSigned = {-1, -1, -1, -1};",
        "long getSBSignedMem0() { return sbSigned.i; }",
        "long getSBSignedMem1() { return sbSigned.u; }",
        "long getSBSignedMem2() { return sbSigned.uc; }",
        "long getSBSignedMem3() { return sbSigned.sc; }",

        // C11 anonymous struct/union.

        "struct SA {",
        "  double mem0;",
        "  struct { int mem1; struct { char mem2; }; unsigned mem3; };",
        "  union { int mem4; int mem5; };",
        "  float mem6;",
        "} sa = {11, {22, {33}, 44}, {55}, {66}}, *sap = &sa;",

        // constant expression
        "double   globalStructDotAnonMem0() { return sa.mem0; }",
        "int      globalStructDotAnonMem1() { return sa.mem1; }",
        "char     globalStructDotAnonMem2() { return sa.mem2; }",
        "unsigned globalStructDotAnonMem3() { return sa.mem3; }",
        "int      globalStructDotAnonMem4() { return sa.mem4; }",
        "int      globalStructDotAnonMem5() { return sa.mem5; }",
        "float    globalStructDotAnonMem6() { return sa.mem6; }",
        "double   globalStructArrowAnonMem0() { return sap->mem0; }",
        "int      globalStructArrowAnonMem1() { return sap->mem1; }",
        "char     globalStructArrowAnonMem2() { return sap->mem2; }",
        "unsigned globalStructArrowAnonMem3() { return sap->mem3; }",
        "int      globalStructArrowAnonMem4() { return sap->mem4; }",
        "int      globalStructArrowAnonMem5() { return sap->mem5; }",
        "float    globalStructArrowAnonMem6() { return sap->mem6; }",

        // non-constant expression, lvalue
        "double   localStructDotAnonMem0() { struct SA s = {33.2, {3, {-2}, 8}, {22}, .3}; return s.mem0; }",
        "int      localStructDotAnonMem1() { struct SA s = {33.2, {3, {-2}, 8}, {22}, .3}; return s.mem1; }",
        "char     localStructDotAnonMem2() { struct SA s = {33.2, {3, {-2}, 8}, {22}, .3}; return s.mem2; }",
        "unsigned localStructDotAnonMem3() { struct SA s = {33.2, {3, {-2}, 8}, {22}, .3}; return s.mem3; }",
        "int      localStructDotAnonMem4() { struct SA s = {33.2, {3, {-2}, 8}, {22}, .3}; return s.mem4; }",
        "int      localStructDotAnonMem5() { struct SA s = {33.2, {3, {-2}, 8}, {22}, .3}; return s.mem5; }",
        "float    localStructDotAnonMem6() { struct SA s = {33.2, {3, {-2}, 8}, {22}, .3}; return s.mem6; }",
        "double   localStructArrowAnonMem0() { struct SA *p = &sa; return p->mem0; }",
        "int      localStructArrowAnonMem1() { struct SA *p = &sa; return p->mem1; }",
        "char     localStructArrowAnonMem2() { struct SA *p = &sa; return p->mem2; }",
        "unsigned localStructArrowAnonMem3() { struct SA *p = &sa; return p->mem3; }",
        "int      localStructArrowAnonMem4() { struct SA *p = &sa; return p->mem4; }",
        "int      localStructArrowAnonMem5() { struct SA *p = &sa; return p->mem5; }",
        "float    localStructArrowAnonMem6() { struct SA *p = &sa; return p->mem6; }",

        // non-constant expression, rvalue
        "struct SA getSA() { struct SA s = {33.2, {3, {-2}, 8}, {22}, .3}; return s; }",
        "double   valueStructDotAnonMem0() { return getSA().mem0; }",
        "int      valueStructDotAnonMem1() { return getSA().mem1; }",
        "char     valueStructDotAnonMem2() { return getSA().mem2; }",
        "unsigned valueStructDotAnonMem3() { return getSA().mem3; }",
        "int      valueStructDotAnonMem4() { return getSA().mem4; }",
        "int      valueStructDotAnonMem5() { return getSA().mem5; }",
        "float    valueStructDotAnonMem6() { return getSA().mem6; }",

        // Strangely, the LLVM C API function LLVMIsAConstantStruct returns
        // null for this initializer, so BuildLLVM used to fail here. See
        // LLVMValue.getValue for details. The zero value here is significant
        // because 1, for example, does not have the problem.

        "struct {int i;} c_zeroInit = {0};",
        "int getZeroInit() { return c_zeroInit.i; }",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    final long unsignedCharSize = SrcUnsignedCharType.getLLVMWidth()/8;
    final long intSize = SrcIntType.getLLVMWidth()/8;
    final long unsignedIntSize = SrcUnsignedIntType.getLLVMWidth()/8;
    final long doubleSize = SrcDoubleType.getLLVMType(ctxt).getPrimitiveSizeInBits()/8;

    checkIntFn(exec, mod, "globalStructDotMem0", 30);
    checkDoubleFn(exec, mod, "globalStructDotMem1", 938.4e1);
    checkIntFn(exec, mod, "globalUnionDotMem0", repeat5.longValue());
    checkIntFn(exec, mod, "globalUnionDotMem1", 5);
    checkIntGlobalVar(mod, "globalStructDotMem0Size", intSize);
    checkIntGlobalVar(mod, "globalStructDotMem1Size", doubleSize);
    checkIntGlobalVar(mod, "globalUnionDotMem0Size", unsignedIntSize);
    checkIntGlobalVar(mod, "globalUnionDotMem1Size", unsignedCharSize);

    checkIntFn(exec, mod, "globalStructArrowMem0", -193);
    checkDoubleFn(exec, mod, "globalStructArrowMem1", 2931.5);
    checkIntFn(exec, mod, "globalUnionArrowMem0", repeat32.longValue());
    checkIntFn(exec, mod, "globalUnionArrowMem1", 32);
    checkIntGlobalVar(mod, "globalStructArrowMem0Size", intSize);
    checkIntGlobalVar(mod, "globalStructArrowMem1Size", doubleSize);
    checkIntGlobalVar(mod, "globalUnionArrowMem0Size", unsignedIntSize);
    checkIntGlobalVar(mod, "globalUnionArrowMem1Size", unsignedCharSize);

    checkIntFn(exec, mod, "localStructDotMem0", 5);
    checkDoubleFn(exec, mod, "localStructDotMem1", 2);
    checkIntFn(exec, mod, "localUnionDotMem0", repeat108.longValue());
    checkIntFn(exec, mod, "localUnionDotMem1", 108);
    checkIntFn(exec, mod, "localStructDotMem0Size", intSize);
    checkIntFn(exec, mod, "localStructDotMem1Size", doubleSize);
    checkIntFn(exec, mod, "localUnionDotMem0Size", intSize);
    checkIntFn(exec, mod, "localUnionDotMem1Size", unsignedCharSize);

    checkIntFn(exec, mod, "localStructArrowMem0", 0);
    checkDoubleFn(exec, mod, "localStructArrowMem1", -300.8);
    checkIntFn(exec, mod, "localStructArrowMem0Size", intSize);
    checkIntFn(exec, mod, "localStructArrowMem1Size", doubleSize);
    checkIntFn(exec, mod, "localUnionArrowMem0", repeat120.longValue());
    checkIntFn(exec, mod, "localUnionArrowMem1", 120);
    checkIntFn(exec, mod, "localUnionArrowMem0Size", intSize);
    checkIntFn(exec, mod, "localUnionArrowMem1Size", unsignedCharSize);

    checkIntFn(exec, mod, "valueStructDotMem0", 873);
    checkDoubleFn(exec, mod, "valueStructDotMem1", 23.7e-1);
    checkIntFn(exec, mod, repeat210.longValue(), false, "valueUnionDotMem0");
    checkIntFn(exec, mod, 210, false, "valueUnionDotMem1");
    checkIntGlobalVar(mod, "valueStructDotMem0Size", intSize);
    checkIntGlobalVar(mod, "valueStructDotMem1Size", doubleSize);
    checkIntGlobalVar(mod, "valueUnionDotMem0Size", intSize);
    checkIntGlobalVar(mod, "valueUnionDotMem1Size", unsignedCharSize);

    // bit-fields

    checkIntFn(exec, mod, "globalStructDotBitField0", -1);
    checkIntFn(exec, mod, "globalStructDotBitField1", 7);
    checkIntFn(exec, mod, "globalStructDotBitField2", 15);
    checkIntFn(exec, mod, "globalStructDotBitField3", 0);
    checkIntFn(exec, mod, "localStructDotBitField0", -1);
    checkIntFn(exec, mod, "localStructDotBitField1", 7);
    checkIntFn(exec, mod, "localStructDotBitField2", -16);
    checkIntFn(exec, mod, "localStructDotBitField3", -1);
    checkIntFn(exec, mod, "rvalueStructDotBitField0", -1);
    checkIntFn(exec, mod, "rvalueStructDotBitField1", 7);
    checkIntFn(exec, mod, "rvalueStructDotBitField2", -1);
    checkIntFn(exec, mod, "rvalueStructDotBitField3", -1);

    checkIntFn(exec, mod, "globalUnionDotBitField0", -11);
    checkIntFn(exec, mod, "globalUnionDotBitField1", 5);
    checkIntFn(exec, mod, "globalUnionDotBitField2", -6);
    checkIntFn(exec, mod, "globalUnionDotBitField3", -1);
    checkIntFn(exec, mod, "localUnionDotBitField0", -5);
    checkIntFn(exec, mod, "localUnionDotBitField1", 6);
    checkIntFn(exec, mod, "localUnionDotBitField2", -3);
    checkIntFn(exec, mod, "localUnionDotBitField3", -1);
    checkIntFn(exec, mod, "rvalueUnionDotBitField0", 10);
    checkIntFn(exec, mod, "rvalueUnionDotBitField1", 2);
    checkIntFn(exec, mod, "rvalueUnionDotBitField2", 5);
    checkIntFn(exec, mod, "rvalueUnionDotBitField3", 0);

    checkIntFn(exec, mod, "bitFieldIntPromote", SrcIntType.getLLVMWidth()/8);
    checkIntFn(exec, mod, "bitFieldLongPromote", SrcLongType.getLLVMWidth()/8);

    checkIntFn(exec, mod, "getSBSignedMem0", -1);
    checkIntFn(exec, mod, "getSBSignedMem1", 65535);
    checkIntFn(exec, mod, "getSBSignedMem2", 15);
    checkIntFn(exec, mod, "getSBSignedMem3", -1);

    // C11 anonymous struct/union.

    // constant expression
    checkDoubleFn(exec, mod, "globalStructDotAnonMem0", 11);
    checkIntFn(exec, mod, "globalStructDotAnonMem1", 22);
    checkIntFn(exec, mod, "globalStructDotAnonMem2", 33);
    checkIntFn(exec, mod, "globalStructDotAnonMem3", 44);
    checkIntFn(exec, mod, "globalStructDotAnonMem4", 55);
    checkIntFn(exec, mod, "globalStructDotAnonMem5", 55);
    checkFloatFn(exec, mod, "globalStructDotAnonMem6", 66);
    checkDoubleFn(exec, mod, "globalStructArrowAnonMem0", 11);
    checkIntFn(exec, mod, "globalStructArrowAnonMem1", 22);
    checkIntFn(exec, mod, "globalStructArrowAnonMem2", 33);
    checkIntFn(exec, mod, "globalStructArrowAnonMem3", 44);
    checkIntFn(exec, mod, "globalStructArrowAnonMem4", 55);
    checkIntFn(exec, mod, "globalStructArrowAnonMem5", 55);
    checkFloatFn(exec, mod, "globalStructArrowAnonMem6", 66);

    // non-constant expression, lvalue
    checkDoubleFn(exec, mod, "localStructDotAnonMem0", 33.2);
    checkIntFn(exec, mod, "localStructDotAnonMem1", 3);
    checkIntFn(exec, mod, "localStructDotAnonMem2", -2);
    checkIntFn(exec, mod, "localStructDotAnonMem3", 8);
    checkIntFn(exec, mod, "localStructDotAnonMem4", 22);
    checkIntFn(exec, mod, "localStructDotAnonMem5", 22);
    checkFloatFn(exec, mod, "localStructDotAnonMem6", .3);
    checkDoubleFn(exec, mod, "localStructArrowAnonMem0", 11);
    checkIntFn(exec, mod, "localStructArrowAnonMem1", 22);
    checkIntFn(exec, mod, "localStructArrowAnonMem2", 33);
    checkIntFn(exec, mod, "localStructArrowAnonMem3", 44);
    checkIntFn(exec, mod, "localStructArrowAnonMem4", 55);
    checkIntFn(exec, mod, "localStructArrowAnonMem5", 55);
    checkFloatFn(exec, mod, "localStructArrowAnonMem6", 66);

    checkDoubleFn(exec, mod, "valueStructDotAnonMem0", 33.2);
    checkIntFn(exec, mod, "valueStructDotAnonMem1", 3);
    checkIntFn(exec, mod, "valueStructDotAnonMem2", -2);
    checkIntFn(exec, mod, "valueStructDotAnonMem3", 8);
    checkIntFn(exec, mod, "valueStructDotAnonMem4", 22);
    checkIntFn(exec, mod, "valueStructDotAnonMem5", 22);
    checkFloatFn(exec, mod, "valueStructDotAnonMem6", .3);

    // LLVM weirdness.

    checkIntFn(exec, mod, "getZeroInit", 0);

    exec.dispose();
  }

  @Test public void addressAndIndirection() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        "#include <stddef.h>",

        // Variables and temporaries.

        "int i = 3;",
        "int a[1] = {5};",
        "int *pa = a;",
        "int *pi = &i;", // simple & in const expr
        "int *pa2 = &*a;", // &* collapses in const expr
        "int *pa3 = *&a;", // *& collapses in const expr
        "int *pi2 = &*&*&*&i;", // &*& collapses in const expr

        "int derefGlobalPtr() { return *pa; }", // simple * locally
        "int derefGlobalPtr2() { return *pa2; }", // pa2 init expr ok
        "int derefGlobalPtr3() { return *pa3; }", // pa3 init expr ok
        "int derefGlobalArray() { return *a; }", // * runs array-to-pointer
        "int derefGlobalPtrToAddrOf() { return *pi; }", // pi init expr ok
        "int derefGlobalPtrToAddrOf2() { return *pi2; }", // pi2 init expr ok
        "int *addrOfGlobal() { return &i; }", // simple & locally
        "int derefTmpPtr() { return *addrOfGlobal(); }", // addrGlobal ret expr ok
        "int *addrOfGlobal2() { return &*&*&i; }", // &*& collapses locally
        "int derefTmpPtr2() { return *addrOfGlobal2(); }", // addrGlobal2 ret expr ok
        "int derefAndAddrOfLocal() { int i = 98; int *p = &i; return *p; }",
        "int *getNullPtr() { int *p = 0; return p; }",
        "int *collapseNoLoadFromNull() { int *p = 0; return &*p; }", // &* collapses locally

        // Function designators and pointers.

        "int fn(void) { return 71; }",

        "int derefFnPtr() { int (*p)(void) = fn; return (*p)(); }",
        "int derefFnDesig() { return (*fn)(); }",
        "int addrOfFnDesig() { return (&fn)(); }",
        "int addrOfFnPtr() {int (*p)()=fn; int (**pp)()=&p; return (**pp)();}",

        "int derefFnPtr2() { int (*p)(void) = fn; return (&*p)(); }",
        "int derefFnDesig2() { return (&*fn)(); }",
        "int addrOfFnDesig2() { return (*&fn)(); }",
        "int addrOfFnPtr2() {int (*p)()=fn; int (**pp)()=&*&p; return (**pp)();}",

        "int derefFnPtr3() { int (*p)(void) = fn; return (*&*p)(); }",
        "int derefFnDesig3() { return (*&*fn)(); }",
        "int addrOfFnDesig3() { return (&*&fn)(); }",
        "int addrOfFnPtr3() {int (*p)()=fn; int (**pp)()=&*&*&p; return (**pp)();}",

        // Check that at least a few of those work in constant expressions,
        // which shouldn't matter because indirection/address operators
        // generate no code.
        "int (*pFnGlobal)(void) = *fn;",
        "int (**ppFnGlobal)(void) = &pFnGlobal;",
        "int derefFnDesignGlobal() { return pFnGlobal(); }",
        "int addrOfFnPtrGlobal() { int (*p)() = *ppFnGlobal; return (*p)(); }",

        // sizeof

        "void *pvoid;",
        "size_t sizeofPtr() { return sizeof pvoid; }",
        "size_t sizeofAddrOfInt() { return sizeof &i; }",
        "size_t sizeofAddrOfFnDesig() { return sizeof &fn; }",
        "size_t sizeofDerefPtrToInt() { return sizeof *pi; }",

        // Check that &[] collapses to +.

        "int *arrayAdd = a + 1;",
        "int *addressOfSubscript = &a[1];", // must be a constant expression

        // & with . at file scope. At one time, BuildLLVM failed here with a
        // null pointer exception because it tried to construct an alloca
        // builder (which is sometimes used for accessing union members) using
        // a null pointer instead of a function.

        "struct S { int i; } s = {5};",
        "union U { int i; } u = {6};",
        "int *psi = &s.i;",
        "int *pui = &u.i;",
        "int get_psi() { return *psi; }",
        "int get_pui() { return *pui; }",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    checkIntFn(exec, mod, "derefGlobalPtr", 5);
    checkIntFn(exec, mod, "derefGlobalPtr2", 5);
    checkIntFn(exec, mod, "derefGlobalPtr3", 5);
    checkIntFn(exec, mod, "derefGlobalArray", 5);
    checkIntFn(exec, mod, "derefGlobalPtrToAddrOf", 3);
    checkIntFn(exec, mod, "derefGlobalPtrToAddrOf2", 3);
    checkIntFn(exec, mod, "derefTmpPtr", 3);
    checkIntFn(exec, mod, "derefTmpPtr2", 3);
    checkIntFn(exec, mod, "derefAndAddrOfLocal", 98);
    final LLVMGenericValue nullPtr = runFn(exec, mod, "getNullPtr");
    checkPointerFn(exec, mod, "collapseNoLoadFromNull", nullPtr);

    checkIntFn(exec, mod, "derefFnPtr", 71);
    checkIntFn(exec, mod, "derefFnDesig", 71);
    checkIntFn(exec, mod, "addrOfFnDesig", 71);
    checkIntFn(exec, mod, "addrOfFnPtr", 71);

    checkIntFn(exec, mod, "derefFnPtr2", 71);
    checkIntFn(exec, mod, "derefFnDesig2", 71);
    checkIntFn(exec, mod, "addrOfFnDesig2", 71);
    checkIntFn(exec, mod, "addrOfFnPtr2", 71);

    checkIntFn(exec, mod, "derefFnPtr3", 71);
    checkIntFn(exec, mod, "derefFnDesig3", 71);
    checkIntFn(exec, mod, "addrOfFnDesig3", 71);
    checkIntFn(exec, mod, "addrOfFnPtr3", 71);

    checkIntFn(exec, mod, "derefFnDesignGlobal", 71);
    checkIntFn(exec, mod, "addrOfFnPtrGlobal", 71);

    final long sizeofPtr
      = runFn(exec, mod, "sizeofPtr").toInt(false).longValue();
    checkIntFn(exec, mod, "sizeofAddrOfInt", sizeofPtr);
    checkIntFn(exec, mod, "sizeofAddrOfFnDesig", sizeofPtr);
    checkIntFn(exec, mod, "sizeofDerefPtrToInt", SrcIntType.getLLVMWidth()/8);

    final LLVMConstant arrayAdd = checkGlobalVarGetInit(mod, "arrayAdd");
    final LLVMConstant addressOfSubscript
      = checkGlobalVarGetInit(mod, "addressOfSubscript");
    assertEquals("addressOfSubscript", arrayAdd, addressOfSubscript);

    checkIntFn(exec, mod, "get_psi", 5);
    checkIntFn(exec, mod, "get_pui", 6);

    exec.dispose();
  }

  @Test public void unaryArithmeticOperators() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final BigInteger intAllOnes
      = BigInteger.ONE.shiftLeft((int)SrcIntType.getWidth())
       .subtract(BigInteger.ONE);
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        "#include <stddef.h>",

        // integer operand types, constant expressions
        "int plusInt   = +'*';    size_t plusIntSize    = sizeof +'*';",
        "int minusInt  = -1;      size_t minusIntSize   = sizeof -1;",
        "int complInt  = ~'\\5';  size_t complIntSize   = sizeof ~'\\5';",
        "int notInt0   = !'\\0';  size_t notInt0Size    = sizeof !'\\0';",
        "int notIntPos = !'\\1';  size_t notIntPosSize  = sizeof !'\\1';",
        "int notIntNeg = !-'\\7'; size_t notIntNegSize  = sizeof !-'\\7';",

        // floating operand types, constant expressions
        "float plusFloat    = +3.5f;  size_t plusFloatSize    = sizeof +3.5f;",
        "double minusDouble = -28e7;  size_t minusDoubleSize  = sizeof -28e7;",
        "int notFloat0      = !0.f;   size_t notFloat0Size    = sizeof !0.f;",
        "int notDoublePos   = !1.;    size_t notDoublePosSize = sizeof !1.;",
        "int notFloatNeg    = !-3e-1; size_t notFloatNegSize  = sizeof !-3e-1;",

        // pointer operand types, constant expressions
        "int notPtrNull    = !0;          size_t notPtrNullSize    = sizeof !0;",
        "int notPtrNonNull = !&plusFloat; size_t notPtrNonNullSize = sizeof !&plusFloat;",

        // integer operand types, non-constant expressions
        "#define PLUS_INT(S)    {char o='*'; return S +o;}",
        "#define MINUS_INT(S)   {int  o=8;   return S -o;}",
        "#define COMPL_INT(S)   {char o=10;  return S ~o;}",
        "#define NOT_INT0(S)    {char o=0;   return S !o;}",
        "#define NOT_INT_POS(S) {char o=1;   return S !o;}",
        "#define NOT_INT_NEG(S) {int  o=-9;  return S !o;}",
        "int plusIntL()   PLUS_INT()    size_t plusIntSizeL()   PLUS_INT(sizeof)",
        "int minusIntL()  MINUS_INT()   size_t minusIntSizeL()  MINUS_INT(sizeof)",
        "int complIntL()  COMPL_INT()   size_t complIntSizeL()  COMPL_INT(sizeof)",
        "int notInt0L()   NOT_INT0()    size_t notInt0SizeL()   NOT_INT0(sizeof)",
        "int notIntPosL() NOT_INT_POS() size_t notIntPosSizeL() NOT_INT_POS(sizeof)",
        "int notIntNegL() NOT_INT_NEG() size_t notIntNegSizeL() NOT_INT_NEG(sizeof)",

        // floating operand types, non-constant expressions
        "#define PLUS_DOUBLE(S)    {double o=-35e-8; return S +o;}",
        "#define MINUS_FLOAT(S)    {float  o=-2.1f;  return S -o;}",
        "#define NOT_DOUBLE0(S)    {double o=0.0;    return S !o;}",
        "#define NOT_FLOAT_POS(S)  {float  o=89e2f;  return S !o;}",
        "#define NOT_DOUBLE_NEG(S) {double o=-5.3e2; return S !o;}",
        "double plusDoubleL()   PLUS_DOUBLE()    size_t plusDoubleSizeL()   PLUS_DOUBLE(sizeof)",
        "float  minusFloatL()   MINUS_FLOAT()    size_t minusFloatSizeL()   MINUS_FLOAT(sizeof)",
        "int    notDouble0L()   NOT_DOUBLE0()    size_t notDouble0SizeL()   NOT_DOUBLE0(sizeof)",
        "int    notFloatPosL()  NOT_FLOAT_POS()  size_t notFloatPosSizeL()  NOT_FLOAT_POS(sizeof)",
        "int    notDoubleNegL() NOT_DOUBLE_NEG() size_t notDoubleNegSizeL() NOT_DOUBLE_NEG(sizeof)",

        // pointer operand types, non-constant expressions
        "#define NOT_PTR_NULL(S)     {void *o=0;        return S !o;}",
        "#define NOT_PTR_NON_NULL(S) {void *o=&plusInt; return S !o;}",
        "int notPtrNullL()    NOT_PTR_NULL()     size_t notPtrNullSizeL()    NOT_PTR_NULL(sizeof)",
        "int notPtrNonNullL() NOT_PTR_NON_NULL() size_t notPtrNonNullSizeL() NOT_PTR_NON_NULL(sizeof)",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final long intSize = SrcIntType.getWidth()/8;
    final long floatSize = SrcFloatType.getLLVMType(ctxt).getPrimitiveSizeInBits()/8;
    final long doubleSize = SrcDoubleType.getLLVMType(ctxt).getPrimitiveSizeInBits()/8;
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    checkIntGlobalVar(mod, "plusInt", 42);
    checkIntGlobalVar(mod, "minusInt", -1);
    checkIntGlobalVar(mod, "complInt",
                      intAllOnes.subtract(BigInteger.valueOf(5)).longValue());
    checkIntGlobalVar(mod, "notInt0", 1);
    checkIntGlobalVar(mod, "notIntPos", 0);
    checkIntGlobalVar(mod, "notIntNeg", 0);

    checkIntGlobalVar(mod, "plusIntSize", intSize);
    checkIntGlobalVar(mod, "minusIntSize", intSize);
    checkIntGlobalVar(mod, "complIntSize", intSize);
    checkIntGlobalVar(mod, "notInt0Size", intSize);
    checkIntGlobalVar(mod, "notIntPosSize", intSize);
    checkIntGlobalVar(mod, "notIntNegSize", intSize);

    checkFloatGlobalVar(mod, "plusFloat", 3.5);
    checkDoubleGlobalVar(mod, "minusDouble", -28e7);
    checkIntGlobalVar(mod, "notFloat0", 1);
    checkIntGlobalVar(mod, "notDoublePos", 0);
    checkIntGlobalVar(mod, "notFloatNeg", 0);

    checkIntGlobalVar(mod, "plusFloatSize", floatSize);
    checkIntGlobalVar(mod, "minusDoubleSize", doubleSize);
    checkIntGlobalVar(mod, "notFloat0Size", intSize);
    checkIntGlobalVar(mod, "notDoublePosSize", intSize);
    checkIntGlobalVar(mod, "notFloatNegSize", intSize);

    checkIntGlobalVar(mod, "notPtrNull", 1);
    checkIntGlobalVar(mod, "notPtrNonNull", 0);

    checkIntGlobalVar(mod, "notPtrNullSize", intSize);
    checkIntGlobalVar(mod, "notPtrNonNullSize", intSize);

    checkIntFn(exec, mod, "plusIntL", 42);
    checkIntFn(exec, mod, "minusIntL", -8);
    checkIntFn(exec, mod, "complIntL",
               intAllOnes.subtract(BigInteger.valueOf(10)).longValue());
    checkIntFn(exec, mod, "notInt0L", 1);
    checkIntFn(exec, mod, "notIntPosL", 0);
    checkIntFn(exec, mod, "notIntNegL", 0);

    checkIntFn(exec, mod, "plusIntSizeL", intSize);
    checkIntFn(exec, mod, "minusIntSizeL", intSize);
    checkIntFn(exec, mod, "complIntSizeL", intSize);
    checkIntFn(exec, mod, "notInt0SizeL", intSize);
    checkIntFn(exec, mod, "notIntPosSizeL", intSize);
    checkIntFn(exec, mod, "notIntNegSizeL", intSize);

    checkDoubleFn(exec, mod, "plusDoubleL", -35e-8);
    checkFloatFn(exec, mod, "minusFloatL", 2.1);
    checkIntFn(exec, mod, "notDouble0L", 1);
    checkIntFn(exec, mod, "notFloatPosL", 0);
    checkIntFn(exec, mod, "notDoubleNegL", 0);

    checkIntFn(exec, mod, "plusDoubleSizeL", doubleSize);
    checkIntFn(exec, mod, "minusFloatSizeL", floatSize);
    checkIntFn(exec, mod, "notDouble0SizeL", intSize);
    checkIntFn(exec, mod, "notFloatPosSizeL", intSize);
    checkIntFn(exec, mod, "notDoubleNegSizeL", intSize);

    checkIntFn(exec, mod, "notPtrNullL", 1);
    checkIntFn(exec, mod, "notPtrNonNullL", 0);

    checkIntFn(exec, mod, "notPtrNullSizeL", intSize);
    checkIntFn(exec, mod, "notPtrNonNullSizeL", intSize);

    exec.dispose();
  }

  private String genSizeofVars(String name) {
    final String type = name.replace("_", " ");
    return genSizeofVars(type, name, "");
  }
  private String genSizeofVars(String type_pre, String name, String type_post) {
    return type_pre + " " + name + "_var" + type_post + ";"
           + " size_t " + name + "_expr = sizeof " + name + "_var;"
           + " size_t " + name + "_type = sizeof (" + type_pre + type_post + ");";
  }
  private void checkSizeofVars(LLVMModule mod, String name, long size) {
    checkIntGlobalVar(mod, name + "_expr", size);
    checkIntGlobalVar(mod, name + "_type", size);
  }

  /**
   * sizeof for an expression operand is mostly tested where the kind of
   * expression is otherwise tested (other test methods in this class). Here
   * we check sizeof for type operands and for a few special cases, such as
   * those those mentioned in ISO C99 sec. 6.5.3.4, whether or not they're
   * also tested elsewhere.
   */
  @Test public void sizeofOperator() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        "#include <stddef.h>",

        // ISO C99 sec. 6.5.3.4p3 mentions these.

        genSizeofVars("char"),
        genSizeofVars("unsigned_char"),
        genSizeofVars("signed_char"),
        genSizeofVars("int", "array", "[2]"),

        "size_t ptr() { void *p; return sizeof p; }",
        // a/fn is ptr not array/function
        "size_t arrayParam_(double a[10000]) { return sizeof a; }",
        "size_t arrayParam() { return arrayParam_(0); }",
        "size_t fnParam_(double fn()) { return sizeof fn; }",
        "size_t fnParam() { return fnParam_(0); }",

        "struct S {double d; float i;};",
        genSizeofVars("struct_S"),
        "union U {float i; double d;};",
        genSizeofVars("union_U"),

        "size_t derefPtrToDouble() { double *p; return sizeof *p; }",
        "size_t arrayElements() { int a[5]; return sizeof a / sizeof a[5]; }",

        // Check various type list forms that Cetus builds for a
        // SizeofExpression node out of a sizeof type operand. For
        // completeness, check the corresponding expression operands as well.

        genSizeofVars("long int const", "simpleSpecs", ""),
        genSizeofVars("int *", "ptrSpec1", ""),
        genSizeofVars("char const **", "ptrSpec2", ""),
        genSizeofVars("void *", "vd_ptrArray", "[5]"),
        genSizeofVars("int (*", "nd_arrayPtr", ")[5]"),
        genSizeofVars("void (*", "nd_fnPtr", ")()"),

        // Make sure sizeof sizeof works.
        "size_t cSizeofSizeofInt = sizeof sizeof 3;",
        "size_t sizeofSizeofInt() { void *p; return sizeof sizeof p; }",

        // Make sure BuildLLVM can handle a sizeof expression containing an
        // array dimension expression.  The former requires computing sizes
        // of expressions, but the latter requires computing values of
        // expressions, so exprMode has to push and pop at the latter.

        "#include <stddef.h>",
        "size_t arrayDimExpr = sizeof ((int(*)[3])0);",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final long intSize = SrcIntType.getWidth()/8;
    final long longSize = SrcLongType.getWidth()/8;
    final long doubleSize = SrcDoubleType.getLLVMType(ctxt)
                            .getPrimitiveSizeInBits()/8;
    final long sizeTSize = SRC_SIZE_T_TYPE.getLLVMType(ctxt)
                           .getPrimitiveSizeInBits()/8;
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    final long ptrSize = runFn(exec, mod, "ptr").toInt(false).longValue();

    // ISO C99 sec. 6.5.3.4p3 mentions these, and it specifies 1 explicitly
    // for the char types.
    checkSizeofVars(mod, "char", 1);
    checkSizeofVars(mod, "unsigned_char", 1);
    checkSizeofVars(mod, "signed_char", 1);
    checkSizeofVars(mod, "array", intSize*2);
    checkIntFn(exec, mod, "arrayParam", ptrSize);
    checkIntFn(exec, mod, "fnParam", ptrSize);
    checkSizeofVars(mod, "struct_S", 2*doubleSize);
    checkSizeofVars(mod, "union_U", doubleSize);
    checkIntFn(exec, mod, "derefPtrToDouble", doubleSize);
    checkIntFn(exec, mod, "arrayElements", 5);

    // Various type list forms that Cetus builds for a SizeofExpression node
    // out of a sizeof type operand.
    checkSizeofVars(mod, "simpleSpecs", longSize);
    checkSizeofVars(mod, "ptrSpec1", ptrSize);
    checkSizeofVars(mod, "ptrSpec2", ptrSize);
    checkSizeofVars(mod, "vd_ptrArray", 5*ptrSize);
    checkSizeofVars(mod, "nd_arrayPtr", ptrSize);
    checkSizeofVars(mod, "nd_fnPtr", ptrSize);

    checkIntGlobalVar(mod, "cSizeofSizeofInt", sizeTSize);
    checkIntFn(exec, mod, "sizeofSizeofInt", sizeTSize);

    checkIntGlobalVar(mod, "arrayDimExpr", ptrSize);

    exec.dispose();
  }

  @Test public void offsetofOperator() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        "#include <stddef.h>",
        "struct S { struct {int i; int j;} f1; int f2; int f3; } s;",
        "union U { struct {int i; int j;} f1; int f2; int f3; } u;",
        "struct SA {",
        "  double f1;",
        "  struct { int f2; struct { char f3; }; unsigned f4; };",
        "  union { int f5; int f6; };",
        "  float f7;",
        "} sa;",

        "size_t sizeof_S_f1 = sizeof(offsetof(struct S, f1));",
        "size_t sizeof_S_f2 = sizeof(offsetof(struct S, f2));",
        "size_t sizeof_S_f3 = sizeof(offsetof(struct S, f3));",
        "size_t sizeof_U_f1 = sizeof(offsetof(union U, f1));",
        "size_t sizeof_U_f2 = sizeof(offsetof(union U, f2));",
        "size_t sizeof_U_f3 = sizeof(offsetof(union U, f3));",

        "size_t S_f1 = offsetof(struct S, f1);",
        "size_t S_f2 = offsetof(struct S, f2);",
        "size_t U_f1 = offsetof(union U, f1);",
        "size_t U_f2 = offsetof(union U, f2);",
        "size_t U_f3 = offsetof(union U, f3);",
        "size_t SA_f1 = offsetof(struct SA, f1);",
        "size_t SA_f3 = offsetof(struct SA, f3);",
        "size_t SA_f6 = offsetof(struct SA, f6);",

        "size_t s_f2 = ((char*)&s.f2 - (char*)&s)/sizeof(char);",
        "size_t s_f3 = ((char*)&s.f3 - (char*)&s)/sizeof(char);",
        "size_t sa_f3 = ((char*)&sa.f3 - (char*)&sa)/sizeof(char);",
        "size_t sa_f6 = ((char*)&sa.f6 - (char*)&sa)/sizeof(char);",
        "int compare_S_f2() { return S_f2 == s_f2; }",
        "int compare_S_f3() {",
        "  return offsetof(struct S, f3) == s_f3;",
        "}",
        "int compare_SA_f3() { return SA_f3 == sa_f3; }",
        "int compare_SA_f6() { return SA_f6 == sa_f6; }",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMIntegerType sizeT = SRC_SIZE_T_TYPE.getLLVMType(ctxt);
    final long sizeTSize = SRC_SIZE_T_TYPE.getLLVMWidth()/8;
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);

    checkGlobalVar(mod, "sizeof_S_f1",
                   LLVMConstantInteger.get(sizeT, sizeTSize, false));
    checkGlobalVar(mod, "sizeof_S_f2",
                   LLVMConstantInteger.get(sizeT, sizeTSize, false));
    checkGlobalVar(mod, "sizeof_S_f3",
                   LLVMConstantInteger.get(sizeT, sizeTSize, false));
    checkGlobalVar(mod, "sizeof_U_f1",
                   LLVMConstantInteger.get(sizeT, sizeTSize, false));
    checkGlobalVar(mod, "sizeof_U_f2",
                   LLVMConstantInteger.get(sizeT, sizeTSize, false));
    checkGlobalVar(mod, "sizeof_U_f3",
                   LLVMConstantInteger.get(sizeT, sizeTSize, false));

    checkGlobalVar(mod, "S_f1", LLVMConstant.constNull(sizeT));
    checkGlobalVar(mod, "U_f1", LLVMConstant.constNull(sizeT));
    checkGlobalVar(mod, "U_f2", LLVMConstant.constNull(sizeT));
    checkGlobalVar(mod, "U_f3", LLVMConstant.constNull(sizeT));
    checkGlobalVar(mod, "SA_f1", LLVMConstant.constNull(sizeT));

    checkIntFn(exec, mod, 1, "compare_S_f2");
    checkIntFn(exec, mod, 1, "compare_S_f3");
    checkIntFn(exec, mod, 1, "compare_SA_f3");
    checkIntFn(exec, mod, 1, "compare_SA_f6");

    exec.dispose();
  }


  /**
   * Type conversions are checked thoroughly in
   * {@link BuildLLVMTest_TypeConversions}, so here we focus on type
   * conversions that are unique to explicit casts (conversion to void) and
   * the various type list forms that Cetus builds for a
   * {@link cetus.hir.Typecast} node.
   */
  @Test public void explicitCast() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        "#include <stddef.h>",

        // memcpy is provided by the native target.
        "void *memcpy(void *, void*, size_t);",

        "int voidCasts() {",
        "  struct S {int i;} s;",
        "  int a[3];",
        "  void fn();",
        // Here, the goal is just to make sure the expressions compile and
        // don't crash anything.
        "  (void)3;", // simple cast to void
        "  (void)s;", // struct can't be cast to anything but void
        "  (void)a;", // array can't be cast to anything but void
        "  (void)fn;", // function can't be cast to anything but void
        "  (void)(void)fn;", // void can't be cast to anything but void
        // Now we check that void cast doesn't suppress evaluation and thus
        // side effects.
        "  int i = 99, j = 88;",
        "  (void)memcpy(&i, &j, sizeof i);",
        "  return i;",
        "}",

        // Quick check that sizeof works on cast operator.
        "size_t ptrSize() { void *p; return sizeof p; }",
        "size_t cSizeofCharCast = sizeof ((char)voidCasts);",
        "size_t sizeofPtrCast() { char c; return sizeof ((void*)c); }",

        // Various type list forms that Cetus builds for a Typecast node.
        "double volatile cSimpleSpec = (volatile double)3;",
        "double volatile simpleSpec() { int i = 5; return (volatile double)i; }",
        "size_t (*ptr())(void) { return ptrSize; }",
        "void *cPtrSpec1_ = (void*)ptrSize;",
        "void *cPtrSpec1() { return cPtrSpec1_; }",
        "void *ptrSpec1() { long long l = (long long)ptr(); return (void*)l; }",
        "void **cPtrSpec2_ = (void**)&cPtrSpec1_;",
        "void *cPtrSpec2() { return *cPtrSpec2_; }",
        "void *ptrSpec2() { void *p = (void*)&cPtrSpec1_; return *(void**)p; }",
        "int intArr[5] = {99, 88, 77, 66, 55};",
        "int (*cNd_arrayPtr_)[3] = (int(*)[3])intArr;",
        "int cNd_arrayPtr() { return **cNd_arrayPtr_; }",
        "int nd_arrayPtr() { void *p = (void*)intArr; return **(int(*)[2])p; }",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final long charSize = SrcCharType.getWidth()/8;
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    final long ptrSize = runFn(exec, mod, "ptrSize").toInt(false).longValue();

    checkIntFn(exec, mod, "voidCasts", 88);

    checkIntGlobalVar(mod, "cSizeofCharCast", charSize);
    checkIntFn(exec, mod, "sizeofPtrCast", ptrSize);
    checkDoubleGlobalVar(mod, "cSimpleSpec", 3.);
    checkDoubleFn(exec, mod, "simpleSpec", 5.);
    final LLVMGenericValue ptr = runFn(exec, mod, "ptr");
    checkPointerFn(exec, mod, "cPtrSpec1", ptr);
    checkPointerFn(exec, mod, "ptrSpec1", ptr);
    checkPointerFn(exec, mod, "cPtrSpec2", ptr);
    checkPointerFn(exec, mod, "ptrSpec2", ptr);
    checkIntFn(exec, mod, "cNd_arrayPtr", 99);
    checkIntFn(exec, mod, "nd_arrayPtr", 99);

    exec.dispose();
  }

  /**
   * The usual arithmetic conversions, which are performed by many binary
   * operators, are checked thoroughly in
   * {@link BuildLLVMTest_TypeConversions#usualArithmeticConversions} for the
   * case of multiplication. Here we just briefly check that the various
   * binary operators that are supposed to perform the usual arithmetic
   * conversions actually do so (that is, we can mix operand types without
   * LLVM complaints or incorrect values), that constant and non-constant
   * expressions are handled, that integer and floating and pointer operations
   * are handled, that sizeof works, and that short-circuiting works.
   */
  @Test public void binaryOperators() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final BigInteger intMax
      = BigInteger.ONE.shiftLeft((int)SrcIntType.getPosWidth())
        .subtract(BigInteger.ONE);
    final BigInteger unsignedMax
      = BigInteger.ONE.shiftLeft((int)SrcUnsignedIntType.getWidth())
        .subtract(BigInteger.ONE);
    final BigInteger longMax
      = BigInteger.ONE.shiftLeft((int)SrcLongType.getPosWidth())
        .subtract(BigInteger.ONE);
    final SimpleResult simpleResult = buildLLVMSimple(
      "", TARGET_DATA_LAYOUT,
      new String[]{
        "#include <stddef.h>",
        "int a10i[10] = {0, 11, 22, 33, 44, 55, 66, 77, 88, 99};",
        "double a10d[10] = {0,    11.1, 22.2, 33.3, 44.4,\n",
        "                   55.5, 66.6, 77.7, 88.8, 99.9};",
        "struct S { float f0; float f1; float f2; } s;",
        "size_t pSize() { void *p; return sizeof p; }",

        // multiplicative and additive operators for non-pointer types
        genBinaryArith("*", "i",   "c",    "-3",    "5"),
        genBinaryArith("*", "i",   "d",    "2",     "-8."),
        genBinaryArith("/", "i",   "u",    "7",     "3"),
        genBinaryArith("/", "i",   "f",    "25",    "-8."),
        genBinaryArith("%", "c",   "l",    "8",     "3"),
        genBinaryArith("%", "uc",  "i",    "27",    "-8."),
        genBinaryArith("+", "i",   "i",    "27",    "-8"),
        genBinaryArith("+", "ld",  "i",    "-89.",   "5"),
        genBinaryArith("-", "l",   "c",    "398",    "90"),
        genBinaryArith("-", "d",   "f",    "34.2e5", "-90e4"),

        // additive operators for pointer types
        //
        // ISO C99 says we should be able to point one past the end of an
        // array, so we make sure to exercise that case here for each operator.
        genBinaryPtr(  "+", "pi", "pi", "i",  "a10i",    "5"),
        genBinaryPtr(  "+", "pi", "pi", "c",  "a10i+5",  "-2"),
        genBinaryPtr(  "+", "pi", "b",  "pi", "1",       "a10i"),
        genBinaryPtr(  "+", "pi", "l",  "pi", "-3",      "a10i+10"), // end+1
        genBinaryPtr(  "+", "pd", "l",  "pd", "-3",      "a10d+5"), // end+1
        genBinaryPtr(  "-", "pi", "pi", "ul", "a10i+10", "4"),       // end+1
        genBinaryPtr(  "-", "pi", "pi", "l",  "a10i+5",  "-4"),
        genBinaryArith("-",       "pi", "pi", "a10i+10", "a10i+8"),  // end+1
        genBinaryArith("-",       "pd", "pd", "a10d+2",  "a10d+8"),

        // ISO C99 sec. 6.5.5p6 says the result here should always be a.
        "int divMultAddRem(int a, int b) { return (a/b)*b + a%b; }",

        // left shift operator, signed after promotion
        genBinaryArith("<<", "i",   "i",  "5",   "3"),
        genBinaryArith("<<", "b",   "b",  "1",   "1"),
        genBinaryArith("<<", "i",   "c",  "10",  "2"),
        genBinaryArith("<<", "c",   "u",  "-38", "5"),
        genBinaryArith("<<", "l",   "c",  "9",   "5"),
        genBinaryArith("<<", "i",   "l",  "-4",  "9"),
        genBinaryArith("<<", "l",   "i",
                       String.valueOf(longMax.longValue()/2),
                       "1"),

        // left shift operator, unsigned after promotion
        genBinaryArith("<<", "u",   "i",  "32",  "4"),
        genBinaryArith("<<", "ul",  "ul", "3",   "2"),

        // right shift operator, signed after promotion
        genBinaryArith(">>", "i",   "i",  "5",   "2"),
        genBinaryArith(">>", "sc",  "sc", "-8",  "2"),
        genBinaryArith(">>", "uc",  "b",  "4",   "1"),
        genBinaryArith(">>", "ll",  "u",  "-52", "2"), // op1 not op2 => ashr not lshr
        genBinaryArith(">>", "l",   "l",  "-6",  "2"),

        // right shift operator, unsigned after promotion
        genBinaryArith(">>", "u",   "sc", "9",   "2"),
        genBinaryArith(">>", "ull", "u",  "88",  "3"),
        genBinaryArith(">>", "u",   "i",  "-1",  "1"), // op1 not op2 => lshr not ashr

        // relational operators for integer types
        genBinaryArith("<",  "u",   "c",  unsignedMax.toString(), "5"),
        genBinaryArith("<",  "i",   "uc", "-1",  "20"),
        genBinaryArith("<",  "i",   "i",  "-3",  "-3"),
        genBinaryArith("<",  "u",   "u",  "5",   "5"),
        genBinaryArith(">",  "b",   "u",  "0",   unsignedMax.toString()),
        genBinaryArith(">",  "u",   "l",  "2",   "-5"),
        genBinaryArith(">",  "l",   "i",  "2",   "2"),
        genBinaryArith(">",  "ull", "c",  "0",   "0"),
        genBinaryArith("<=", "u",   "c",  unsignedMax.toString(), "5"),
        genBinaryArith("<=", "i",   "uc", "-1",  "20"),
        genBinaryArith("<=", "i",   "i",  "-3",  "-3"),
        genBinaryArith("<=", "u",   "u",  "5",   "5"),
        genBinaryArith(">=", "b",   "u",  "0",   unsignedMax.toString()),
        genBinaryArith(">=", "u",   "l",  "2",   "-5"),
        genBinaryArith(">=", "l",   "i",  "2",   "2"),
        genBinaryArith(">=", "ull", "c",  "0",   "0"),

        // relational operators for floating types
        genBinaryArith("<",  "f",   "c",  "0.5",  "1"),
        genBinaryArith("<",  "f",   "d",  "5",    "5"),
        genBinaryArith("<",  "i",   "f",  "8",    "5"),
        genBinaryArith(">",  "u",   "ld", "1",    "-1e5"),
        genBinaryArith(">",  "d",   "d",  "0",    "0"),
        genBinaryArith(">",  "i",   "d",  "-5",   "0"),
        genBinaryArith("<=", "i",   "d",  "-4",   "1e5"),
        genBinaryArith("<=", "ld",  "d",  "3.5",  "3.5"),
        genBinaryArith("<=", "d",   "d",  "3.5",  "3.4"),
        genBinaryArith(">=", "b",   "f",  "1",    "-1e5"),
        genBinaryArith(">=", "d",   "i",  "1e1",  "10"),
        genBinaryArith(">=", "i",   "f",  "1",    "1.1"),

        // relational operators for pointer types
        //
        // ISO C99 says we should be able to point one past the end of an
        // array, so we make sure to exercise that case here for each operator.
        genBinaryArith("<",  "pi", "pi", "a10i+10", "a10i+8"),  // end+1
        genBinaryArith("<",  "pf", "pf", "&s.f1",   "&s.f2"),
        genBinaryArith("<",  "pd", "pd", "a10d+1",  "a10d+1"),
        genBinaryArith(">",  "pi", "pi", "a10i+10", "a10i+8"),  // end+1
        genBinaryArith(">",  "pf", "pf", "&s.f1",   "&s.f2"),
        genBinaryArith(">",  "pd", "pd", "a10d+1",  "a10d+1"),
        genBinaryArith("<=", "pi", "pi", "a10i+10", "a10i+8"),  // end+1
        genBinaryArith("<=", "pf", "pf", "&s.f1",   "&s.f2"),
        genBinaryArith("<=", "pd", "pd", "a10d+1",  "a10d+1"),
        genBinaryArith(">=", "pi", "pi", "a10i+10", "a10i+8"),  // end+1
        genBinaryArith(">=", "pf", "pf", "&s.f1",   "&s.f2"),
        genBinaryArith(">=", "pd", "pd", "a10d+1",  "a10d+1"),

        // equality operators for integer types
        genBinaryArith("==", "i",  "c",  "-3",  "5"),
        genBinaryArith("==", "u",  "b",  "1",   "1"),
        genBinaryArith("==", "l",  "i",  "100", "-1"),
        genBinaryArith("!=", "sc", "i",  "0",   "5"),
        genBinaryArith("!=", "u",  "lu", "5",   "5"),
        genBinaryArith("!=", "l",  "c",  "-2",  "-10"),

        // equality operators for floating types
        genBinaryArith("==", "f",   "c",  "0.5",  "1"),
        genBinaryArith("==", "f",   "d",  "5",    "5"),
        genBinaryArith("==", "i",   "f",  "8",    "5"),
        genBinaryArith("!=", "i",   "d",  "-4",   "1e5"),
        genBinaryArith("!=", "ld",  "d",  "3.5",  "3.5"),
        genBinaryArith("!=", "d",   "d",  "3.5",  "3.4"),

        // equality operators for pointer types not involving void pointer
        //
        // ISO C99 says we should be able to point one past the end of an
        // array, so we make sure to exercise that case here for each operator.
        genBinaryArith("==", "pi", "pi", "a10i+10", "a10i+8"),  // end+1
        genBinaryArith("==", "pf", "pf", "&s.f1",   "&s.f2"),
        genBinaryArith("==", "pd", "pd", "a10d+1",  "a10d+1"),
        genBinaryArith("!=", "pi", "pi", "a10i+10", "a10i+8"),  // end+1
        genBinaryArith("!=", "pf", "pf", "&s.f1",   "&s.f2"),
        genBinaryArith("!=", "pd", "pd", "a10d+1",  "a10d+1"),
        "void fn1() {}",
        "void fn2() {}",
        "int fn1_eq_fn2 = fn1 == fn2;",
        "int fn1_eq_fn1 = fn1 == fn1;",
        "int fn1_ne_fn2() {",
        "  void (*p1)() = fn1; void(*p2)() = fn2;",
        "  return p1 != p2;",
        "}",
        "int fn2_ne_fn2 = fn2 != fn2;",

        // equality operators for pointer types involving void pointer
        genBinaryArith("==", "pi", "pv", "a10i+10", "a10i+8"),  // end+1
        genBinaryArith("==", "pv", "pf", "&s.f1",   "&s.f2"),
        genBinaryArith("==", "pv", "pv", "a10d+1",  "a10d+1"),
        genBinaryArith("!=", "pv", "pi", "a10i+10", "a10i+8"),  // end+1
        genBinaryArith("!=", "pv", "pv", "&s.f1",   "&s.f2"),
        genBinaryArith("!=", "pd", "pv", "a10d+1",  "a10d+1"),

        // equals operator for pointer types and null pointer constants
        "int c_inpc_eq_inpc = 0 == 0;",
        "int c_vnpc_eq_vnpc = (void*)0 == (void*)0;",
        "int c_vnpc_eq_inpc = (void*)0 == 0;",
        "int c_inpc_eq_vnpc = 0 == (void*)0;",
        "int c_pi_eq_inpc = a10i == 0;",
        "int c_vnpc_eq_pd = (void*)0 == a10d;",
        "int c_fn1_eq_vnpc = fn1 == (void*)0;",
        "int c_inpc_eq_fn2 = (void*)0 == fn2;",

        // not-equals operator for pointer types and null pointer constants
        "int c_inpc_ne_inpc = 0 != 0;",
        "int c_vnpc_ne_vnpc = (void*)0 != (void*)0;",
        "int c_vnpc_ne_inpc = (void*)0 != 0;",
        "int c_inpc_ne_vnpc = 0 != (void*)0;",
        "int c_pi_ne_inpc = a10i != 0;",
        "int c_vnpc_ne_pd = (void*)0 != a10d;",
        "int c_fn1_ne_vnpc = fn1 != (void*)0;",
        "int c_inpc_ne_fn2 = (void*)0 != fn2;",

        // non-constant-expression equality operators for pointer types and
        // null pointer constants
        "int vnpc_eq_inpc() { void *p = 0; return p == 0; }",
        "int inpc_ne_fn2() { void (*p)() = fn2; return (void*)0 != p; }",

        // bitwise &, ^, and |
        genBinaryArith("&",  "i",   "i",   "5",   "3"),
        genBinaryArith("&",  "i",   "u",   "-1",  "10"),
        genBinaryArith("&",  "l",   "u",   "13",  "11"),
        genBinaryArith("&",  "b",   "l",   "1",   "-2"),
        genBinaryArith("^",  "i",   "uc",  "5",   "3"),
        genBinaryArith("^",  "c",   "i",   "-1",  "10"),
        genBinaryArith("^",  "c",   "u",   "-1",  "1"),
        genBinaryArith("|",  "uc",  "ull", "13",  "11"),
        genBinaryArith("|",  "uc",  "b",   "2",   "1"),

        // && and || with constant expressions
        genBinaryArith("&&", "i",   "i",   "2",   "1"),
        genBinaryArith("&&", "i",   "c",   "5",   "0"),
        genBinaryArith("&&", "u",   "l",   "0",   "-1"),
        genBinaryArith("&&", "b",   "b",   "0",   "0"),
        genBinaryArith("&&", "f",   "pi",  "3.5", "a10i"),
        genBinaryArith("&&", "pv",  "d",   "0",   "-10.5"),
        genBinaryArith("&&", "d",   "d",   "0.0", "1e10"),
        genBinaryArith("||", "l",   "b",   "-9",  "1"),
        genBinaryArith("||", "ll",  "i",   "-3",  "0"),
        genBinaryArith("||", "ll",  "ll",  "0",   "1"),
        genBinaryArith("||", "u",   "u",   "0",   "0"),
        genBinaryArith("||", "f",   "pi",  "0.0", "a10i"),
        genBinaryArith("||", "pv",  "d",   "0",   "-10.5"),
        genBinaryArith("||", "pv",  "f",   "0",   "0.0"),
        "int logicalAndStaticLocal(int new) {",
        "  static int i = 5 && 0;", // must be a constant expression
        "  int old = i;",
        "  i = new;",
        "  return old;",
        "}",
        "int logicalOrStaticLocal(int new) {",
        "  static int i = 5 || 2;", // must be a constant expression
        "  int old = i;",
        "  i = new;",
        "  return old;",
        "}",

        // && and || with non-constant expression => potential short-circuiting
        "int sideEffect;",
        "int getSideEffect() { return sideEffect; }",
        "int getSetSideEffect(int new) {",
        "  int old = sideEffect;",
        "  sideEffect = new;",
        "  return old;",
        "}",
        "int logicalAndLocal(int first, int second) {",
        "  getSetSideEffect(first);",
        "  return getSetSideEffect(second) && getSetSideEffect(99);",
        "}",
        "int logicalOrLocal(int first, int second) {",
        "  getSetSideEffect(first);",
        "  return getSetSideEffect(second) || getSetSideEffect(99);",
        "}",

        // && and || with multiple basic blocks in operands.
        "int logicalAndMultipleBBs(int i, int j, int k, int l) {",
        "  return (i && j) && (k || l);",
        "}",
        "int logicalOrMultipleBBs(int i, int j, int k, int l) {",
        "  return (i && j) || (k || l);",
        "}",

        // []

        "size_t cs_subscript = sizeof a10i[5];",
        "int *cva_subscript = &a10i[5];",
        "int cv_subscript() { return *cva_subscript; }",
        "size_t s_subscript() {int i; int *p; return sizeof p[i];}",
        "int v_subscript() {int i=-1; int *p=a10i+10; return p[i];}",

        "size_t cs_subscript_rev = sizeof (-2)[a10d+5];",
        "double *cva_subscript_rev = &(-2)[a10d+5];",
        "double cv_subscript_rev() { return *cva_subscript_rev; }",
        "size_t s_subscript_rev() {char i; double *p; return sizeof i[p];}",
        "double v_subscript_rev() {char i=4; double *p=a10d; return i[p];}",

        "int aai[3][2] = {{0, 1}, {10, 11}, {20, 22}};",
        "int v_subsubscript(int i, int j) { return aai[i][j]; }",
        "int v_subsubscript_rev(int i, int j) { return i[aai][j]; }",
        "int v_subsubscript_revrev(int i, int j) { return j[i[aai]]; }",
        "size_t s_subsubscript() { return sizeof aai[1][1]; }",
        "size_t s_subsubscript_rev() { return sizeof 2[aai][0]; }",
        "size_t s_subsubscript_revrev() { return sizeof 0[1[aai]]; }",
        "size_t s_aai_sub() { return sizeof aai[0]; }",
        "size_t s_aai() { return sizeof aai; }",

        // This used to a problem because BuildLLVM attempted type conversions
        // for the sub-array's initializer but saw it as a string constant
        // initializing a non-character array type (because the unsigned
        // element type is never a character type) and so complained that an
        // array can never be assigned to. Now, BuildLLVM doesn't attempt type
        // conversions for compound initializers because they're actually
        // unnecessary.
        "unsigned c_aau[1][1] = {{1}};",
        "unsigned get_c_aau() { return c_aau[0][0]; }",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    final long ldSize = SIZEOF_LONG_DOUBLE;
    final long dSize = SrcDoubleType.getLLVMType(ctxt).getPrimitiveSizeInBits()/8;
    final long fSize = SrcFloatType.getLLVMType(ctxt).getPrimitiveSizeInBits()/8;
    final long iSize = SrcIntType.getLLVMWidth()/8;
    final long uSize = SrcUnsignedIntType.getLLVMWidth()/8;
    final long lSize = SrcLongType.getLLVMWidth()/8;
    final long ulSize = SrcUnsignedLongType.getLLVMWidth()/8;
    final long llSize = SrcLongLongType.getLLVMWidth()/8;
    final long ullSize = SrcUnsignedLongLongType.getLLVMWidth()/8;
    final long ptrdiffTSize = SRC_PTRDIFF_T_TYPE.getLLVMWidth()/8;

    // multiplicative and additive operators for non-pointer types
    checkBinaryArith(exec, mod, "*", "i",    "c",   iSize,   -15);
    checkBinaryArith(exec, mod, "*", "i",    "d",   dSize,   -16.);
    checkBinaryArith(exec, mod, "/", "i",    "u",   iSize,   2);
    checkBinaryArith(exec, mod, "/", "i",    "f",   fSize,   -3.125);
    checkBinaryArith(exec, mod, "%", "c",    "l",   lSize,   2);
    checkBinaryArith(exec, mod, "%", "uc",   "i",   iSize,   3);
    checkBinaryArith(exec, mod, "+", "i",    "i",   iSize,   19);
    checkBinaryArith(exec, mod, "+", "ld",   "i",   ldSize,  -84);
    checkBinaryArith(exec, mod, "-", "l",    "c",   lSize,   308);
    checkBinaryArith(exec, mod, "-", "d",    "f",   dSize,   43.2e5);

    // additive operators for pointer types
    final long pSize = runFn(exec, mod, "pSize").toInt(false).longValue();
    checkBinaryPtr(  exec, mod, "+", "pi", "pi", "i",  pSize,        55);
    checkBinaryPtr(  exec, mod, "+", "pi", "pi", "c",  pSize,        33);
    checkBinaryPtr(  exec, mod, "+", "pi", "b",  "pi", pSize,        11);
    checkBinaryPtr(  exec, mod, "+", "pi", "l",  "pi", pSize,        77);
    checkBinaryPtr(  exec, mod, "+", "pd", "l",  "pd", pSize,        22.2);
    checkBinaryPtr(  exec, mod, "-", "pi", "pi", "ul", pSize,        66);
    checkBinaryPtr(  exec, mod, "-", "pi", "pi", "l",  pSize,        99);
    checkBinaryArith(exec, mod, "-",       "pi", "pi", ptrdiffTSize, 2);
    checkBinaryArith(exec, mod, "-",       "pd", "pd", ptrdiffTSize, -6);

    // ISO C99 sec. 6.5.5p6.
    checkIntFn(exec, mod, 7, true, "divMultAddRem",
               getIntGeneric(7, ctxt), getIntGeneric(2, ctxt));
    checkIntFn(exec, mod, 9, true, "divMultAddRem",
               getIntGeneric(9, ctxt), getIntGeneric(3, ctxt));
    checkIntFn(exec, mod, -9, true, "divMultAddRem",
               getIntGeneric(-9, ctxt), getIntGeneric(2, ctxt));
    checkIntFn(exec, mod, 7, true, "divMultAddRem",
               getIntGeneric(7, ctxt), getIntGeneric(-2, ctxt));
    checkIntFn(exec, mod, -8, true, "divMultAddRem",
               getIntGeneric(-8, ctxt), getIntGeneric(-3, ctxt));

     // left shift operator, signed after promotion
    checkBinaryArith(exec, mod, "<<", "i",   "i",  iSize,   40);
    checkBinaryArith(exec, mod, "<<", "b",   "b",  iSize,   2);
    checkBinaryArith(exec, mod, "<<", "i",   "c",  iSize,   40);
    checkBinaryArith(exec, mod, "<<", "c",   "u",  iSize,   -1216);
    checkBinaryArith(exec, mod, "<<", "l",   "c",  lSize,   288);
    checkBinaryArith(exec, mod, "<<", "i",   "l",  iSize,   -2048);
    checkBinaryArith(exec, mod, "<<", "l",   "i",  lSize,
                     longMax.longValue());

    // left shift operator, unsigned after promotion
    checkBinaryArith(exec, mod, "<<", "u",   "i",  uSize,   512);
    checkBinaryArith(exec, mod, "<<", "ul",  "ul", ulSize,  12);

    // right shift operator, signed after promotion
    checkBinaryArith(exec, mod, ">>", "i",   "i",  iSize,   1);
    checkBinaryArith(exec, mod, ">>", "sc",  "sc", iSize,   -2);
    checkBinaryArith(exec, mod, ">>", "uc",  "b",  iSize,   2);
    checkBinaryArith(exec, mod, ">>", "ll",  "u",  llSize,  -13);
    checkBinaryArith(exec, mod, ">>", "l",   "l",  lSize,   -2);

    // right shift operator, unsigned after promotion
    checkBinaryArith(exec, mod, ">>", "u",   "sc", uSize,   2);
    checkBinaryArith(exec, mod, ">>", "ull", "u",  ullSize, 11);
    checkBinaryArith(exec, mod, ">>", "u",   "i",  uSize,
                     intMax.longValue());

    // relational operators for integer types
    checkBinaryArith(exec, mod, "<",  "u",   "c",  iSize,   0);
    checkBinaryArith(exec, mod, "<",  "i",   "uc", iSize,   1);
    checkBinaryArith(exec, mod, "<",  "i",   "i",  iSize,   0);
    checkBinaryArith(exec, mod, "<",  "u",   "u",  iSize,   0);
    checkBinaryArith(exec, mod, ">",  "b",   "u",  iSize,   0);
    checkBinaryArith(exec, mod, ">",  "u",   "l",  iSize,   1);
    checkBinaryArith(exec, mod, ">",  "l",   "i",  iSize,   0);
    checkBinaryArith(exec, mod, ">",  "ull", "c",  iSize,   0);
    checkBinaryArith(exec, mod, "<=", "u",   "c",  iSize,   0);
    checkBinaryArith(exec, mod, "<=", "i",   "uc", iSize,   1);
    checkBinaryArith(exec, mod, "<=", "i",   "i",  iSize,   1);
    checkBinaryArith(exec, mod, "<=", "u",   "u",  iSize,   1);
    checkBinaryArith(exec, mod, ">=", "b",   "u",  iSize,   0);
    checkBinaryArith(exec, mod, ">=", "u",   "l",  iSize,   1);
    checkBinaryArith(exec, mod, ">=", "l",   "i",  iSize,   1);
    checkBinaryArith(exec, mod, ">=", "ull", "c",  iSize,   1);

    // relational operators for floating types
    checkBinaryArith(exec, mod, "<",  "f",   "c",  iSize,   1);
    checkBinaryArith(exec, mod, "<",  "f",   "d",  iSize,   0);
    checkBinaryArith(exec, mod, "<",  "i",   "f",  iSize,   0);
    checkBinaryArith(exec, mod, ">",  "u",   "ld", iSize,   1);
    checkBinaryArith(exec, mod, ">",  "d",   "d",  iSize,   0);
    checkBinaryArith(exec, mod, ">",  "i",   "d",  iSize,   0);
    checkBinaryArith(exec, mod, "<=", "i",   "d",  iSize,   1);
    checkBinaryArith(exec, mod, "<=", "ld",  "d",  iSize,   1);
    checkBinaryArith(exec, mod, "<=", "d",   "d",  iSize,   0);
    checkBinaryArith(exec, mod, ">=", "b",   "f",  iSize,   1);
    checkBinaryArith(exec, mod, ">=", "d",   "i",  iSize,   1);
    checkBinaryArith(exec, mod, ">=", "i",   "f",  iSize,   0);

    // relational operators for pointer types
    checkBinaryArith(exec, mod, "<",  "pi",  "pi", iSize,   0);
    checkBinaryArith(exec, mod, "<",  "pf",  "pf", iSize,   1);
    checkBinaryArith(exec, mod, "<",  "pd",  "pd", iSize,   0);
    checkBinaryArith(exec, mod, ">",  "pi",  "pi", iSize,   1);
    checkBinaryArith(exec, mod, ">",  "pf",  "pf", iSize,   0);
    checkBinaryArith(exec, mod, ">",  "pd",  "pd", iSize,   0);
    checkBinaryArith(exec, mod, "<=", "pi",  "pi", iSize,   0);
    checkBinaryArith(exec, mod, "<=", "pf",  "pf", iSize,   1);
    checkBinaryArith(exec, mod, "<=", "pd",  "pd", iSize,   1);
    checkBinaryArith(exec, mod, ">=", "pi",  "pi", iSize,   1);
    checkBinaryArith(exec, mod, ">=", "pf",  "pf", iSize,   0);
    checkBinaryArith(exec, mod, ">=", "pd",  "pd", iSize,   1);

    // equality operators for integer types
    checkBinaryArith(exec, mod, "==", "i",  "c",  iSize,   0);
    checkBinaryArith(exec, mod, "==", "u",  "b",  iSize,   1);
    checkBinaryArith(exec, mod, "==", "l",  "i",  iSize,   0);
    checkBinaryArith(exec, mod, "!=", "sc", "i",  iSize,   1);
    checkBinaryArith(exec, mod, "!=", "u",  "lu", iSize,   0);
    checkBinaryArith(exec, mod, "!=", "l",  "c",  iSize,   1);

    // equality operators for floating types
    checkBinaryArith(exec, mod, "==", "f",  "c",  iSize,   0);
    checkBinaryArith(exec, mod, "==", "f",  "d",  iSize,   1);
    checkBinaryArith(exec, mod, "==", "i",  "f",  iSize,   0);
    checkBinaryArith(exec, mod, "!=", "i",  "d",  iSize,   1);
    checkBinaryArith(exec, mod, "!=", "ld", "d",  iSize,   0);
    checkBinaryArith(exec, mod, "!=", "d",  "d",  iSize,   1);

    // equality operators for pointer types not involving void pointer
    checkBinaryArith(exec, mod, "==", "pi", "pi", iSize,   0);
    checkBinaryArith(exec, mod, "==", "pf", "pf", iSize,   0);
    checkBinaryArith(exec, mod, "==", "pd", "pd", iSize,   1);
    checkBinaryArith(exec, mod, "!=", "pi", "pi", iSize,   1);
    checkBinaryArith(exec, mod, "!=", "pf", "pf", iSize,   1);
    checkBinaryArith(exec, mod, "!=", "pd", "pd", iSize,   0);
    checkIntGlobalVar(mod, "fn1_eq_fn2", 0);
    checkIntGlobalVar(mod, "fn1_eq_fn1", 1);
    checkIntFn(exec, mod, "fn1_ne_fn2", 1);
    checkIntGlobalVar(mod, "fn2_ne_fn2", 0);

    // equality operators for pointer types involving void pointer
    checkBinaryArith(exec, mod, "==", "pi", "pv", iSize,   0);
    checkBinaryArith(exec, mod, "==", "pv", "pf", iSize,   0);
    checkBinaryArith(exec, mod, "==", "pv", "pv", iSize,   1);
    checkBinaryArith(exec, mod, "!=", "pv", "pi", iSize,   1);
    checkBinaryArith(exec, mod, "!=", "pv", "pv", iSize,   1);
    checkBinaryArith(exec, mod, "!=", "pd", "pv", iSize,   0);

    // equals operator for pointer types and null pointer constants
    checkIntGlobalVar(mod, "c_inpc_eq_inpc", 1);
    checkIntGlobalVar(mod, "c_vnpc_eq_vnpc", 1);
    checkIntGlobalVar(mod, "c_vnpc_eq_inpc", 1);
    checkIntGlobalVar(mod, "c_inpc_eq_vnpc", 1);
    checkIntGlobalVar(mod, "c_pi_eq_inpc", 0);
    checkIntGlobalVar(mod, "c_vnpc_eq_pd", 0);
    checkIntGlobalVar(mod, "c_fn1_eq_vnpc", 0);
    checkIntGlobalVar(mod, "c_inpc_eq_fn2", 0);

    // not-equals operator for pointer types and null pointer constants
    checkIntGlobalVar(mod, "c_inpc_ne_inpc", 0);
    checkIntGlobalVar(mod, "c_vnpc_ne_vnpc", 0);
    checkIntGlobalVar(mod, "c_vnpc_ne_inpc", 0);
    checkIntGlobalVar(mod, "c_inpc_ne_vnpc", 0);
    checkIntGlobalVar(mod, "c_pi_ne_inpc", 1);
    checkIntGlobalVar(mod, "c_vnpc_ne_pd", 1);
    checkIntGlobalVar(mod, "c_fn1_ne_vnpc", 1);
    checkIntGlobalVar(mod, "c_inpc_ne_fn2", 1);

    // non-constant-expression equality operators for pointer types and
    // null pointer constants
    checkIntFn(exec, mod, "vnpc_eq_inpc", 1);
    checkIntFn(exec, mod, "inpc_ne_fn2", 1);

    // bitwise &, ^, and |
    checkBinaryArith(exec, mod, "&",  "i",   "i",   iSize,   1);
    checkBinaryArith(exec, mod, "&",  "i",   "u",   iSize,   10);
    checkBinaryArith(exec, mod, "&",  "l",   "u",   lSize,   9);
    checkBinaryArith(exec, mod, "&",  "b",   "l",   lSize,   0);
    checkBinaryArith(exec, mod, "^",  "i",   "uc",  iSize,   6);
    checkBinaryArith(exec, mod, "^",  "c",   "u",   iSize,
                     unsignedMax.longValue()-1);
    checkBinaryArith(exec, mod, "|",  "uc",  "ull", ullSize, 15);
    checkBinaryArith(exec, mod, "|",  "uc",  "b",   iSize,   3);

    // && and || with constant expressions
    checkBinaryArith(exec, mod, "&&", "i",   "i",   iSize,   1);
    checkBinaryArith(exec, mod, "&&", "i",   "c",   iSize,   0);
    checkBinaryArith(exec, mod, "&&", "u",   "l",   iSize,   0);
    checkBinaryArith(exec, mod, "&&", "b",   "b",   iSize,   0);
    checkBinaryArith(exec, mod, "&&", "f",   "pi",  iSize,   1);
    checkBinaryArith(exec, mod, "&&", "pv",  "d",   iSize,   0);
    checkBinaryArith(exec, mod, "&&", "d",   "d",   iSize,   0);
    checkBinaryArith(exec, mod, "||", "l",   "b",   iSize,   1);
    checkBinaryArith(exec, mod, "||", "ll",  "i",   iSize,   1);
    checkBinaryArith(exec, mod, "||", "ll",  "ll",  iSize,   1);
    checkBinaryArith(exec, mod, "||", "u",   "u",   iSize,   0);
    checkBinaryArith(exec, mod, "||", "f",   "pi",  iSize,   1);
    checkBinaryArith(exec, mod, "||", "pv",  "d",   iSize,   1);
    checkBinaryArith(exec, mod, "||", "pv",  "f",   iSize,   0);
    checkIntFn(exec, mod, 0, "logicalAndStaticLocal", 99);
    checkIntFn(exec, mod, 99, "logicalAndStaticLocal", 0);
    checkIntFn(exec, mod, 1, "logicalOrStaticLocal", 99);
    checkIntFn(exec, mod, 99, "logicalOrStaticLocal", 0);

    // && with non-constant expression => potential short-circuiting
    checkIntFn(exec, mod, 1, "logicalAndLocal", 1, 2);
    checkIntFn(exec, mod, 99, "getSideEffect");
    checkIntFn(exec, mod, 0, "logicalAndLocal", -5, 0);
    checkIntFn(exec, mod, 99, "getSideEffect");
    checkIntFn(exec, mod, 0, "logicalAndLocal", 0, 3);
    checkIntFn(exec, mod, 3, "getSideEffect");
    checkIntFn(exec, mod, 0, "logicalAndLocal", 0, 0);
    checkIntFn(exec, mod, 0, "getSideEffect");

    // || with non-constant expression => potential short-circuiting
    checkIntFn(exec, mod, 1, "logicalOrLocal", 1, 2);
    checkIntFn(exec, mod, 2, "getSideEffect");
    checkIntFn(exec, mod, 1, "logicalOrLocal", -5, 8);
    checkIntFn(exec, mod, 8, "getSideEffect");
    checkIntFn(exec, mod, 1, "logicalOrLocal", 0, 3);
    checkIntFn(exec, mod, 99, "getSideEffect");
    checkIntFn(exec, mod, 0, "logicalOrLocal", 0, 0);
    checkIntFn(exec, mod, 99, "getSideEffect");

    // && and || with multiple basic blocks in operands.
    //                                                 && ?? ||
    checkIntFn(exec, mod, 0, "logicalAndMultipleBBs", 0, 0, 0, 0);
    checkIntFn(exec, mod, 0, "logicalOrMultipleBBs",  0, 0, 0, 0);
    checkIntFn(exec, mod, 0, "logicalAndMultipleBBs", 0, 1, 0, 0);
    checkIntFn(exec, mod, 0, "logicalOrMultipleBBs",  0, 1, 0, 0);
    checkIntFn(exec, mod, 0, "logicalAndMultipleBBs", 1, 0, 0, 0);
    checkIntFn(exec, mod, 0, "logicalOrMultipleBBs",  1, 0, 0, 0);
    checkIntFn(exec, mod, 0, "logicalAndMultipleBBs", 1, 1, 0, 0);
    checkIntFn(exec, mod, 1, "logicalOrMultipleBBs",  1, 1, 0, 0);
    checkIntFn(exec, mod, 1, "logicalAndMultipleBBs", 1, 1, 0, 1);
    checkIntFn(exec, mod, 1, "logicalOrMultipleBBs",  0, 0, 0, 1);
    checkIntFn(exec, mod, 1, "logicalAndMultipleBBs", 1, 1, 1, 0);
    checkIntFn(exec, mod, 1, "logicalOrMultipleBBs",  0, 0, 1, 0);
    checkIntFn(exec, mod, 1, "logicalAndMultipleBBs", 1, 1, 1, 1);
    checkIntFn(exec, mod, 1, "logicalOrMultipleBBs",  0, 0, 1, 1);

    // []

    checkIntGlobalVar(mod, "cs_subscript", iSize);
    checkIntFn(exec, mod, "cv_subscript", 55);
    checkIntFn(exec, mod, "s_subscript", iSize);
    checkIntFn(exec, mod, "v_subscript", 99);

    checkIntGlobalVar(mod, "cs_subscript_rev", dSize);
    checkDoubleFn(exec, mod, "cv_subscript_rev", 33.3);
    checkIntFn(exec, mod, "s_subscript_rev", dSize);
    checkDoubleFn(exec, mod, "v_subscript_rev", 44.4);

    checkIntFn(exec, mod, 1,  "v_subsubscript", 0, 1);
    checkIntFn(exec, mod, 10, "v_subsubscript", 1, 0);
    checkIntFn(exec, mod, 22,  "v_subsubscript_rev", 2, 1);
    checkIntFn(exec, mod, 11, "v_subsubscript_rev", 1, 1);
    checkIntFn(exec, mod, 0,  "v_subsubscript_revrev", 0, 0);
    checkIntFn(exec, mod, 20, "v_subsubscript_revrev", 2, 0);
    checkIntFn(exec, mod, iSize, "s_subsubscript");
    checkIntFn(exec, mod, iSize, "s_subsubscript_rev");
    checkIntFn(exec, mod, iSize, "s_subsubscript_revrev");
    checkIntFn(exec, mod, iSize*2, "s_aai_sub");
    checkIntFn(exec, mod, iSize*2*3, "s_aai");

    checkIntFn(exec, mod, 1, "get_c_aau");

    exec.dispose();
  }

  @Test public void conditionalOperator() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final BigInteger uMax
      = BigInteger.ONE.shiftLeft((int)SrcUnsignedIntType.getWidth())
        .subtract(BigInteger.ONE);
    final SimpleResult simpleResult = buildLLVMSimple(
      "", TARGET_DATA_LAYOUT,
      new String[]{
        "#include <stddef.h>",
        "int a10i[10] = {0, 11, 22, 33, 44, 55, 66, 77, 88, 99};",
        "double a10d[10] = {0,    11.1, 22.2, 33.3, 44.4,\n",
        "                   55.5, 66.6, 77.7, 88.8, 99.9};",
        "int *get_a10i() { return a10i; }",
        "void *getNullPtr() { return (void*)0; }",
        "struct S { int i; int j; } s1={1,11}, s2={2,22};",
        "union U { int i; int j; } u1={11}, u2={22};",
        "size_t pSize() { void *p; return sizeof p; }",

        // varying op1 type, arithmetic op2 and op3
        genTernaryArith("?",":", "i",  "i",  "i", "0",    "1",    "2"),
        genTernaryArith("?",":", "i",  "c",  "i", "1",    "1",    "2"),
        genTernaryArith("?",":", "b",  "i",  "u", "0",    "1",
                        uMax.toString()),
        genTernaryArith("?",":", "d",  "sc", "u", "-0.3", "-1",   "2"),
        genTernaryArith("?",":", "f",  "sc", "u", "0.0",  "-1",   "2"),
        genTernaryArith("?",":", "pi", "d",  "i", "a10i", "5e5",  "9"),
        genTernaryArith("?",":", "pd", "ld", "i", "0",    "1e-2", "-8"),

        // op2 and op3 have the same struct/union type
        //
        // A conditional operator produces an rvalue, and a struct/union
        // rvalue is never a constant expression because it requires a load
        // from an lvalue (a compound literal, for example, is an lvalue), so
        // we cannot exercise this case at file scope except within sizeof.
        "size_t cs_cond_i_s_s = sizeof(0 ? s1 : s2);",
        "size_t cs_cond_i_u_u = sizeof(1 ? u1 : u2);",
        "size_t s_cond_i_s_s() { return sizeof(0 ? s1 : s2); }",
        "size_t s_cond_i_u_u() { return sizeof(0 ? u1 : u2); }",
        "int v_cond_i_s_s() { return (0 ? s1 : s2).j; }",
        "int v_cond_i_u_u() { return (1 ? u1 : u2).i; }",

        // op2 and op3 have void type
        // Also check that the unchosen op's evaluation is skipped.
        "int sideEffect1;",
        "int sideEffect2;",
        "int getSideEffect1() { return sideEffect1; }",
        "int getSideEffect2() { return sideEffect2; }",
        "void setSideEffect1(int new) { sideEffect1 = new; }",
        "void setSideEffect2(int new) { sideEffect2 = new; }",
        "void voidExpression(int i, int j, int k) {",
        "  setSideEffect1(99); setSideEffect2(99);",
        "  i ? setSideEffect1(j) : setSideEffect2(k);",
        "}",

        // pointer/null-pointer-constant op2 and op3
        genTernaryPtr("?",":", "pi", "i", "pi", "pi", "1", "a10i+1", "a10i+5"),
        genTernaryPtr("?",":", "pd", "b", "pd", "pd", "0", "a10d+1", "a10d+5"),
        genTernaryPtr("?",":", "pi", "c", "pi", "pv", "1", "a10i+2", "(void*)0"),
        genTernaryPtr("?",":", "pi", "l", "pv", "pi", "0", "a10d",   "a10i+3"),
        "void *v_cond_b_npc_pi(_Bool b) {",
        "  int *p = a10i;",
        "  return (void*)(b ? 0 : p);",
        "}",
        "void *v_cond_b_pi_vnpc(_Bool b) {",
        "  int *p = a10i;",
        "  return (void*)(b ? p : (void*)0);",
        "}",
        "size_t s_cond_b_npc_pi() {",
        "  _Bool b; int *p = a10i;",
        "  return sizeof(b ? 0 : p);",
        "}",
        "size_t s_cond_b_pi_vnpc() {",
        "  _Bool b; int *p = a10i;",
        "  return sizeof(b ? p : (void*)0);",
        "}",
        "void *cv_cond_b_vnpc_pi_ = (_Bool)1 ? (void*)0 : a10i;",
        "void *cv_cond_b_pi_npc_ = (_Bool)0 ? a10i : 0;",
        "void *cv_cond_i_pi_npc_ = 1 ? a10i : 0;",
        "void *cv_cond_b_vnpc_pi() { return cv_cond_b_vnpc_pi_; }",
        "void *cv_cond_b_pi_npc() { return cv_cond_b_pi_npc_; }",
        "void *cv_cond_i_pi_npc() { return cv_cond_i_pi_npc_; }",

        // function/null-pointer-constant op2 and op3
        "void v_cond_i_pfn_pfn(int i) {",
        "  setSideEffect1(0); setSideEffect2(0);",
        "  (i ? setSideEffect1 : setSideEffect2)(99);",
        "}",
        "int v_cond_i_pfn_npc(int i) {",
        "  return (int)(setSideEffect1 == (i ? setSideEffect1 : 0));",
        "}",
        "int v_cond_i_vnpc_pfn(int i) {",
        "  return (int)(setSideEffect1 == (i ? (void*)0 : setSideEffect1));",
        "}",

        // operands with multiple basic blocks
        "int multipleBBs(int i, int j, int k, int l, int m, int n) {",
        "  return (i && j) ? (k && l) : (m || n);",
        "}",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    final long ldSize = SIZEOF_LONG_DOUBLE;
    final long dSize = SrcDoubleType.getLLVMType(ctxt).getPrimitiveSizeInBits()/8;
    final long iSize = SrcIntType.getLLVMWidth()/8;
    final long uSize = SrcUnsignedIntType.getLLVMWidth()/8;

    // varying op1 type, arithmetic op2 and op3
    checkTernaryArith(exec, mod, "?",":", "i",  "i",  "i", iSize,  2);
    checkTernaryArith(exec, mod, "?",":", "i",  "c",  "i", iSize,  1);
    checkTernaryArith(exec, mod, "?",":", "b",  "i",  "u", uSize,
                      uMax.longValue());
    checkTernaryArith(exec, mod, "?",":", "d",  "sc", "u", uSize,
                      uMax.longValue());
    checkTernaryArith(exec, mod, "?",":", "f",  "sc", "u", uSize,  2);
    checkTernaryArith(exec, mod, "?",":", "pi", "d",  "i", dSize,  5e5);
    checkTernaryArith(exec, mod, "?",":", "pd", "ld", "i", ldSize, -8);

    // op2 and op3 have the same struct/union type
    checkIntGlobalVar(mod, "cs_cond_i_s_s", 2*iSize);
    checkIntGlobalVar(mod, "cs_cond_i_u_u", iSize);
    checkIntFn(exec, mod, "s_cond_i_s_s", 2*iSize);
    checkIntFn(exec, mod, "s_cond_i_u_u", iSize);
    checkIntFn(exec, mod, "v_cond_i_s_s", 22);
    checkIntFn(exec, mod, "v_cond_i_u_u", 11);

    // op1 and op3 have void type
    runFn(exec, mod, "voidExpression", 0, 1, -2);
    checkIntFn(exec, mod, 99, "getSideEffect1");
    checkIntFn(exec, mod, -2, "getSideEffect2");
    runFn(exec, mod, "voidExpression", 1, 5, 4);
    checkIntFn(exec, mod, 5, "getSideEffect1");
    checkIntFn(exec, mod, 99, "getSideEffect2");

    // pointer/null-pointer-constant op2 and op3
    final long pSize = runFn(exec, mod, "pSize").toInt(false).longValue();
    checkTernaryPtr(exec, mod, "?",":", "pi", "i", "pi", "pi", pSize, 11);
    checkTernaryPtr(exec, mod, "?",":", "pd", "b", "pd", "pd", pSize, 55.5);
    checkTernaryPtr(exec, mod, "?",":", "pi", "c", "pi", "pv", pSize, 22);
    checkTernaryPtr(exec, mod, "?",":", "pi", "l", "pv", "pi", pSize, 33);
    final LLVMGenericValue a10i = runFn(exec, mod, "get_a10i");
    final LLVMGenericValue nullPtr = runFn(exec, mod, "getNullPtr");
    final LLVMGenericValue trueBool = getBoolGeneric(1,ctxt);
    final LLVMGenericValue falseBool = getBoolGeneric(0,ctxt);
    checkPointerFn(exec, mod, "v_cond_b_npc_pi", trueBool, nullPtr);
    checkPointerFn(exec, mod, "v_cond_b_npc_pi", falseBool, a10i);
    checkPointerFn(exec, mod, "v_cond_b_pi_vnpc", trueBool, a10i);
    checkPointerFn(exec, mod, "v_cond_b_pi_vnpc", falseBool, nullPtr);
    checkIntFn(exec, mod, "s_cond_b_npc_pi", pSize);
    checkIntFn(exec, mod, "s_cond_b_pi_vnpc", pSize);
    checkPointerFn(exec, mod, "cv_cond_b_vnpc_pi", nullPtr);
    checkPointerFn(exec, mod, "cv_cond_b_pi_npc", nullPtr);
    checkPointerFn(exec, mod, "cv_cond_i_pi_npc", a10i);

    // function op2 and op3
    runFn(exec, mod, "v_cond_i_pfn_pfn", 1);
    checkIntFn(exec, mod, "getSideEffect1", 99);
    checkIntFn(exec, mod, "getSideEffect2", 0);
    runFn(exec, mod, "v_cond_i_pfn_pfn", 0);
    checkIntFn(exec, mod, "getSideEffect1", 0);
    checkIntFn(exec, mod, "getSideEffect2", 99);
    checkIntFn(exec, mod, 0, "v_cond_i_pfn_npc", 0);
    checkIntFn(exec, mod, 1, "v_cond_i_pfn_npc", 1);
    checkIntFn(exec, mod, 1, "v_cond_i_vnpc_pfn", 0);
    checkIntFn(exec, mod, 0, "v_cond_i_vnpc_pfn", 1);

    // operands with multiple basic blocks
    //                                       && ?  && :  ||
    checkIntFn(exec, mod, 1, "multipleBBs", 1, 1, 1, 1, 0, 0);
    checkIntFn(exec, mod, 0, "multipleBBs", 1, 1, 1, 0, 0, 1);
    checkIntFn(exec, mod, 0, "multipleBBs", 1, 1, 0, 1, 1, 0);
    checkIntFn(exec, mod, 0, "multipleBBs", 1, 1, 0, 0, 1, 1);
    checkIntFn(exec, mod, 0, "multipleBBs", 0, 0, 1, 1, 0, 0);
    checkIntFn(exec, mod, 1, "multipleBBs", 0, 1, 0, 0, 1, 0);
    checkIntFn(exec, mod, 1, "multipleBBs", 1, 0, 1, 0, 0, 1);
    checkIntFn(exec, mod, 1, "multipleBBs", 0, 0, 0, 1, 1, 1);

    exec.dispose();
  }

  @Test public void commaOperator() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        "#include <stddef.h>",

        // If BuildLLVM fails to drop all but the last of comma operands off
        // the stack, the final result will be 5.2+6=11.2=>11, and the size
        // will be double.
        "int i = 3 + (5.2, 6);",
        "size_t s = sizeof(3 + (5.2, 6));",

        // All comma operands must be evaluated and in the order specified.
        "int sideEffect;",
        "int getSideEffect() { return sideEffect; }",
        "void setSideEffect(int new) { sideEffect = new; }",
        "double x2(double d) {",
        "  setSideEffect(0);",
        "  return 2*(setSideEffect(88), setSideEffect(sideEffect+7),",
        "            setSideEffect(sideEffect*2), d);",
        "}",

        // Example from ISO C99 sec. 6.5.17p3.
        "int f(int x, int y, int z) { return x + y + z; }",
        "int call_f(int a, int c) {",
        "  int t = 0;",
        "  return f(a, (t=3, t+2), c);",
        "}",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    final long iSize = SrcIntType.getLLVMWidth()/8;

    checkIntGlobalVar(mod, "i", 9);
    checkIntGlobalVar(mod, "s", iSize);
    checkDoubleFn(exec, mod, 6.4, "x2", getDoubleGeneric(3.2, ctxt));
    checkIntFn(exec, mod, "getSideEffect", 190);
    checkIntFn(exec, mod, 8, "call_f", 1, 2);

    exec.dispose();
  }

  @Test public void assignment() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        "#include <stddef.h>",
        "void *malloc(size_t size);",
        "void free(void *);",
        "size_t pSize() { void *p; return sizeof p; }",

        "int ptr() {",
        "  int *p, *p1;",
        "  p1 = p = malloc(5 * sizeof(int));",
        "  p[0] = 0;",
        "  p[1] = 1;",
        "  p[2] = 2;",
        "  p[3] = 3;",
        "  p[4] = 4;",
        "  int x;",
        "  *p1 *= (x = 10000); p1 += 1; x /= 10;",
        "  *p1 *= x;           p1 += 1; x /= 10;",
        "  *p1 *= x;           p1 += 1; x /= 10;",
        "  *p1 *= x;           p1 += 1; x /= 10;",
        "  *p1 *= x;           p1 += 1; x /= 10;",
        "  x = 9999;",
        "  int y;",
        "  p1 -= 1; x -= *p1;",
        "  p1 -= 1; x -= *p1;",
        "  p1 -= 1; x -= *p1;",
        "  p1 -= 1; x -= *p1;",
        "  p1 -= 1; y = x -= *p1;",
        "  free(p);",
        "  return y + x*10000;",
        "}",

        "double mul(double x, double y) { x *=  y; return x; }",
        "double div(double x, double y) { x /=  y; return x; }",
        "int    rem(int    x, int    y) { x %=  y; return x; }",
        "double add(double x, double y) { x +=  y; return x; }",
        "double sub(double x, double y) { x -=  y; return x; }",
        "int    shl(int    x, int    y) { x <<= y; return x; }",
        "int    shr(int    x, int    y) { x >>= y; return x; }",
        "int    and(int    x, int    y) { x &=  y; return x; }",
        "int    xor(int    x, int    y) { x ^=  y; return x; }",
        "int    or (int    x, int    y) { x |=  y; return x; }",

        "size_t s_mul() { double x; double y; return sizeof(x *=  y); }",
        "size_t s_div() { double x; double y; return sizeof(x /=  y); }",
        "size_t s_rem() { int    x; int    y; return sizeof(x %=  y); }",
        "size_t s_add() { double x; double y; return sizeof(x +=  y); }",
        "size_t s_sub() { double x; double y; return sizeof(x -=  y); }",
        "size_t s_shl() { int    x; int    y; return sizeof(x <<= y); }",
        "size_t s_shr() { int    x; int    y; return sizeof(x >>= y); }",
        "size_t s_and() { int    x; int    y; return sizeof(x &=  y); }",
        "size_t s_xor() { int    x; int    y; return sizeof(x ^=  y); }",
        "size_t s_or () { int    x; int    y; return sizeof(x |=  y); }",
        "size_t s_ptr() { int   *x; int    y; return sizeof(x +=  y); }",

        "double d1, d2;",
        "int i1, i2;",
        "int *p;",
        "size_t cs_mul = sizeof(d1 *=  d2);",
        "size_t cs_div = sizeof(d1 /=  d2);",
        "size_t cs_rem = sizeof(i1 %=  i2);",
        "size_t cs_add = sizeof(d1 +=  d2);",
        "size_t cs_sub = sizeof(d1 -=  d2);",
        "size_t cs_shl = sizeof(i1 <<= i2);",
        "size_t cs_shr = sizeof(i1 >>= i2);",
        "size_t cs_and = sizeof(i1 &=  i2);",
        "size_t cs_xor = sizeof(i1 ^=  i2);",
        "size_t cs_or  = sizeof(i1 |=  i2);",
        "size_t cs_ptr = sizeof(p  -=  i1);",

        // bit-fields

        "struct SB { int i:4; int :1; unsigned j:3; int k:5; int :0; int l:1; };",
        "struct SB sb;",
        "void assignSB() { struct SB *p = &sb; p->i=9; p->j=5; p->k=17; p->l=1; }",
        "int getSBMem0() { return sb.i; }",
        "int getSBMem1() { return sb.j; }",
        "int getSBMem2() { return sb.k; }",
        "int getSBMem3() { return sb.l; }",

        "union UB { int i:4; int :1; unsigned j:3; int k:5; int :0; int l:1; };",
        "union UB ub = {9};",                                         // 10010
        "int getSetUBMem0() { int o = ub.i; ub.i = -1; return o; }",  // 11110
        "int getSetUBMem1() { int o = ub.j; ub.j = 5;  return o; }",  // 10110
        "int getSetUBMem2() { int o = ub.k; ub.k = 17; return o; }",  // 10001
        "int getSetUBMem3() { int o = ub.l; ub.l = 0;  return o; }",  // 00001
        "int getUBMem2() { return ub.k; }",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    final long iSize = SrcIntType.getLLVMWidth()/8;
    final long dSize = SrcDoubleType.getLLVMType(ctxt).getPrimitiveSizeInBits()/8;
    final long pSize = runFn(exec, mod, "pSize").toInt(false).longValue();

    checkIntFn(exec, mod, "ptr", 87658765);

    checkDoubleFn(exec, mod, 15.,  "mul", 5.,  3.);
    checkDoubleFn(exec, mod, 7.5,  "div", 30., 4.);
    checkIntFn   (exec, mod, 2,    "rem", 30,  4);
    checkDoubleFn(exec, mod, 2.4,  "add", 2.3, .1);
    checkDoubleFn(exec, mod, -1.5, "sub", 8.5, 10.);
    checkIntFn   (exec, mod, 40,   "shl", 10,  2);
    checkIntFn   (exec, mod, 2,    "shr", 10,  2);
    checkIntFn   (exec, mod, 1,    "and", 9,   3);
    checkIntFn   (exec, mod, 10,   "xor", 9,   3);
    checkIntFn   (exec, mod, 11,   "or",  9,   3);

    checkIntFn(exec, mod, "s_mul", dSize);
    checkIntFn(exec, mod, "s_div", dSize);
    checkIntFn(exec, mod, "s_rem", iSize);
    checkIntFn(exec, mod, "s_add", dSize);
    checkIntFn(exec, mod, "s_sub", dSize);
    checkIntFn(exec, mod, "s_shl", iSize);
    checkIntFn(exec, mod, "s_shr", iSize);
    checkIntFn(exec, mod, "s_and", iSize);
    checkIntFn(exec, mod, "s_xor", iSize);
    checkIntFn(exec, mod, "s_or",  iSize);
    checkIntFn(exec, mod, "s_ptr", pSize);

    checkIntGlobalVar(mod, "cs_mul", dSize);
    checkIntGlobalVar(mod, "cs_div", dSize);
    checkIntGlobalVar(mod, "cs_rem", iSize);
    checkIntGlobalVar(mod, "cs_add", dSize);
    checkIntGlobalVar(mod, "cs_sub", dSize);
    checkIntGlobalVar(mod, "cs_shl", iSize);
    checkIntGlobalVar(mod, "cs_shr", iSize);
    checkIntGlobalVar(mod, "cs_and", iSize);
    checkIntGlobalVar(mod, "cs_xor", iSize);
    checkIntGlobalVar(mod, "cs_or",  iSize);
    checkIntGlobalVar(mod, "cs_ptr", pSize);

    runFn(exec, mod, "assignSB");
    checkIntFn(exec, mod, "getSBMem0", -7);
    checkIntFn(exec, mod, "getSBMem1", 5);
    checkIntFn(exec, mod, "getSBMem2", -15);
    checkIntFn(exec, mod, "getSBMem3", -1);

    checkIntFn(exec, mod, "getSetUBMem0", -7);
    checkIntFn(exec, mod, "getSetUBMem1", 7);
    checkIntFn(exec, mod, "getSetUBMem2", -10);
    checkIntFn(exec, mod, "getSetUBMem3", -1);
    checkIntFn(exec, mod, "getUBMem2", 1);

    exec.dispose();
  }

  @Test public void incDec() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        "#include <stddef.h>",
        "size_t pSize() { void *p; return sizeof p; }",

        "int i = 0;",
        "int get_i() { return i; }",
        "int iPostInc() { return i++; }",
        "int iPostDec() { return i--; }",
        "int iPreInc()  { return ++i; }",
        "int iPreDec()  { return --i; }",
        "size_t siPostInc() { return sizeof(i++); }",
        "size_t siPostDec() { return sizeof(i--); }",
        "size_t siPreInc()  { return sizeof(++i); }",
        "size_t siPreDec()  { return sizeof(--i); }",
        "size_t csiPostInc = sizeof(i++);",
        "size_t csiPostDec = sizeof(i--);",
        "size_t csiPreInc  = sizeof(++i);",
        "size_t csiPreDec  = sizeof(--i);",

        "float f = 5.1;",
        "float get_f() { return f; }",
        "float fPostInc() { return f++; }",
        "float fPostDec() { return f--; }",
        "float fPreInc()  { return ++f; }",
        "float fPreDec()  { return --f; }",
        "size_t sfPostInc() { return sizeof(f++); }",
        "size_t sfPostDec() { return sizeof(f--); }",
        "size_t sfPreInc()  { return sizeof(++f); }",
        "size_t sfPreDec()  { return sizeof(--f); }",

        "int a[3] = {0, 1, 2};",
        "int *p = a;",
        "int get_p() { return *p; }",
        "int pPostInc() { return *p++; }",
        "int pPostDec() { return *p--; }",
        "int pPreInc()  { return *++p; }",
        "int pPreDec()  { return *--p; }",
        "size_t spPostInc() { return sizeof(p++); }",
        "size_t spPostDec() { return sizeof(p--); }",
        "size_t spPreInc()  { return sizeof(++p); }",
        "size_t spPreDec()  { return sizeof(--p); }",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    final long iSize = SrcIntType.getLLVMWidth()/8;
    final long fSize = SrcFloatType.getLLVMType(ctxt).getPrimitiveSizeInBits()/8;
    final long pSize = runFn(exec, mod, "pSize").toInt(false).longValue();

    checkIntFn(exec, mod, "iPostInc", 0); checkIntFn(exec, mod, "get_i", 1);
    checkIntFn(exec, mod, "iPostInc", 1); checkIntFn(exec, mod, "get_i", 2);
    checkIntFn(exec, mod, "iPostDec", 2); checkIntFn(exec, mod, "get_i", 1);
    checkIntFn(exec, mod, "iPostDec", 1); checkIntFn(exec, mod, "get_i", 0);

    checkIntFn(exec, mod, "iPreInc", 1); checkIntFn(exec, mod, "get_i", 1);
    checkIntFn(exec, mod, "iPreInc", 2); checkIntFn(exec, mod, "get_i", 2);
    checkIntFn(exec, mod, "iPreDec", 1); checkIntFn(exec, mod, "get_i", 1);
    checkIntFn(exec, mod, "iPreDec", 0); checkIntFn(exec, mod, "get_i", 0);

    checkIntFn(exec, mod, "siPostInc", iSize);
    checkIntFn(exec, mod, "siPostDec", iSize);
    checkIntFn(exec, mod, "siPreInc", iSize);
    checkIntFn(exec, mod, "siPreDec", iSize);
    checkIntGlobalVar(mod, "csiPostInc", iSize);
    checkIntGlobalVar(mod, "csiPostDec", iSize);
    checkIntGlobalVar(mod, "csiPreInc", iSize);
    checkIntGlobalVar(mod, "csiPreDec", iSize);

    checkFloatFn(exec,mod,"fPostInc",5.1); checkFloatFn(exec,mod,"get_f",6.1);
    checkFloatFn(exec,mod,"fPostInc",6.1); checkFloatFn(exec,mod,"get_f",7.1);
    checkFloatFn(exec,mod,"fPostDec",7.1); checkFloatFn(exec,mod,"get_f",6.1);
    checkFloatFn(exec,mod,"fPostDec",6.1); checkFloatFn(exec,mod,"get_f",5.1);

    checkFloatFn(exec,mod,"fPreInc",6.1); checkFloatFn(exec,mod,"get_f",6.1);
    checkFloatFn(exec,mod,"fPreInc",7.1); checkFloatFn(exec,mod,"get_f",7.1);
    checkFloatFn(exec,mod,"fPreDec",6.1); checkFloatFn(exec,mod,"get_f",6.1);
    checkFloatFn(exec,mod,"fPreDec",5.1); checkFloatFn(exec,mod,"get_f",5.1);

    checkIntFn(exec, mod, "sfPostInc", fSize);
    checkIntFn(exec, mod, "sfPostDec", fSize);
    checkIntFn(exec, mod, "sfPreInc", fSize);
    checkIntFn(exec, mod, "sfPreDec", fSize);

    checkIntFn(exec, mod, "pPostInc", 0); checkIntFn(exec, mod, "get_p", 1);
    checkIntFn(exec, mod, "pPostInc", 1); checkIntFn(exec, mod, "get_p", 2);
    checkIntFn(exec, mod, "pPostDec", 2); checkIntFn(exec, mod, "get_p", 1);
    checkIntFn(exec, mod, "pPostDec", 1); checkIntFn(exec, mod, "get_p", 0);

    checkIntFn(exec, mod, "pPreInc", 1); checkIntFn(exec, mod, "get_p", 1);
    checkIntFn(exec, mod, "pPreInc", 2); checkIntFn(exec, mod, "get_p", 2);
    checkIntFn(exec, mod, "pPreDec", 1); checkIntFn(exec, mod, "get_p", 1);
    checkIntFn(exec, mod, "pPreDec", 0); checkIntFn(exec, mod, "get_p", 0);

    checkIntFn(exec, mod, "spPostInc", pSize);
    checkIntFn(exec, mod, "spPostDec", pSize);
    checkIntFn(exec, mod, "spPreInc", pSize);
    checkIntFn(exec, mod, "spPreDec", pSize);

    exec.dispose();
  }

  @Test public void stdarg() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = buildLLVMSimple(
      // TODO: In order for this test to pass on other platforms, we need to
      // somehow get hold of the correct target triple.
      "x86_64", "",
      new String[]{
        "#include <stdarg.h>",
        "int sum1, sum2;",
        "int get_sum1() { return sum1; }",
        "int get_sum2() { return sum2; }",

        "void intArgs(int n, ...) {",
        "  va_list args1;",
        "  va_list args2;",
        "  va_start(args1, n);",
        "  sum1 += va_arg(args1, int);",
        "  sum1 += va_arg(args1, int);",
        "  va_copy(args2, args1);",
        "  sum1 += va_arg(args1, int);",
        "  sum1 += va_arg(args1, int);",
        "  sum1 += va_arg(args1, int);",
        "  va_end(args1);",
        "  sum2 += va_arg(args2, int);",
        "  sum2 += va_arg(args2, int);",
        "  sum2 += va_arg(args2, int);",
        "  va_end(args2);",
        "}",
        "void call_intArgs() {",
        "  intArgs(5, 11, 22, 33, 44, 55);",
        "}",

        "#define N 100",
        "double dNa[N];",
        "int iNa[N];",
        "struct S { double dNa[N]; int iNa[N]; };",
        "double sum3;",
        "int sum4;",
        "double get_sum3() { return sum3; }",
        "int get_sum4() { return sum4; }",

        "void dinit(double *d, int i, int n) {",
        "  i < n ? (d[i] = (i+1)/100., dinit(d, i+1, n)) : (void)0;",
        "}",
        "void iinit(int *d, int i, int n) {",
        "  i < n ? (d[i] = i+1, iinit(d, i+1, n)) : (void)0;",
        "}",
        "void dcpy(double *d, double *s, int i, int n) {",
        "  i < n ? (d[i] = s[i], dcpy(d, s, i+1, n)) : (void)0;",
        "}",
        "void icpy(int *d, int *s, int i, int n) {",
        "  i < n ? (d[i] = s[i], icpy(d, s, i+1, n)) : (void)0;",
        "}",
        "void dsum(double *d, double *s, int i, int n) {",
        "  i < n ? (*d += s[i], dsum(d, s, i+1, n)) : (void)0;",
        "}",
        "void isum(int *d, int *s, int i, int n) {",
        "  i < n ? (*d += s[i], isum(d, s, i+1, n)) : (void)0;",
        "}",
        "void hardArgs(int n, ...) {",
        "  va_list args;",
        "  va_start(args, n);",
        // TODO: When we try to pass an actual struct instead of a pointer to
        // one, compiling the LLVM IR produces:
        //
        //   Unknown type!
        //   UNREACHABLE executed at /Users/jdenny/installs/llvm/3.2/src/lib/VMCore/ValueTypes.cpp:229!
        //   Stack dump:
        //   0.  Running pass 'X86 DAG->DAG Instruction Selection' on function '@hardArgs'
        //
        // Perhaps we're not handling enough of the x86_64 ABI here and LLVM's
        // x86_64 backend won't do the rest. The LLVM IR that clang generates
        // for va_arg stuff is much more complicated.
        "  struct S *s = va_arg(args, struct S*);",

        "  void (*dfn)(double*, double*, int, int)",
        "    = va_arg(args, void(*)(double*, double*, int, int));",
        "  void (*ifn)(int*, int*, int, int)",
        "    = va_arg(args, void(*)(int*, int*, int, int));",

        "  double (*da)[N] = va_arg(args, double(*)[N]);",
        "  int (*ia)[N] = va_arg(args, int(*)[N]);",

        "  dfn(*da, s->dNa, 0, N);",
        "  ifn(*ia, s->iNa, 0, N);",

        "  va_end(args);",
        "}",
        "void call_hardArgs() {",
        "  struct S s;",
        "  dinit(s.dNa, 0, N);",
        "  iinit(s.iNa, 0, N);",
        "  hardArgs(4, &s, dcpy, icpy, dNa, iNa);",
        "  dsum(&sum3, dNa, 0, N);",
        "  isum(&sum4, iNa, 0, N);",
        "}",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);

    runFn(exec, mod, "call_intArgs");
    checkIntFn(exec, mod, 165, "get_sum1");
    checkIntFn(exec, mod, 132, "get_sum2");

    runFn(exec, mod, "call_hardArgs");
    checkDoubleFn(exec, mod, 50.5, "get_sum3");
    checkIntFn(exec, mod, 5050, "get_sum4");

    exec.dispose();
  }

  @Test public void otherBuiltins() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = buildLLVMSimple(
      "", TARGET_DATA_LAYOUT,
      new String[]{
        "#include <stddef.h>",

        // fabs*

        "float  fabsf(float  v) { return __builtin_fabsf(v); }",
        "double fabs (double v) { return __builtin_fabs (v); }",
        // long double is not supported by LLVMGenericValueToFloat.
        // At least this checks implicit type conversions for arguments to
        // builtin functions.
        "double fabsl(double v) { return __builtin_fabsl(v); }",

        "size_t sizeof_fabsf = sizeof(__builtin_fabsf(5.f));",
        "size_t sizeof_fabs  = sizeof(__builtin_fabs (5. ));",
        "size_t sizeof_fabsl = sizeof(__builtin_fabsl(5.l));",

        // inf*

        "float  inff() { return __builtin_inff(); }",
        "double inf () { return __builtin_inf (); }",
        // long double is not supported by LLVMGenericValueToFloat.
        "double infl() { return __builtin_infl(); }",

        "size_t sizeof_inff = sizeof(__builtin_inff());",
        "size_t sizeof_inf  = sizeof(__builtin_inf ());",
        "size_t sizeof_infl = sizeof(__builtin_infl());",

        // alloca

        "int callAlloca() {",
        "  int *p = __builtin_alloca(sizeof (int));",
        "  *p = 5;",
        "  return *p;",
        "}",

        // __FUNCTION__ and __PRETTY_FUNCTION__
        //
        // Try it with at least two functions that don't just use sizeof so
        // we can make sure the globals they generate don't collide. Also,
        // within a function, make sure it's always the same global.

        "int strcmp(char*, char*);",
        "int functionName1() {",
        "  char *p1 = __FUNCTION__;",
        "  char *p2 = __PRETTY_FUNCTION__;",
        "  if (0 != strcmp(\"functionName1\", p1)) return 1;",
        "  if (0 != strcmp(p1, p2)) return 2;",
        "  if (0 != strcmp(p1, __FUNCTION__)) return 3;",
        "  if (0 != strcmp(p1, __PRETTY_FUNCTION__)) return 4;",
        "  if (p1 != p2) return 5;",
        "  if (p1 != __FUNCTION__) return 6;",
        "  if (p2 != __PRETTY_FUNCTION__) return 7;",
        "  return 0;",
        "}",
        "int functionName2() {",
        "  char *p1 = __FUNCTION__;",
        "  char *p2 = __PRETTY_FUNCTION__;",
        "  if (0 != strcmp(\"functionName2\", p1)) return 1;",
        "  if (0 != strcmp(\"functionName2\", p2)) return 2;",
        "  if (p1 != p2) return 3;",
        "  return 0;",
        "}",
        "size_t sizeofFunctionName() {",
        "  return sizeof __FUNCTION__;",
        "}",
        "size_t sizeofPrettyFunctionName() {",
        "  return sizeof __PRETTY_FUNCTION__;",
        "}",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    final long fSize = SrcFloatType.getLLVMType(ctxt).getPrimitiveSizeInBits()/8;
    final long dSize = SrcDoubleType.getLLVMType(ctxt).getPrimitiveSizeInBits()/8;
    final long ldSize = SIZEOF_LONG_DOUBLE;

    // fabs*

    checkFloatFn(exec, mod, 0.,    "fabsf", 0.);
    checkFloatFn(exec, mod, 3.5,   "fabsf", -3.5);
    checkFloatFn(exec, mod, 3e-10, "fabsf", 3e-10);
    checkIntGlobalVar(mod, "sizeof_fabsf", fSize);

    checkDoubleFn(exec, mod, 0.,    "fabs", 0.);
    checkDoubleFn(exec, mod, 3.5,   "fabs", -3.5);
    checkDoubleFn(exec, mod, 3e-10, "fabs", 3e-10);
    checkIntGlobalVar(mod, "sizeof_fabs", dSize);

    checkDoubleFn(exec, mod, 0.,    "fabsl", 0.);
    checkDoubleFn(exec, mod, 3.5,   "fabsl", -3.5);
    checkDoubleFn(exec, mod, 3e-10, "fabsl", 3e-10);
    checkIntGlobalVar(mod, "sizeof_fabsl", ldSize);

    // inf*

    checkFloatFn(exec, mod, Double.POSITIVE_INFINITY, "inff");
    checkDoubleFn(exec, mod, Double.POSITIVE_INFINITY, "inf");
    checkDoubleFn(exec, mod, Double.POSITIVE_INFINITY, "infl");

    checkIntGlobalVar(mod, "sizeof_inff", fSize);
    checkIntGlobalVar(mod, "sizeof_inf",  dSize);
    checkIntGlobalVar(mod, "sizeof_infl", ldSize);

    // alloca

    checkIntFn(exec, mod, "callAlloca", 5);

    // __FUNCTION__

    checkIntFn(exec, mod, "functionName1", 0);
    checkIntFn(exec, mod, "functionName2", 0);
    checkIntFn(exec, mod, "sizeofFunctionName",
               19*SrcCharType.getLLVMWidth()/8);
    checkIntFn(exec, mod, "sizeofPrettyFunctionName",
               25*SrcCharType.getLLVMWidth()/8);

    exec.dispose();
  }

  @Test public void statementExpression() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        "#include <stddef.h>",

        // Based on example from GCC 4.9.2 manual.
        "int incAbs(int val) {",
        "  val += 1;", // make sure this happens ahead of time
        "  return ({ int y = val; int z;",
        "            if (y > 0) z = y;",
        "            else z = - y;",
        "            z; });",
        "}",

        // Make sure it can be a void expression.
        "int sum() {",
        "  int res = 0;",
        "  for (int i = 0; i < 20; ({do { ++i; } while (i%4 != 0);}))",
        "    res += i;",
        "  return res;",
        "}",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);

    checkIntFn(exec, mod, 1, "incAbs", -2);
    checkIntFn(exec, mod, 6, "incAbs", 5);
    checkIntFn(exec, mod, 9, "incAbs", -10);

    checkIntFn(exec, mod, 40, "sum");

    exec.dispose();
  }
}
