package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedCharType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedIntType;

import java.math.BigInteger;

import org.jllvm.LLVMContext;
import org.jllvm.LLVMExecutionEngine;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMTargetData;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Checks the ability to build correct LLVM IR for C statements.
 * 
 * For controlling expressions, expressions with different types evaluated as
 * conditions are checked more thoroughly for ?:, &&, and || in
 * {@link BuildLLVMTest_OtherExpressions}.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class BuildLLVMTest_Statements extends BuildLLVMTest {
  @BeforeClass public static void setup() {
    System.loadLibrary("jllvm");
  }

  /**
   * Return statements are exercised in many parts of the test suite. Here,
   * we just check a few basic capabilities.
   */
  @Test public void returnStatement() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        // Check that type conversions are happening. Type conversions are
        // thoroughly checked for other cases in BuildLLVMTest_TypeConversions.
        "int intToDouble() { return 5.9; }",

        // Check that void functions work.
        "int sideEffect;",
        "int getSideEffect() { return sideEffect; }",
        "void setSideEffect(int i) { sideEffect = i; }",

        // Check that code after a return statement doesn't break the compiler
        // even though it's unreachable.
        "int afterReturnExpr() {",
        "  setSideEffect(88);",
        "  int i = 77;",
        "  return i;",
        "  i = 0;",
        "  setSideEffect(0);",
        "}",
        "void afterReturnVoid() {",
        "  setSideEffect(66);",
        "  return;",
        "  setSideEffect(0);",
        "}",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);

    checkIntFn(exec, mod, "intToDouble", 5);

    runFn(exec, mod, "setSideEffect", 99);
    checkIntFn(exec, mod, "getSideEffect", 99);

    checkIntFn(exec, mod, "afterReturnExpr", 77);
    checkIntFn(exec, mod, "getSideEffect", 88);
    runFn(exec, mod, "afterReturnVoid");
    checkIntFn(exec, mod, "getSideEffect", 66);

    exec.dispose();
  }

  @Test public void ifStatement() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        "int ifOnly(int i, int j) {",
        "  int res = 99;",
        "  if (i == j) res = 88;",
        "  return res;",
        "}",

        "int ifElse(int i) {",
        "  int res = 99;",
        "  if (i) res = 88;",
        "  else res = 77;",
        "  return res;",
        "}",

        "int ifElseIf(int i, int j, int k) {",
        "  int res = 99;",
        "  if (i == j) res = 88;",
        "  else if (i == k) res = 77;",
        "  return res;",
        "}",

        "int ifElseIfElse(int i, int j, int k) {",
        "  int res = 99;",
        "  if (i == j) res = 88;",
        "  else if (i == k) res = 77;",
        "  else res = 66;",
        "  return res;",
        "}",

        "int multipleBlocks(int i, int j, int k, int l, int m, int n, int o) {",
        "  int res = 99;",
        "  if (i && j) res = k ? 88 : 77;",
        "  else if (l || m) res = n ? 66 : 55;",
        "  else res = o ? 44 : 33;",
        "  return res;",
        "}",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);

    checkIntFn(exec, mod, 88, "ifOnly", 3, 3);
    checkIntFn(exec, mod, 99, "ifOnly", 3, 2);

    checkIntFn(exec, mod, 88, "ifElse", 5);
    checkIntFn(exec, mod, 77, "ifElse", 0);

    checkIntFn(exec, mod, 88, "ifElseIf", 3, 3, 5);
    checkIntFn(exec, mod, 77, "ifElseIf", 5, 3, 5);
    checkIntFn(exec, mod, 99, "ifElseIf", 7, 3, 5);

    checkIntFn(exec, mod, 88, "ifElseIfElse", 3, 3, 5);
    checkIntFn(exec, mod, 77, "ifElseIfElse", 5, 3, 5);
    checkIntFn(exec, mod, 66, "ifElseIfElse", 7, 3, 5);

    checkIntFn(exec, mod, 33, "multipleBlocks", 0,0,0, 0,0,0, 0);
    checkIntFn(exec, mod, 44, "multipleBlocks", 0,1,1, 0,0,1, 1);
    checkIntFn(exec, mod, 55, "multipleBlocks", 1,0,0, 0,1,0, 0);
    checkIntFn(exec, mod, 66, "multipleBlocks", 0,0,1, 1,0,1, 1);
    checkIntFn(exec, mod, 55, "multipleBlocks", 0,1,1, 1,1,0, 0);
    checkIntFn(exec, mod, 77, "multipleBlocks", 1,1,0, 0,1,0, 1);
    checkIntFn(exec, mod, 88, "multipleBlocks", 1,1,1, 0,0,1, 0);

    exec.dispose();
  }

  private static enum LoopKind {
    WHILE {
      String gen(String i, String c, String s, String b) {
        return i+"; while ("+c+") {"+b+" "+s+"; }";
      }
    },
    DO {
      String gen(String i, String c, String s, String b) {
        return i+"; do {"+b+" "+s+"; } while ("+c+");";
      }
    },
    FOR {
      String gen(String i, String c, String s, String b) {
        return "for ("+i+"; "+c+"; "+s+") {"+b+"}";
      }
    };
    abstract String gen(String i, String c, String s, String b);
  };
  private String[] genLoopTest(LoopKind kind) {
    return new String[]{
      "int a[100];",

      // One basic block in the controlling expression, and one basic block
      // in the body. Controlling expression is already a boolean. Init is
      // declaration.
      "void init1to100() {",
      "  " + kind.gen("int i = 0", "i < 100", "++i", "a[i] = i+1;"),
      "}",

      // One basic block in the controlling expression, and one basic block
      // in the body. Controlling expression isn't already a boolean.
      // Init is expression.
      "int sum1to100() {",
      "  int sum = 0, i;",
      "  " + kind.gen("i = 100", "i", "--i", "sum += a[i-1];"),
      "  return sum;",
      "}",

      // The conditional operator inserts additional basic blocks into the
      // body. Init is missing.
      "int sum1to100Even() {",
      "  int sum = 0, i = 0;",
      "  " + kind.gen("", "i < 100", "++i",
                      "sum += a[i]%2==0 ? a[i] : 0;"),
      "  return sum;",
      "}",

      // The && operator inserts additional basic blocks into the controlling
      // expression. Step is missing.
      "int sum1to100LtCap(int cap) {",
      "  int sum = 0;",
      "  " + kind.gen("int i = 0", "i < 100 && sum < cap", "",
                      "sum += a[i++];"),
      "  return sum;",
      "}",

      // Combine the last two.
      "int sum1to100EvenLtCap(int cap) {",
      "  int sum = 0, i = 0;",
      "  " + kind.gen("", "i < 100 && sum < cap", "",
                      "sum += a[i]%2==0 ? a[i] : 0; ++i;"),
      "  return sum;",
      "}",

      // Check continue and break.
      "int skipAndStop(int skip, int stop) {",
      "  int sum = 0;",
      "  " + kind.gen("int i = 1", "i <= 100", "++i",
                      "if (i == skip)"
                      + (kind != LoopKind.FOR ? "{++i; continue;}"
                                              : "continue;")
                      + " else if (i == stop) break;"
                      + " else sum += i;"),
      "  return sum;",
      "}",
    };
  }
  private void checkLoopTest(LoopKind kind, LLVMExecutionEngine exec,
                             LLVMModule mod)
  {
    runFn(exec, mod, "init1to100");
    checkIntFn(exec, mod, 5050, "sum1to100");
    checkIntFn(exec, mod, 2550, "sum1to100Even");
    // First operand (first bb) in controlling expression is false first.
    checkIntFn(exec, mod, 5050, "sum1to100LtCap", 6000);
    // Second operand (second bb) in controlling expression is false first.
    checkIntFn(exec, mod, 105, "sum1to100LtCap", 100);
    // Condition is always false, so zero iterations unless do-while, which
    // always has one iteration.
    checkIntFn(exec, mod, kind==LoopKind.DO ? 1 : 0, "sum1to100LtCap", 0);
    checkIntFn(exec, mod, 110, "sum1to100EvenLtCap", 100);
    checkIntFn(exec, mod, 2550, "sum1to100EvenLtCap", 3000);

    checkIntFn(exec, mod, 4748, "skipAndStop", 5, 98);
  }

  @Test public void whileStatement() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult
      = buildLLVMSimple("", "", genLoopTest(LoopKind.WHILE));
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);

    checkLoopTest(LoopKind.WHILE, exec, mod);

    exec.dispose();
  }

  @Test public void doStatement() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult
      = buildLLVMSimple("", "", genLoopTest(LoopKind.DO));
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);

    checkLoopTest(LoopKind.DO, exec, mod);

    exec.dispose();
  }

  @Test public void forStatement() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "", true,
      genLoopTest(LoopKind.FOR),
      new String[]{
        "int missingConditionIsTrue() {",
        "  int i, sum = 0;",
        "  for (i = 0;; ++i) {",
        "    sum += i+1;",
        "    if (i+1 >= 100)",
        "      return sum;",
        "  }",
        "  return 99;",
        "}",

        // Multiple blocks in init and step.
        "int initStepMultipleBlocks(int j, int k) {",
        "  int sum = 0;",
        "  for (int i = j && k; i < 100; i = i + 1 + (j || k))",
        "    sum += i+1;",
        "  return sum;",
        "}",

        // If we fail to discard the value from the step expression in a for
        // loop, the following will fail because a's stack allocation is pushed
        // to postOrderValuesAndTypes before initializers are evaluated and
        // then used afterward. I don't know of a way to see if the step
        // expressions was discarded without using a statement expression.
        "int discardStepValue() {",
        "  int a[2] = {11, (({for(;0;3);}), 22)};",
        "  return a[0] + a[1];",
        "}",

        // Cetus used to fail to parse this because it resolved the
        // expr/decl ambiguity here always in favor of expr.
        "typedef int T;",
        "int typedefStartsDecl() {",
        "  int sum = 0;",
        "  for (T i = 0; i < 100; ++i)",
        "    sum += i+1;",
        "  return sum;",
        "}",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);

    checkLoopTest(LoopKind.FOR, exec, mod);
    checkIntFn(exec, mod, 5050, "missingConditionIsTrue");
    checkIntFn(exec, mod, 5050, "initStepMultipleBlocks", 0, 0);
    checkIntFn(exec, mod, 2500, "initStepMultipleBlocks", 0, 1);
    checkIntFn(exec, mod, 2500, "initStepMultipleBlocks", 1, 0);
    checkIntFn(exec, mod, 2550, "initStepMultipleBlocks", 1, 1);
    checkIntFn(exec, mod, 33, "discardStepValue");
    checkIntFn(exec, mod, 5050, "typedefStartsDecl");

    exec.dispose();
  }

  private String[] genNestedLoopTest(LoopKind outerKind, LoopKind innerKind) {
    return new String[]{
      "int "+outerKind.name()+"_"+innerKind.name()
        +"(int outerSkip, int outerStop, int innerSkip, int innerStop) {",
      "  int sum = 0;",
      "  "
      +outerKind.gen(
         "int i = 0", "i <= 9", "++i",
         "if (i == outerSkip) "+(outerKind != LoopKind.FOR
                                 ? "{++i; continue;}" : "continue;")
         + " else if (i == outerStop) break; "
         + innerKind.gen(
             "int j = 1", "j <= 10", "++j",
             "if (j == innerSkip) "+(innerKind != LoopKind.FOR
                                     ? "{++j; continue;}" : "continue;")
             + " else if (j == innerStop) break;"
             + " sum += i*10 + j;")),
      "  return sum;",
      "}",
    };
  }
  /**
   * Check that loops can be nested without losing track of which continue or
   * break belongs to which loop.
   */
  @Test public void nestedLoops() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "", true,
      genNestedLoopTest(LoopKind.FOR, LoopKind.FOR),
      genNestedLoopTest(LoopKind.FOR, LoopKind.WHILE),
      genNestedLoopTest(LoopKind.DO, LoopKind.WHILE)
      );
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);

    checkIntFn(exec, mod, 5050, "FOR_FOR", 10, 10, 11, 11);
    checkIntFn(exec, mod, 166, "FOR_FOR", 1, 3, 5, 8);

    checkIntFn(exec, mod, 5050, "FOR_WHILE", 10, 10, 11, 11);
    checkIntFn(exec, mod, 166, "FOR_WHILE", 1, 3, 5, 8);

    checkIntFn(exec, mod, 5050, "DO_WHILE", 10, 10, 11, 11);
    checkIntFn(exec, mod, 166, "DO_WHILE", 1, 3, 5, 8);

    exec.dispose();
  }

  @Test public void gotoAndLabel() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        "int sum1to10() {",
        "  int i = 1, sum = 0;",
        "  loop: sum += i;", // same label in sum1to100
        "  if (++i <= 10) goto loop;",
        "  return sum;",
        "}",

        "int sum1to100() {",
        "  int i = 0, j, sum = 0;",
        "  loop:", // same label in sum1to10
        "  j = 1;",
        "  sum += i*10*10;", // add ten's place for every value of one's place
        "  goto loopInner;", // label in child scope, and forward reference
        "  {",
        "    return 99;", // dead code
        "    loopInner:",
        "    sum += j;",
        "    if (++j <= 10) goto loopInner;",
        "    if (++i <= 9) goto loop;", // label in parent scope
        "  }",
        "  return sum;",
        "}",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);

    checkIntFn(exec, mod, 55, "sum1to10");
    checkIntFn(exec, mod, 5050, "sum1to100");

    exec.dispose();
  }

  @Test public void switchStatement() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final BigInteger uMax
      = BigInteger.ONE.shiftLeft((int)SrcUnsignedIntType.getWidth())
        .subtract(BigInteger.ONE);
    final BigInteger ucMax
      = BigInteger.ONE.shiftLeft((int)SrcUnsignedCharType.getWidth())
        .subtract(BigInteger.ONE);
    final SimpleResult simpleResult = buildLLVMSimple(
      "", "",
      new String[]{
        "int sideEffect;",
        "void setSideEffect(int i) { sideEffect = i; }",
        "int getSideEffect() { return sideEffect; }",

        "int empty() {",
        "  setSideEffect(88);",
        "  switch (sideEffect) {}",
        "  return 77;",
        "}",

        "int unreachable(int i) {",
        "  int res = 99;",
        "  switch (i) {",
        "    setSideEffect(99);",
        "  default: res = 11; break;",
        "    setSideEffect(99);",
        "  case 0: res = 10; break;",
        "    setSideEffect(99);",
        "  }",
        "  return res;",
        "}",

        // Inner cases and default should not be confused with outer.
        "int nested(int i, int j) {",
        "  int res = 99;",
        "  switch (i) {",
        "    case 0: res = 10; break;",
        "    default:",
        "      switch (j) {",
        "      case 0: res = 20; break;",
        "      case 1: res = 21; break;",
        "      default: res = 22; break;",
        "      }",
        "      break;",
        "    case 1: res = 11; break;",
        "  }",
        "  return res;",
        "}",

        "int caseConvert(unsigned i) {",
        "  switch (i) {",
        "  case (signed char)-1:", // converts to unsigned int uMax
        "    return 88;",
        "  }",
        "  return 99;",
        "}",

        "int controlPromote1(unsigned char i) {",
        "  switch (i) {", // promotes to int
        "  case "+uMax.longValue()+"u:", // converts to int -1
        "    return 88;",
        "  case (unsigned char)"+ucMax.longValue()+":", // converts to int
        "    return 77;",
        "  }",
        "  return 99;",
        "}",

        "int controlPromote2(signed char i) {",
        "  switch (i) {", // promotes to int
        "  case "+uMax.longValue()+"u:", // converts to int -1
        "    return 88;",
        "  case (unsigned char)"+ucMax.longValue()+":", // converts to int
        "    return 77;",
        "  }",
        "  return 99;",
        "}",

        // Similar to example from ISO C99 sec. 6.8.4.2p7.
        "int isoExample(int expr) {",
        "  switch (expr)",
        "  {",
        "    int i = 4;", // unreachable
        "    setSideEffect(i);", // unreachable
        "  case 0:",
        "    i = 17;",
        "    /* falls through into default code */",
        "  default:",
        "    return i;",
        "  }",
        "}",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);

    runFn(exec, mod, "setSideEffect", 99);
    checkIntFn(exec, mod, 77, "empty");
    checkIntFn(exec, mod, 88, "getSideEffect");

    runFn(exec, mod, "setSideEffect", 88);
    checkIntFn(exec, mod, 10, "unreachable", 0);
    checkIntFn(exec, mod, 11, "unreachable", 1);
    checkIntFn(exec, mod, 11, "unreachable", 2);
    checkIntFn(exec, mod, 88, "getSideEffect");

    checkIntFn(exec, mod, 10, "nested", 0, 1);
    checkIntFn(exec, mod, 11, "nested", 1, 0);
    checkIntFn(exec, mod, 20, "nested", 2, 0);
    checkIntFn(exec, mod, 21, "nested", 3, 1);
    checkIntFn(exec, mod, 22, "nested", 4, 2);
    checkIntFn(exec, mod, 22, "nested", 5, 3);

    checkIntFn(exec, mod, 88, "caseConvert",
               getUnsignedIntGeneric(uMax.longValue(), ctxt));
    checkIntFn(exec, mod, 99, "caseConvert",
               getUnsignedIntGeneric(0, ctxt));

    checkIntFn(exec, mod, 77, "controlPromote1",
               getUnsignedCharGeneric(ucMax.longValue(), ctxt));
    checkIntFn(exec, mod, 88, "controlPromote2",
               getSignedCharGeneric(-1, ctxt)); // same bits as ucMax

    runFn(exec, mod, "setSideEffect", 99);
    checkIntFn(exec, mod, 17, "isoExample", 0);
    checkIntFn(exec, mod, 99, "getSideEffect");

    runFn(exec, mod, "setSideEffect", 99);
    runFn(exec, mod, "isoExample", 1); // indeterminate result
    checkIntFn(exec, mod, 99, "getSideEffect");

    exec.dispose();
  }
}
