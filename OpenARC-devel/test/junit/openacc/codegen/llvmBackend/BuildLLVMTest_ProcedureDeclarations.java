package openacc.codegen.llvmBackend;

import static org.jllvm.bindings.LLVMAttribute.*;
import static org.jllvm.bindings.LLVMLinkage.*;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.io.IOException;

import org.hamcrest.CoreMatchers;
import org.jllvm.LLVMFunction;
import org.jllvm.bindings.LLVMAttribute;
import org.jllvm.bindings.LLVMLinkage;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Checks the ability to generate a single correct LLVM function for a
 * a procedure that's declared/defined multiple times in the source code.
 * 
 * <p>
 * The number of parameters, variadicity, linkage, inline hint, and
 * declared/defined status of the LLVM function that results from one or more
 * declarations of the same C function are checked here. Parameter types and
 * return types are checked in {@link BuildLLVMTest_Types}, and function
 * behavior is checked in many other test groups.
 * </p>
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class BuildLLVMTest_ProcedureDeclarations extends BuildLLVMTest {
  @BeforeClass public static void setup() {
    System.loadLibrary("jllvm");
  }

  private static class ExpectParams {
    public ExpectParams(String functionName, int nParams, boolean varArgs,
                        boolean defined)
    {
      this.functionName = functionName;
      this.nParams = nParams;
      this.varArgs = varArgs;
      this.defined = defined;
    }
    public final String functionName;
    public final int nParams;
    public final boolean varArgs;
    public final boolean defined;
  }

  private void checkParams(ExpectParams[] expects, String[] decls)
    throws IOException
  {
    String[] src = new String[decls.length + 1];
    System.arraycopy(decls, 0, src, 0, decls.length);
    src[decls.length] = "int main() { return 0; }";
    final SimpleResult simpleResult = buildLLVMSimple("", "", src);
    for (ExpectParams expect : expects) {
      LLVMFunction func
        = simpleResult.llvmModule.getNamedFunction(expect.functionName);
      assertNotNull(expect.functionName + " must exist", func.getInstance());
      assertEquals(expect.functionName + "'s parameter count",
                   expect.nParams, func.countParameters());
      assertEquals(expect.functionName + "'s variadicity",
                   expect.varArgs, func.getFunctionType().isVarArg());
      assertEquals(expect.functionName + " definition's existence",
                   expect.defined, !func.isDeclaration());
    }
    assertNotNull("main must exist",
                  simpleResult.llvmModule.getNamedFunction("main"));
    // If we accidentally generated a function multiple times, LLVM would
    // assign new names to later occurrences, so make sure we have exactly the
    // number of functions we expect.
    int nGlobals = 0;
    for (LLVMFunction global = simpleResult.llvmModule.getFirstFunction();
         global.getInstance() != null;
         global = global.getNextFunction())
      ++nGlobals;
    assertEquals("function count", expects.length+1, nGlobals);
  }

  private String getMsgForVar_0(String fn) {
    return
      "incompatible declarations of function \"" + fn + "\": function types"
      + " are incompatible because one has an unspecified parameter list"
      + " while the other is variadic";
  }

  // Declared with 0 parameters first.
  @Test public void params_decl0First() throws IOException {
    checkParams(
      new ExpectParams[]{
        new ExpectParams("decl0",         0, true,  false),
        new ExpectParams("decl0_0",       0, true,  false),
        new ExpectParams("decl0_Void",    0, false, false),
        new ExpectParams("decl0_1",       1, false, false),
        new ExpectParams("decl0_2",       2, false, false),
        new ExpectParams("decl0_2_0",     2, false, false),
        new ExpectParams("decl0_def0",    0, false, true),
        new ExpectParams("decl0_def1",    1, false, true),
        new ExpectParams("decl0_def2",    2, false, true),
      },
      new String[]{
        "void decl0();",
        "void decl0_0();",
        "void decl0_0();",
        "void decl0_Void();",
        "void decl0_Void(void);",
        "void decl0_1();",
        "void decl0_1(int);",
        "void decl0_2();",
        "void decl0_2(int, int);",
        "void decl0_2_0();",
        "void decl0_2_0(int, int);",
        "void decl0_2_0();",
        "void decl0_def0();",
        "void decl0_def0() {};",
        "void decl0_def1();",
        "void decl0_def1(int x) {}",
        "void decl0_def2();",
        "void decl0_def2(int x, int y) {}",
      });
  }

  @Test public void params_decl0_def2Var() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      getMsgForVar_0("decl0_def2Var")));
    buildLLVMSimple("", "",
      "void decl0_def2Var();",
      "void decl0_def2Var(int, int, ...) {}");
  }

  // Declared with void parameters first.
  @Test public void params_declVoidFirst() throws IOException {
    checkParams(
      new ExpectParams[]{
        new ExpectParams("declVoid",      0, false, false),
        new ExpectParams("declVoid_0",    0, false, false),
        new ExpectParams("declVoid_Void", 0, false, false),
        new ExpectParams("declVoid_def0", 0, false, true),
      },
      new String[]{
        "void declVoid(void);",
        "void declVoid_0(void);",
        "void declVoid_0();",
        "void declVoid_Void(void);",
        "void declVoid_Void(void);",
        "void declVoid_def0(void);",
        "void declVoid_def0() {};",
      });
  }

  // Declared with 1 parameter first.
  @Test public void params_decl1First() throws IOException {
    checkParams(
      new ExpectParams[]{
        new ExpectParams("decl1",      1, false, false),
        new ExpectParams("decl1_0",    1, false, false),
        new ExpectParams("decl1_1",    1, false, false),
        new ExpectParams("decl1_def1", 1, false, true),
      },
      new String[]{
        "void decl1(int);",
        "void decl1_0(int);",
        "void decl1_0();",
        "void decl1_1(int);",
        "void decl1_1(int);",
        "void decl1_def1(int);",
        "void decl1_def1(int x) {}",
      });
  }

  // Declared with 2 parameters first.
  @Test public void params_decl2First() throws IOException {
    checkParams(
      new ExpectParams[]{
        new ExpectParams("decl2",      2, false, false),
        new ExpectParams("decl2_0",    2, false, false),
        new ExpectParams("decl2_2",    2, false, false),
        new ExpectParams("decl2_def2", 2, false, true),
      },
      new String[]{
        "void decl2(int, int);",
        "void decl2_0(int, int);",
        "void decl2_0();",
        "void decl2_2(int, int);",
        "void decl2_2(int, int);",
        "void decl2_def2(int, int);",
        "void decl2_def2(int x, int y) {}",
      });
  }

  // Declared with 1 parameter and variadic first.
  @Test public void params_decl1VarFirst() throws IOException {
    checkParams(
      new ExpectParams[]{
        new ExpectParams("decl1Var",         1, true, false),
        new ExpectParams("decl1Var_1Var",    1, true, false),
        new ExpectParams("decl1Var_def1Var", 1, true, true),
      },
      new String[]{
        "void decl1Var(int, ...);",
        "void decl1Var_1Var(int, ...);",
        "void decl1Var_1Var(int, ...);",
        "void decl1Var_def1Var(int, ...);",
        "void decl1Var_def1Var(int x, ...) {}",
      });
  }

  @Test public void params_decl1Var_0() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      getMsgForVar_0("decl1Var_0")));
    buildLLVMSimple("", "",
      "void decl1Var_0(int, ...);",
      "void decl1Var_0();");
  }

  // Declared with 2 parameters and variadic first.
  @Test public void params_decl2VarFirst() throws IOException {
    checkParams(
      new ExpectParams[]{
        new ExpectParams("decl2Var",         2, true, false),
        new ExpectParams("decl2Var_2Var",    2, true, false),
        new ExpectParams("decl2Var_def2Var", 2, true, true),
      },
      new String[]{
        "void decl2Var(int, int, ...);",
        "void decl2Var_2Var(int, int, ...);",
        "void decl2Var_2Var(int, int, ...);",
        "void decl2Var_def2Var(int, int, ...);",
        "void decl2Var_def2Var(int x, int y, ...) {}",
      });
  }

  @Test public void params_decl2Var_0_2Var() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      getMsgForVar_0("decl2Var_0_2Var")));
    buildLLVMSimple("", "",
      "void decl2Var_0_2Var(int, int, ...);",
      "void decl2Var_0_2Var();",
      "void decl2Var_0_2Var(int, int, ...);");
  }

  // Defined with 0 parameters first.
  @Test public void params_def0First() throws IOException {
    checkParams(
      new ExpectParams[]{
        new ExpectParams("def0",          0, false, true),
        new ExpectParams("def0_decl0",    0, false, true),
        new ExpectParams("def0_declVoid", 0, false, true),
      },
      new String[]{
        "void def0() {}",
        "void def0_decl0() {}",
        "void def0_decl0();",
        "void def0_declVoid() {}",
        "void def0_declVoid(void);",
      });
  }

  // Defined with void parameters first.
  @Test public void params_defVoidFirst() throws IOException {
    checkParams(
      new ExpectParams[]{
        new ExpectParams("defVoid",          0, false, true),
        new ExpectParams("defVoid_decl0",    0, false, true),
        new ExpectParams("defVoid_declVoid", 0, false, true),
      },
      new String[]{
        "void defVoid(void) {}",
        "void defVoid_decl0(void) {}",
        "void defVoid_decl0();",
        "void defVoid_declVoid(void) {}",
        "void defVoid_declVoid(void);",
      });
  }

  // Defined with 1 parameter first.
  @Test public void params_def1First() throws IOException {
    checkParams(
      new ExpectParams[]{
        new ExpectParams("def1",       1, false, true),
        new ExpectParams("def1_decl0", 1, false, true),
        new ExpectParams("def1_decl1", 1, false, true),
      },
      new String[]{
        "void def1(int x) {}",
        "void def1_decl0(int x) {}",
        "void def1_decl0();",
        "void def1_decl1(int x) {}",
        "void def1_decl1(int);",
      });
  }

  // Defined with 2 parameters first.
  @Test public void params_def2First() throws IOException {
    checkParams(
      new ExpectParams[]{
        new ExpectParams("def2",       2, false, true),
        new ExpectParams("def2_decl0", 2, false, true),
        new ExpectParams("def2_decl2", 2, false, true),
      },
      new String[]{
        "void def2(int x, int y) {}",
        "void def2_decl0(int x, int y) {}",
        "void def2_decl0();",
        "void def2_decl2(int x, int y) {}",
        "void def2_decl2(int x, int y);",
      });
  }

  // Defined with 1 parameter and variadic first.
  @Test public void params_def1VarFirst() throws IOException {
    checkParams(
      new ExpectParams[]{
        new ExpectParams("def1Var",          1, true, true),
        new ExpectParams("def1Var_decl1Var", 1, true, true),
      },
      new String[]{
        "void def1Var(int x, ...) {}",
        "void def1Var_decl1Var(int x, ...) {}",
        "void def1Var_decl1Var(int, ...);",
      });
  }

  @Test public void params_def1Var_decl0() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      getMsgForVar_0("def1Var_decl0")));
    buildLLVMSimple("", "",
      "void def1Var_decl0(int x, ...) {}",
      "void def1Var_decl0();");
  }

  // Defined with 2 parameters and variadic first.
  @Test public void params_def2VarFirst() throws IOException {
    checkParams(
      new ExpectParams[]{
        new ExpectParams("def2Var",          2, true, true),
        new ExpectParams("def2Var_decl2Var", 2, true, true),
      },
      new String[]{
        "void def2Var(int x, int y, ...) {}",
        "void def2Var_decl2Var(int x, int y, ...) {}",
        "void def2Var_decl2Var(int, int y, ...);",
      });
  }

  @Test public void params_def2Var_decl0() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      getMsgForVar_0("def2Var_decl0")));
    buildLLVMSimple("", "",
      "void def2Var_decl0(int x, int y, ...) {}",
      "void def2Var_decl0();");
  }

  @Test public void params_localDecl() throws IOException {
    checkParams(
      new ExpectParams[]{
        new ExpectParams("declOutIn", 1, false, false),
        new ExpectParams("defOutIn", 1, false, true),
        new ExpectParams("declOutShadowIn", 0, true, false),
        new ExpectParams("defOutShadowIn", 1, false, true),
        new ExpectParams("fn", 0, false, true),
        new ExpectParams("shadowInOnly", 1, true, false),
        new ExpectParams("shadowInDeclOut", 1, false, false),
        new ExpectParams("shadowInDefOut", 1, false, true),
        new ExpectParams("inOnly", 1, false, false),
        new ExpectParams("inDeclOut", 1, false, false),
        new ExpectParams("inDefOut", 1, false, true),
      },
      new String[]{
        "void declOutIn(int);",
        "void defOutIn(int) {}",
        "void declOutShadowIn();",
        "void defOutShadowIn(int) {}",
        "void fn() {",
        "  void declOutIn();",
        "  void defOutIn();",
        "  int declOutShadowIn;",
        "  int defOutShadowIn;",
        "  int shadowInOnly;",
        "  int shadowInDeclOut;",
        "  int shadowInDefOut;",
        "  {",
        "    void declOutShadowIn(int);",
        "    void defOutShadowIn();",
        "    void shadowInOnly(int, ...);",
        "    void shadowInDeclOut();",
        "    void shadowInDefOut();",
        "  }",
        "  void inOnly(int);",
        "  void inDeclOut();",
        "  void inDefOut();",
        "}",
        "void shadowInDeclOut(int);",
        "void shadowInDefOut(int) {}",
        "void inDeclOut(int);",
        "void inDefOut(int) {}",
      });
  }

  private static class ExpectLinkage {
    public ExpectLinkage(String functionName, LLVMLinkage linkage,
                         boolean defined, boolean inlineHint)
    {
      this(functionName, linkage, defined,
           inlineHint ? new LLVMAttribute[]{LLVMInlineHintAttribute}
                      : new LLVMAttribute[]{});
    }
    public ExpectLinkage(String functionName, LLVMLinkage linkage,
                         boolean defined, LLVMAttribute... attributes)
    {
      this.functionName = functionName;
      this.linkage = linkage;
      this.defined = defined;
      this.attributes = attributes;
    }
    public final String functionName;
    public final LLVMLinkage linkage;
    public final boolean defined;
    public final LLVMAttribute[] attributes;
  }

  private void checkLinkage(ExpectLinkage[] expects, String[] decls)
    throws IOException
  {
    String[] src = new String[decls.length + 1];
    System.arraycopy(decls, 0, src, 0, decls.length);
    src[decls.length] = "int main() { return 0; }";
    // Warnings not treated as errors because of warnings about externally
    // linked inline functions without definitions.
    final SimpleResult simpleResult = buildLLVMSimple("", "", false, src);
    for (ExpectLinkage expect : expects) {
      LLVMFunction func
        = simpleResult.llvmModule.getNamedFunction(expect.functionName);
      assertNotNull(expect.functionName + " must exist", func.getInstance());
      assertEquals(expect.functionName + "'s linkage",
                   expect.linkage, func.getLinkage());
      assertEquals(expect.functionName + " definition's existence",
                   expect.defined, !func.isDeclaration());
      boolean expectsInlineHint = false;
      boolean expectsAlwaysInline = false;
      for (LLVMAttribute attr : expect.attributes) {
        assertTrue(expect.functionName + " must have attribute "
                   + attr.toString(),
                   func.hasAttribute(attr));
        if (attr == LLVMInlineHintAttribute)
          expectsInlineHint = true;
        if (attr == LLVMAlwaysInlineAttribute)
          expectsAlwaysInline = true;
      }
      if (!expectsInlineHint)
        assertFalse(expect.functionName + " must not have attribute "
                    + LLVMInlineHintAttribute.toString(),
                    func.hasAttribute(LLVMInlineHintAttribute));
      if (!expectsAlwaysInline)
        assertFalse(expect.functionName + " must not have attribute "
                    + LLVMAlwaysInlineAttribute.toString(),
                    func.hasAttribute(LLVMAlwaysInlineAttribute));
    }
    assertNotNull("main must exist",
                  simpleResult.llvmModule.getNamedFunction("main"));
    // If we accidentally generated a function multiple times, LLVM would
    // assign new names to later occurrences, so make sure we have exactly the
    // number of functions we expect.
    int nGlobals = 0;
    for (LLVMFunction global = simpleResult.llvmModule.getFirstFunction();
         global.getInstance() != null;
         global = global.getNextFunction())
      ++nGlobals;
    assertEquals("function count", expects.length+1, nGlobals);
  }

  @Test public void linkage_once() throws IOException {
    checkLinkage(
      new ExpectLinkage[]{
        new ExpectLinkage("decl",   LLVMExternalLinkage, false),
        new ExpectLinkage("def",    LLVMExternalLinkage, true),
        new ExpectLinkage("exDecl", LLVMExternalLinkage, false),
        new ExpectLinkage("exDef",  LLVMExternalLinkage, true),
        // This is stripped out because it has internal linkage, no
        // definition, and no uses, so LLVM module verification would complain
        // if we didn't strip it out.
        //new ExpectLinkage("inDecl", LLVMInternalLinkage, false),
        new ExpectLinkage("inDef",  LLVMInternalLinkage, true),

        new ExpectLinkage("exInlnDecl", LLVMExternalLinkage, false, LLVMInlineHintAttribute),
        // When there's no definition, LLVM does not permit
        // LLVMAvailableExternallyLinkage.
        new ExpectLinkage("inlnDecl",   LLVMExternalLinkage, false, LLVMInlineHintAttribute),
        // Stripped out for the same reason as the one above.
        //new ExpectLinkage("inInlnDecl", LLVMInternalLinkage, false, true),

        new ExpectLinkage("exInlnDef", LLVMExternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("inlnDef", LLVMAvailableExternallyLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("inInlnDef", LLVMInternalLinkage, true, LLVMInlineHintAttribute),
      },
      new String[]{
        "void decl(int);",
        "void def() {}",
        "extern float exDecl(int, ...);",
        "extern int exDef(int, int) { return 0; }",
        "static int inDecl();",
        "static void inDef(int, int) {}",

        "extern inline void exInlnDecl();",
        "inline void inlnDecl();",
        "static inline void inInlnDecl();",

        "extern inline void exInlnDef() {}",
        "inline void inlnDef() {}",
        "static inline void inInlnDef() {}",
      });
  }

  @Test public void linkage_defFirst() throws IOException {
    checkLinkage(
      new ExpectLinkage[]{
        new ExpectLinkage("def_decl",          LLVMExternalLinkage, true),
        new ExpectLinkage("def_exDecl",        LLVMExternalLinkage, true),
        new ExpectLinkage("def_inlnDecl",      LLVMExternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("def_exInlnDecl",    LLVMExternalLinkage, true, LLVMInlineHintAttribute),

        new ExpectLinkage("exDef_decl",        LLVMExternalLinkage, true),
        new ExpectLinkage("exDef_exDecl",      LLVMExternalLinkage, true),
        new ExpectLinkage("exDef_inlnDecl",    LLVMExternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("exDef_exInlnDecl",  LLVMExternalLinkage, true, LLVMInlineHintAttribute),

        new ExpectLinkage("inDef_decl",        LLVMInternalLinkage, true),
        new ExpectLinkage("inDef_inDecl",      LLVMInternalLinkage, true),
        new ExpectLinkage("inDef_exDecl",      LLVMInternalLinkage, true),
        new ExpectLinkage("inDef_inlnDecl",    LLVMInternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("inDef_inInlnDecl",  LLVMInternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("inDef_exInlnDecl",  LLVMInternalLinkage, true, LLVMInlineHintAttribute),

        new ExpectLinkage("inDef_decl_exDecl", LLVMInternalLinkage, true),

        new ExpectLinkage("inlnDef_decl",       LLVMExternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("inlnDef_exDecl",     LLVMExternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("inlnDef_inlnDecl", LLVMAvailableExternallyLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("inlnDef_exInlnDecl", LLVMExternalLinkage, true, LLVMInlineHintAttribute),

        new ExpectLinkage("exInlnDef_decl",       LLVMExternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("exInlnDef_exDecl",     LLVMExternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("exInlnDef_inlnDecl",   LLVMExternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("exInlnDef_exInlnDecl", LLVMExternalLinkage, true, LLVMInlineHintAttribute),

        new ExpectLinkage("inInlnDef_decl",       LLVMInternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("inInlnDef_inDecl",     LLVMInternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("inInlnDef_exDecl",     LLVMInternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("inInlnDef_inlnDecl",   LLVMInternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("inInlnDef_inInlnDecl", LLVMInternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("inInlnDef_exInlnDecl", LLVMInternalLinkage, true, LLVMInlineHintAttribute),
      },
      new String[]{
        "void def_decl() {}",
        "void def_decl();",
        "void def_exDecl() {}",
        "extern void def_exDecl();",
        "void def_inlnDecl() {}",
        "inline void def_inlnDecl();",
        "void def_exInlnDecl() {}",
        "extern inline void def_exInlnDecl();",

        "extern void exDef_decl() {}",
        "void exDef_decl();",
        "extern void exDef_exDecl() {}",
        "extern void exDef_exDecl();",
        "extern void exDef_inlnDecl() {}",
        "inline void exDef_inlnDecl();",
        "extern void exDef_exInlnDecl() {}",
        "extern inline void exDef_exInlnDecl();",

        "static void inDef_decl() {}",
        "void inDef_decl();",
        "static void inDef_inDecl() {}",
        "static void inDef_inDecl();",
        "static void inDef_exDecl() {}",
        "extern void inDef_exDecl();",
        "static void inDef_inlnDecl() {}",
        "inline void inDef_inlnDecl();",
        "static void inDef_inInlnDecl() {}",
        "static inline void inDef_inInlnDecl();",
        "static void inDef_exInlnDecl() {}",
        "extern inline void inDef_exInlnDecl();",

        "static void inDef_decl_exDecl() {}",
        "void inDef_decl_exDecl();",
        "extern void inDef_decl_exDecl();",

        "inline void inlnDef_decl() {}",
        "void inlnDef_decl();",
        "inline void inlnDef_exDecl() {}",
        "extern void inlnDef_exDecl();",
        "inline void inlnDef_inlnDecl() {}",
        "inline void inlnDef_inlnDecl();",
        "inline void inlnDef_exInlnDecl() {}",
        "extern inline void inlnDef_exInlnDecl();",

        "extern inline void exInlnDef_decl() {}",
        "void exInlnDef_decl();",
        "extern inline void exInlnDef_exDecl() {}",
        "extern void exInlnDef_exDecl();",
        "extern inline void exInlnDef_inlnDecl() {}",
        "inline void exInlnDef_inlnDecl();",
        "extern inline void exInlnDef_exInlnDecl() {}",
        "extern inline void exInlnDef_exInlnDecl();",

        "static inline void inInlnDef_decl() {}",
        "void inInlnDef_decl();",
        "static inline void inInlnDef_inDecl() {}",
        "static void inInlnDef_inDecl();",
        "static inline void inInlnDef_exDecl() {}",
        "extern void inInlnDef_exDecl();",
        "static inline void inInlnDef_inlnDecl() {}",
        "inline void inInlnDef_inlnDecl();",
        "static inline void inInlnDef_inInlnDecl() {}",
        "static inline void inInlnDef_inInlnDecl();",
        "static inline void inInlnDef_exInlnDecl() {}",
        "extern inline void inInlnDef_exInlnDecl();",
      });
  }

  @Test public void linkage_declFirst() throws IOException {
    checkLinkage(
      new ExpectLinkage[]{
        new ExpectLinkage("decl_decl",         LLVMExternalLinkage, false),
        new ExpectLinkage("decl_exDecl",       LLVMExternalLinkage, false),
        new ExpectLinkage("decl_inlnDecl",     LLVMExternalLinkage, false, LLVMInlineHintAttribute),
        new ExpectLinkage("decl_exInlnDecl",   LLVMExternalLinkage, false, LLVMInlineHintAttribute),
        new ExpectLinkage("decl_def",          LLVMExternalLinkage, true),
        new ExpectLinkage("decl_exDef",        LLVMExternalLinkage, true),
        new ExpectLinkage("decl_inlnDef",      LLVMExternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("decl_exInlnDef",    LLVMExternalLinkage, true, LLVMInlineHintAttribute),

        new ExpectLinkage("exDecl_decl",       LLVMExternalLinkage, false),
        new ExpectLinkage("exDecl_exDecl",     LLVMExternalLinkage, false),
        new ExpectLinkage("exDecl_inlnDecl",   LLVMExternalLinkage, false, LLVMInlineHintAttribute),
        new ExpectLinkage("exDecl_exInlnDecl", LLVMExternalLinkage, false, LLVMInlineHintAttribute),
        new ExpectLinkage("exDecl_def",        LLVMExternalLinkage, true),
        new ExpectLinkage("exDecl_exDef",      LLVMExternalLinkage, true),
        new ExpectLinkage("exDecl_inlnDef",    LLVMExternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("exDecl_exInlnDef",  LLVMExternalLinkage, true, LLVMInlineHintAttribute),

        // These are stripped out because they have internal linkage, no
        // definition, and no uses, so LLVM module verification would complain
        // if we didn't strip them out.
        //new ExpectLinkage("inDecl_decl",       LLVMInternalLinkage, false),
        //new ExpectLinkage("inDecl_inDecl",     LLVMInternalLinkage, false),
        //new ExpectLinkage("inDecl_exDecl",     LLVMInternalLinkage, false),
        //new ExpectLinkage("inDecl_inlnDecl",   LLVMInternalLinkage, false, LLVMInlineHintAttribute),
        //new ExpectLinkage("inDecl_inInlnDecl", LLVMInternalLinkage, false, LLVMInlineHintAttribute),
        //new ExpectLinkage("inDecl_exInlnDecl", LLVMInternalLinkage, false, LLVMInlineHintAttribute),

        new ExpectLinkage("inDecl_def",        LLVMInternalLinkage, true),
        new ExpectLinkage("inDecl_inDef",      LLVMInternalLinkage, true),
        new ExpectLinkage("inDecl_exDef",      LLVMInternalLinkage, true),
        new ExpectLinkage("inDecl_inlnDef",    LLVMInternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("inDecl_inInlnDef",  LLVMInternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("inDecl_exInlnDef",  LLVMInternalLinkage, true, LLVMInlineHintAttribute),

        new ExpectLinkage("inlnDecl_decl",       LLVMExternalLinkage, false, LLVMInlineHintAttribute),
        new ExpectLinkage("inlnDecl_exDecl",     LLVMExternalLinkage, false, LLVMInlineHintAttribute),
        new ExpectLinkage("inlnDecl_inlnDecl",   LLVMExternalLinkage, false, LLVMInlineHintAttribute),
        new ExpectLinkage("inlnDecl_exInlnDecl", LLVMExternalLinkage, false, LLVMInlineHintAttribute),
        new ExpectLinkage("inlnDecl_def",        LLVMExternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("inlnDecl_exDef",      LLVMExternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("inlnDecl_inlnDef", LLVMAvailableExternallyLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("inlnDecl_exInlnDef",  LLVMExternalLinkage, true, LLVMInlineHintAttribute),

        new ExpectLinkage("exInlnDecl_decl",       LLVMExternalLinkage, false, LLVMInlineHintAttribute),
        new ExpectLinkage("exInlnDecl_exDecl",     LLVMExternalLinkage, false, LLVMInlineHintAttribute),
        new ExpectLinkage("exInlnDecl_inlnDecl",   LLVMExternalLinkage, false, LLVMInlineHintAttribute),
        new ExpectLinkage("exInlnDecl_exInlnDecl", LLVMExternalLinkage, false, LLVMInlineHintAttribute),
        new ExpectLinkage("exInlnDecl_def",        LLVMExternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("exInlnDecl_exDef",      LLVMExternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("exInlnDecl_inlnDef",    LLVMExternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("exInlnDecl_exInlnDef",  LLVMExternalLinkage, true, LLVMInlineHintAttribute),

        // These are stripped out because they have internal linkage, no
        // definition, and no uses, so LLVM module verification would complain
        // if we didn't strip them out.
        //new ExpectLinkage("inInlnDecl_decl",       LLVMInternalLinkage, false, LLVMInlineHintAttribute),
        //new ExpectLinkage("inInlnDecl_inDecl",     LLVMInternalLinkage, false, LLVMInlineHintAttribute),
        //new ExpectLinkage("inInlnDecl_exDecl",     LLVMInternalLinkage, false, LLVMInlineHintAttribute),
        //new ExpectLinkage("inInlnDecl_inlnDecl",   LLVMInternalLinkage, false, LLVMInlineHintAttribute),
        //new ExpectLinkage("inInlnDecl_inInlnDecl", LLVMInternalLinkage, false, LLVMInlineHintAttribute),
        //new ExpectLinkage("inInlnDecl_exInlnDecl", LLVMInternalLinkage, false, LLVMInlineHintAttribute),

        new ExpectLinkage("inInlnDecl_def",        LLVMInternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("inInlnDecl_inDef",      LLVMInternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("inInlnDecl_exDef",      LLVMInternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("inInlnDecl_inlnDef",    LLVMInternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("inInlnDecl_inInlnDef",  LLVMInternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("inInlnDecl_exInlnDef",  LLVMInternalLinkage, true, LLVMInlineHintAttribute),

        new ExpectLinkage("inDecl_decl_exDef", LLVMInternalLinkage, true),
        new ExpectLinkage("inDecl_def_exDecl", LLVMInternalLinkage, true),

        new ExpectLinkage("inDecl2nd_def",     LLVMInternalLinkage, true),
      },
      new String[]{
        "void decl_decl();",
        "void decl_decl();",
        "void decl_exDecl();",
        "extern void decl_exDecl();",
        "void decl_inlnDecl();",
        "inline void decl_inlnDecl();",
        "void decl_exInlnDecl();",
        "extern inline void decl_exInlnDecl();",
        "void decl_def();",
        "void decl_def() {};",
        "void decl_exDef();",
        "extern void decl_exDef() {}",
        "void decl_inlnDef();",
        "inline void decl_inlnDef() {}",
        "void decl_exInlnDef();",
        "extern inline void decl_exInlnDef() {}",

        "extern void exDecl_decl();",
        "void exDecl_decl();",
        "extern void exDecl_exDecl();",
        "extern void exDecl_exDecl();",
        "extern void exDecl_inlnDecl();",
        "inline void exDecl_inlnDecl();",
        "extern void exDecl_exInlnDecl();",
        "extern inline void exDecl_exInlnDecl();",
        "extern void exDecl_def();",
        "void exDecl_def() {}",
        "extern void exDecl_exDef();",
        "extern void exDecl_exDef() {}",
        "extern void exDecl_inlnDef();",
        "inline void exDecl_inlnDef() {}",
        "extern void exDecl_exInlnDef();",
        "extern inline void exDecl_exInlnDef() {}",

        "static void inDecl_decl();",
        "void inDecl_decl();",
        "static void inDecl_inDecl();",
        "static void inDecl_inDecl();",
        "static void inDecl_exDecl();",
        "extern void inDecl_exDecl();",
        "static void inDecl_inlnDecl();",
        "inline void inDecl_inlnDecl();",
        "static void inDecl_inInlnDecl();",
        "static inline void inDecl_inInlnDecl();",
        "static void inDecl_exInlnDecl();",
        "extern inline void inDecl_exInlnDecl();",

        "static void inDecl_def();",
        "void inDecl_def() {}",
        "static void inDecl_inDef();",
        "static void inDecl_inDef() {}",
        "static void inDecl_exDef();",
        "extern void inDecl_exDef() {}",
        "static void inDecl_inlnDef();",
        "inline void inDecl_inlnDef() {}",
        "static void inDecl_inInlnDef();",
        "static inline void inDecl_inInlnDef() {}",
        "static void inDecl_exInlnDef();",
        "extern inline void inDecl_exInlnDef() {}",

        "inline void inlnDecl_decl();",
        "void inlnDecl_decl();",
        "inline void inlnDecl_exDecl();",
        "extern void inlnDecl_exDecl();",
        "inline void inlnDecl_inlnDecl();",
        "inline void inlnDecl_inlnDecl();",
        "inline void inlnDecl_exInlnDecl();",
        "extern inline void inlnDecl_exInlnDecl();",
        "inline void inlnDecl_def();",
        "void inlnDecl_def() {};",
        "inline void inlnDecl_exDef();",
        "extern void inlnDecl_exDef() {}",
        "inline void inlnDecl_inlnDef();",
        "inline void inlnDecl_inlnDef() {}",
        "inline void inlnDecl_exInlnDef();",
        "extern inline void inlnDecl_exInlnDef() {}",

        "extern inline void exInlnDecl_decl();",
        "void exInlnDecl_decl();",
        "extern inline void exInlnDecl_exDecl();",
        "extern void exInlnDecl_exDecl();",
        "extern inline void exInlnDecl_inlnDecl();",
        "inline void exInlnDecl_inlnDecl();",
        "extern inline void exInlnDecl_exInlnDecl();",
        "extern inline void exInlnDecl_exInlnDecl();",
        "extern inline void exInlnDecl_def();",
        "void exInlnDecl_def() {}",
        "extern inline void exInlnDecl_exDef();",
        "extern void exInlnDecl_exDef() {}",
        "extern inline void exInlnDecl_inlnDef();",
        "inline void exInlnDecl_inlnDef() {}",
        "extern inline void exInlnDecl_exInlnDef();",
        "extern inline void exInlnDecl_exInlnDef() {}",

        "static inline void inInlnDecl_decl();",
        "void inInlnDecl_decl();",
        "static inline void inInlnDecl_inDecl();",
        "static void inInlnDecl_inDecl();",
        "static inline void inInlnDecl_exDecl();",
        "extern void inInlnDecl_exDecl();",
        "static inline void inInlnDecl_inlnDecl();",
        "inline void inInlnDecl_inlnDecl();",
        "static inline void inInlnDecl_inInlnDecl();",
        "static inline void inInlnDecl_inInlnDecl();",
        "static inline void inInlnDecl_exInlnDecl();",
        "extern inline void inInlnDecl_exInlnDecl();",

        "static inline void inInlnDecl_def();",
        "void inInlnDecl_def() {}",
        "static inline void inInlnDecl_inDef();",
        "static void inInlnDecl_inDef() {}",
        "static inline void inInlnDecl_exDef();",
        "extern void inInlnDecl_exDef() {}",
        "static inline void inInlnDecl_inlnDef();",
        "inline void inInlnDecl_inlnDef() {}",
        "static inline void inInlnDecl_inInlnDef();",
        "static inline void inInlnDecl_inInlnDef() {}",
        "static inline void inInlnDecl_exInlnDef();",
        "extern inline void inInlnDecl_exInlnDef() {}",

        "static void inDecl_decl_exDef();",
        "void inDecl_decl_exDef();",
        "extern void inDecl_decl_exDef() {}",
        "static void inDecl_def_exDecl();",
        "void inDecl_def_exDecl() {}",
        "extern void inDecl_def_exDecl();",

        "static int x, inDecl2nd_def();",
        "int inDecl2nd_def() { return 0; }",
      });
  }

  // inlineDefOutExternIn's or externInInlineDefOut's linkage should not be
  // affected by its declaration within fn even though the same declaration at
  // file scope would convert its linkage to external.
  // 
  // Currently, we let the declaration of *OutInlineIn or inlineIn*Out within
  // fn affect whether it has an inline hint. I have seen nothing in the ISO
  // C99 standard that says that's incorrect behavior, and it mostly leaves
  // inlining decisions up to the compiler implementation anyway, but at least
  // we can exercise the case here to be sure it's stable.
  @Test public void linkage_localDecl() throws IOException {
    checkLinkage(
      new ExpectLinkage[]{
        new ExpectLinkage("inlineDeclOutExternIn", LLVMExternalLinkage, false, LLVMInlineHintAttribute),
        new ExpectLinkage("inlineDefOutExternIn", LLVMAvailableExternallyLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("declOutInlineIn", LLVMExternalLinkage, false, LLVMInlineHintAttribute),
        new ExpectLinkage("defOutInlineIn", LLVMExternalLinkage, true, LLVMInlineHintAttribute),

        // These are stripped out because they have internal linkage, no
        // definition, and no uses, so LLVM module verification would complain
        // if we didn't strip them out.
        //new ExpectLinkage("staticDeclOutIn", LLVMInternalLinkage, false),
        //new ExpectLinkage("staticDeclOutInlineIn", LLVMInternalLinkage, false, true),
        //new ExpectLinkage("staticDeclOutExternIn", LLVMInternalLinkage, false),
        new ExpectLinkage("staticDefOutIn", LLVMInternalLinkage, true),
        new ExpectLinkage("staticDefOutInlineIn", LLVMInternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("staticDefOutExternIn", LLVMInternalLinkage, true),
        new ExpectLinkage("staticInlineDefOutIn", LLVMInternalLinkage, true, LLVMInlineHintAttribute),

        new ExpectLinkage("inlineDeclOutShadowExternIn", LLVMExternalLinkage, false, LLVMInlineHintAttribute),
        new ExpectLinkage("inlineDefOutShadowExternIn", LLVMAvailableExternallyLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("declOutShadowInlineIn", LLVMExternalLinkage, false, LLVMInlineHintAttribute),
        new ExpectLinkage("defOutShadowInlineIn", LLVMExternalLinkage, true, LLVMInlineHintAttribute),

        // These are stripped out because they have internal linkage, no
        // definition, and no uses, so LLVM module verification would complain
        // if we didn't strip them out.
        //new ExpectLinkage("staticDeclOutShadowIn", LLVMInternalLinkage, false),
        //new ExpectLinkage("staticDeclOutShadowInlineIn", LLVMInternalLinkage, false, true),
        //new ExpectLinkage("staticDeclOutShadowExternIn", LLVMInternalLinkage, false),
        new ExpectLinkage("staticDefOutShadowIn", LLVMInternalLinkage, true),
        new ExpectLinkage("staticDefOutShadowInlineIn", LLVMInternalLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("staticDefOutShadowExternIn", LLVMInternalLinkage, true),
        new ExpectLinkage("staticInlineDefOutShadowIn", LLVMInternalLinkage, true, LLVMInlineHintAttribute),

        new ExpectLinkage("fn", LLVMExternalLinkage, true),

        new ExpectLinkage("shadowInOnly", LLVMExternalLinkage, false),
        new ExpectLinkage("shadowExternInOnly", LLVMExternalLinkage, false),
        new ExpectLinkage("shadowInlineInOnly", LLVMExternalLinkage, false, LLVMInlineHintAttribute),
        new ExpectLinkage("shadowExternInInlineDeclOut", LLVMExternalLinkage, false, LLVMInlineHintAttribute),
        new ExpectLinkage("shadowExternInInlineDefOut", LLVMAvailableExternallyLinkage, true, true),
        new ExpectLinkage("shadowInlineInDeclOut", LLVMExternalLinkage, false, LLVMInlineHintAttribute),
        new ExpectLinkage("shadowInlineInDefOut", LLVMExternalLinkage, true, LLVMInlineHintAttribute),

        new ExpectLinkage("inOnly", LLVMExternalLinkage, false),
        new ExpectLinkage("externInOnly", LLVMExternalLinkage, false),
        new ExpectLinkage("inlineInOnly", LLVMExternalLinkage, false, LLVMInlineHintAttribute),
        new ExpectLinkage("externInInlineDeclOut", LLVMExternalLinkage, false, LLVMInlineHintAttribute),
        new ExpectLinkage("externInInlineDefOut", LLVMAvailableExternallyLinkage, true, LLVMInlineHintAttribute),
        new ExpectLinkage("inlineInDeclOut", LLVMExternalLinkage, false, LLVMInlineHintAttribute),
        new ExpectLinkage("inlineInDefOut", LLVMExternalLinkage, true, LLVMInlineHintAttribute),
      },
      new String[]{
        "inline void inlineDeclOutExternIn();",
        "inline void inlineDefOutExternIn() {}",
        "void declOutInlineIn();",
        "void defOutInlineIn() {}",

        "static void staticDeclOutIn();",
        "static void staticDeclOutInlineIn();",
        "static void staticDeclOutExternIn();",
        "static void staticDefOutIn() {}",
        "static void staticDefOutInlineIn() {}",
        "static void staticDefOutExternIn() {}",
        "static inline void staticInlineDefOutIn() {}",

        "inline void inlineDeclOutShadowExternIn();",
        "inline void inlineDefOutShadowExternIn() {}",
        "void declOutShadowInlineIn();",
        "void defOutShadowInlineIn() {}",

        "static void staticDeclOutShadowIn();",
        "static void staticDeclOutShadowInlineIn();",
        "static void staticDeclOutShadowExternIn();",
        "static void staticDefOutShadowIn() {}",
        "static void staticDefOutShadowInlineIn() {}",
        "static void staticDefOutShadowExternIn() {}",
        "static inline void staticInlineDefOutShadowIn() {}",

        "void fn() {",
        "  extern void inlineDeclOutExternIn();",
        "  extern void inlineDefOutExternIn();",
        "  inline void declOutInlineIn();",
        "  inline void defOutInlineIn();",

        "  void staticDeclOutIn();",
        "  inline void staticDeclOutInlineIn();",
        "  extern void staticDeclOutExternIn();",

        "  void staticDefOutIn();",
        "  inline void staticDefOutInlineIn();",
        "  extern void staticDefOutExternIn();",
        "  void staticInlineDefOutIn();",

        "  int inlineDeclOutShadowExternIn;",
        "  int inlineDefOutShadowExternIn;",
        "  int declOutShadowInlineIn;",
        "  int defOutShadowInlineIn;",

        "  int staticDeclOutShadowIn;",
        "  int staticDeclOutShadowInlineIn;",
        "  int staticDeclOutShadowExternIn;",
        "  int staticDefOutShadowIn;",
        "  int staticDefOutShadowInlineIn;",
        "  int staticDefOutShadowExternIn;",
        "  int staticInlineDefOutShadowIn;",

        "  int shadowInOnly;",
        "  int shadowExternInOnly;",
        "  int shadowInlineInOnly;",
        "  int shadowExternInInlineDeclOut;",
        "  int shadowExternInInlineDefOut;",
        "  int shadowInlineInDeclOut;",
        "  int shadowInlineInDefOut;",

        "  {",
        "    extern void inlineDeclOutShadowExternIn();",
        "    extern void inlineDefOutShadowExternIn();",
        "    inline void declOutShadowInlineIn();",
        "    inline void defOutShadowInlineIn();",

        "    void staticDeclOutShadowIn();",
        "    inline void staticDeclOutShadowInlineIn();",
        "    extern void staticDeclOutShadowExternIn();",
        "    void staticDefOutShadowIn();",
        "    inline void staticDefOutShadowInlineIn();",
        "    extern void staticDefOutShadowExternIn();",
        "    void staticInlineDefOutShadowIn();",

        "    void shadowInOnly();",
        "    extern void shadowExternInOnly();",
        "    inline void shadowInlineInOnly();",
        "    extern void shadowExternInInlineDeclOut();",
        "    extern void shadowExternInInlineDefOut();",
        "    inline void shadowInlineInDeclOut();",
        "    inline void shadowInlineInDefOut();",
        "  }",

        "  void inOnly();",
        "  extern void externInOnly();",
        "  inline void inlineInOnly();",
        "  extern void externInInlineDeclOut();",
        "  extern void externInInlineDefOut();",
        "  inline void inlineInDeclOut();",
        "  inline void inlineInDefOut();",
        "}",

        "inline void shadowExternInInlineDeclOut();",
        "inline void shadowExternInInlineDefOut() {}",
        "void shadowInlineInDeclOut();",
        "void shadowInlineInDefOut() {}",

        "inline void externInInlineDeclOut();",
        "inline void externInInlineDefOut() {}",
        "void inlineInDeclOut();",
        "void inlineInDefOut() {}",
      });
  }

  @Test public void gnuInlineAttributes() throws IOException {
    checkLinkage(
      new ExpectLinkage[]{
        // GNU semantics
        new ExpectLinkage("eigd", LLVMAvailableExternallyLinkage, true,
                          LLVMInlineHintAttribute),
        new ExpectLinkage("eiad", LLVMExternalLinkage, true,
                          LLVMAlwaysInlineAttribute),
        new ExpectLinkage("eigad", LLVMAvailableExternallyLinkage, true,
                          LLVMAlwaysInlineAttribute),
        new ExpectLinkage("igd", LLVMExternalLinkage, true,
                          LLVMInlineHintAttribute),
        new ExpectLinkage("igad", LLVMExternalLinkage, true,
                          LLVMAlwaysInlineAttribute),

        // C99 semantics are ignored if ever has __gnu_inline__
        new ExpectLinkage("eigd_ei", LLVMAvailableExternallyLinkage, true,
                          LLVMInlineHintAttribute),
        new ExpectLinkage("ei_eigd", LLVMAvailableExternallyLinkage, true,
                          LLVMInlineHintAttribute),
        new ExpectLinkage("igd_i", LLVMExternalLinkage, true,
                          LLVMInlineHintAttribute),
        new ExpectLinkage("i_igd", LLVMExternalLinkage, true,
                          LLVMInlineHintAttribute),

        // agreement between __gnu_inline__ def and C99 decl
        new ExpectLinkage("eigd_i", LLVMAvailableExternallyLinkage, true,
                          LLVMInlineHintAttribute),
        new ExpectLinkage("i_eigd", LLVMAvailableExternallyLinkage, true,
                          LLVMInlineHintAttribute),
        new ExpectLinkage("igd_ei", LLVMExternalLinkage, true,
                          LLVMInlineHintAttribute),
        new ExpectLinkage("ei_igd", LLVMExternalLinkage, true,
                          LLVMInlineHintAttribute),
      },
      new String[]{
        // semantics
        "extern inline __attribute__((__gnu_inline__)) void eigd() {}",
        "extern inline __attribute__((__always_inline__)) void eiad() {}",
        "extern inline",
        "__attribute__((__gnu_inline__)) __attribute__ ((__always_inline__))",
        "void eigad() {}",
        "inline __attribute__((__gnu_inline__)) void igd() {}",
        "inline",
        "__attribute__((__gnu_inline__)) __attribute__ ((__always_inline__))",
        "void igad() {}",

        // C99 semantics are ignored if ever has __gnu_inline__
        "extern inline __attribute__((__gnu_inline__)) void eigd_ei() {}",
        "extern inline void eigd_ei();",
        "extern inline void ei_eigd();",
        "extern inline __attribute__((__gnu_inline__)) void ei_eigd() {}",
        "inline __attribute__((__gnu_inline__)) void igd_i() {}",
        "inline void igd_i();",
        "inline void i_igd();",
        "inline __attribute__((__gnu_inline__)) void i_igd() {}",

        // agreement between __gnu_inline__ def and C99 decl
        "extern inline __attribute__((__gnu_inline__)) void eigd_i() {}",
        "inline void eigd_i();",
        "inline void i_eigd();",
        "extern inline __attribute__((__gnu_inline__)) void i_eigd() {}",
        "inline __attribute__((__gnu_inline__)) void igd_ei() {}",
        "extern inline void igd_ei();",
        "extern inline void ei_igd();",
        "inline __attribute__((__gnu_inline__)) void ei_igd() {}",
      });
  }
}