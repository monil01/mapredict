package openacc.codegen.llvmBackend;

import static org.jllvm.bindings.LLVMLinkage.LLVMExternalLinkage;
import static org.jllvm.bindings.LLVMLinkage.LLVMInternalLinkage;
import static org.jllvm.bindings.LLVMOpcode.LLVMStore;
import static org.jllvm.bindings.LLVMTypeKind.LLVMIntegerTypeKind;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.hamcrest.CoreMatchers;
import org.jllvm.LLVMBasicBlock;
import org.jllvm.LLVMConstant;
import org.jllvm.LLVMConstantInteger;
import org.jllvm.LLVMFunction;
import org.jllvm.LLVMGlobalVariable;
import org.jllvm.LLVMInstruction;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMReturnInstruction;
import org.jllvm.LLVMValue;
import org.jllvm.bindings.LLVMLinkage;
import org.junit.Test;

/**
 * Checks the ability to generate a single correct LLVM global variable
 * declaration/definition for a static local variable or for a file-scope
 * variable that's declared/defined one or more times in the source code.
 * 
 * <p>
 * Linkage and initialization of the resulting LLVM global variable are
 * checked. In the source, a file-scope variable is permitted to have multiple
 * declarations/definitions: any number of tentative definitions, at most one
 * external definition, any number of extern declarations. Linkage can be
 * specified only on the first declaration/definition, and others can specify
 * extern to refer to the previous definitions.
 * </p>
 * 
 * <p>
 * Various types are not checked here. They are checked in
 * {@link BuildLLVMTest_Types}.
 * </p>
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public class BuildLLVMTest_GlobalVariableDeclarations extends BuildLLVMTest {
  private static class Expect {
    public Expect(String name, Long init) {
      this(name, init, null, false, (Expect[])null);
    }
    public Expect(String name, Long init, LLVMLinkage linkage) {
      this(name, init, linkage, false, (Expect[])null);
    }
    public Expect(String name, Long init, LLVMLinkage linkage,
                  boolean threadLocal)
    {
      this(name, init, linkage, threadLocal, (Expect[])null);
    }
    public Expect(String name, Long init, LLVMLinkage linkage,
                  Expect... localVars)
    {
      this(name, init, linkage, false, localVars);
    }
    public Expect(String name, Long init, LLVMLinkage linkage,
                  boolean threadLocal, Expect... localVars)
    {
      this.name = name;
      assert(localVars == null || init == null);
      this.init = init;
      this.linkage = linkage;
      this.threadLocal = threadLocal;
      this.localVars = localVars;
      if (localVars != null) {
        for (Expect localVar : localVars) {
          assert(localVar.linkage == null);
          assert(!localVar.threadLocal);
        }
      }
    }
    public final String name;
    public final Long init;
    public final LLVMLinkage linkage;
    public final boolean threadLocal;
    public final Expect[] localVars;
  }
  private void checkVar(Expect expect, LLVMModule llvmModule) {
    LLVMGlobalVariable var = llvmModule.getNamedGlobal(expect.name);
    assertNotNull(expect.name + " must exist as a global variable",
                  var.getInstance());
    assertEquals(expect.name + " initializer's existence",
                 expect.init != null, !var.isDeclaration());
    if (expect.init != null) {
      LLVMConstant actualInit = var.getInitializer();
      assertEquals(expect.name + " initializer's type",
                   LLVMIntegerTypeKind, actualInit.typeOf().getTypeKind());
      assertEquals(expect.name + " initializer's value",
                   expect.init.longValue(),
                   ((LLVMConstantInteger)actualInit).getSExtValue());
    }
    assertEquals(expect.name + "'s linkage",
                 expect.linkage, var.getLinkage());
    assertEquals(expect.name + "'s thread_local property",
                 expect.threadLocal, var.isThreadLocal());
  }
  private void checkFn(Expect expect, LLVMModule llvmModule) {
    LLVMFunction fn = llvmModule.getNamedFunction(expect.name);
    assertNotNull(expect.name + " must exist as a function",
                  fn.getInstance());
    assertEquals(expect.name + "'s linkage", expect.linkage, fn.getLinkage());
    LLVMBasicBlock entry = fn.getEntryBasicBlock();
    Map<String, LLVMInstruction> valueTable = new HashMap<>();
    Map<LLVMValue, LLVMValue> initTable = new HashMap<>();
    for (LLVMInstruction insn = entry.getFirstInstruction();
         insn.getInstance() != null; insn = insn.getNextInstruction())
    {
      if (insn.getInstructionOpcode() == LLVMStore) {
        if (initTable.get(insn.getOperand(1)) == null)
          initTable.put(insn.getOperand(1), insn.getOperand(0));
      }
      else if (!(insn instanceof LLVMReturnInstruction)){
        valueTable.put(insn.getValueName(), insn);
      }
    }
    for (Expect localVar : expect.localVars) {
      LLVMInstruction insn = valueTable.get(localVar.name);
      assertNotNull(fn.getValueName() + " must have a local " + localVar.name,
                    insn);
      assertEquals(fn.getValueName() + "." + localVar.name
                    + "'s initializer's existence",
                    localVar.init != null, initTable.get(insn) != null);
      if (localVar.init != null) {
        LLVMValue init = initTable.get(insn);
        assertEquals(fn.getValueName() + "." + localVar.name
                     + "'s initializer",
                     localVar.init.longValue(),
                     ((LLVMConstantInteger)init).getSExtValue());
      }
    }
    assertEquals(fn.getValueName() + "'s local variable declaration count",
                 expect.localVars.length, valueTable.size());
  }
  private void checkDecls(Expect[] expects, String[] decls)
    throws IOException
  {
    String[] src = new String[decls.length + 1];
    System.arraycopy(decls, 0, src, 0, decls.length);
    src[decls.length] = "int main() { return 0; }";
    final SimpleResult simpleResult = buildLLVMSimple("", "", src);
    int globalVarCount = 0;
    for (Expect expect : expects) {
      if (expect.localVars == null) {
        checkVar(expect, simpleResult.llvmModule);
        ++globalVarCount;
      }
      else
        checkFn(expect, simpleResult.llvmModule);
    }
    // If we accidentally generated a variable multiple times, LLVM would
    // assign new names to later occurrences, so make sure we have exactly the
    // number of variables we expect.
    int nGlobals = 0;
    for (LLVMGlobalVariable global = simpleResult.llvmModule.getFirstGlobal();
         global.getInstance() != null;
         global = global.getNextGlobal())
      ++nGlobals;
    assertEquals("global variable declaration count", globalVarCount,
                 nGlobals);
  }

  /**
   * This check used to expose a race condition that sometimes caused LLVM
   * assertion failures ("Use still stuck around after Def is destroyed") in
   * (much) later test cases because it generated a load instruction at file
   * scope. We have since fixed BuildLLVM to throw the exception reporting the
   * non-constant initializer before the point where it would generate the load
   * instruction.
   */
  @Test public void loadAtFileScope() throws Exception {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "initializer is not a constant"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "int *p1;",
        "int *p2 = p1;",
      });
  }

  // Extern declaration only.
  @Test public void exDeclOnly() throws IOException {
    checkDecls(
      new Expect[]{
        new Expect("ex",           null, LLVMExternalLinkage),
        new Expect("ex_ex",        null, LLVMExternalLinkage),
        new Expect("exIncomplete", null, LLVMExternalLinkage),
      },
      new String[]{
        "extern int ex;",
        "extern int ex_ex;",
        "extern int ex_ex;",
        "extern struct T exIncomplete;",
      });
  }

  // Tentative definition only.
  @Test public void tnDefOnly() throws IOException {
    checkDecls(
      new Expect[]{
        new Expect("ex",    new Long(0), LLVMExternalLinkage),
        new Expect("in",    new Long(0), LLVMInternalLinkage),
        new Expect("ex_ex", new Long(0), LLVMExternalLinkage),
        new Expect("in_in", new Long(0), LLVMInternalLinkage),
      },
      new String[]{
        "int ex;",
        "static int in;",
        "int ex_ex;",
        "int ex_ex;",
        "static int in_in;",
        "static int in_in;",
      });
  }

  // External definition only.
  @Test public void exDefOnly() throws IOException {
    checkDecls(
      new Expect[]{
        new Expect("ex0",   new Long(0), LLVMExternalLinkage),
        new Expect("ex2",   new Long(2), LLVMExternalLinkage),
        new Expect("in0",   new Long(0), LLVMInternalLinkage),
        new Expect("in3",   new Long(3), LLVMInternalLinkage),
      },
      new String[]{
        "int ex0 = 0;",
        "int ex2 = 2;",
        "static int in0 = 0;",
        "static int in3 = 3;",
      });
  }

  // Extern declaration first.
  @Test public void exDeclFirst() throws IOException {
    checkDecls(
      new Expect[]{
        new Expect("exDecl_tnDef",        new Long(0), LLVMExternalLinkage),
        new Expect("exDecl_tnDef2nd",     new Long(0), LLVMExternalLinkage),
        new Expect("x",                   new Long(0), LLVMExternalLinkage),
        new Expect("exDecl_exDef0",       new Long(0), LLVMExternalLinkage),
        new Expect("exDecl_exDef2nd",     new Long(1), LLVMExternalLinkage),
        new Expect("y",                   new Long(0), LLVMExternalLinkage),
        new Expect("exDecl_exDef5",       new Long(5), LLVMExternalLinkage),
        new Expect("exDecl_tnDef_exDef3", new Long(3), LLVMExternalLinkage),
        new Expect("exDecl_exDef2_tnDef", new Long(2), LLVMExternalLinkage),
      },
      new String[]{
        "extern int exDecl_tnDef;",
        "int exDecl_tnDef;",
        "extern int exDecl_tnDef2nd;",
        "int x, exDecl_tnDef2nd;",
        "extern int exDecl_exDef0;",
        "int exDecl_exDef0 = 0;",
        "extern int exDecl_exDef2nd;",
        "int y, exDecl_exDef2nd = 1;",
        "extern int exDecl_exDef5;",
        "int exDecl_exDef5 = 5;",
        "extern int exDecl_tnDef_exDef3;",
        "int exDecl_tnDef_exDef3;",
        "int exDecl_tnDef_exDef3 = 3;",
        "extern int exDecl_exDef2_tnDef;",
        "int exDecl_exDef2_tnDef = 2;",
        "int exDecl_exDef2_tnDef;",
      });
  }

  // Tentative definition first.
  @Test public void tnDefFirst() throws IOException {
    checkDecls(
      new Expect[]{
        new Expect("ex_tnDef_exDecl",       new Long(0), LLVMExternalLinkage),
        new Expect("in_tnDef_exDecl",       new Long(0), LLVMInternalLinkage),
        new Expect("ex_tnDef_exDef",        new Long(1), LLVMExternalLinkage),
        new Expect("in_tnDef_exDef",        new Long(3), LLVMInternalLinkage),
        new Expect("ex_tnDef_exDef2nd",     new Long(7), LLVMExternalLinkage),
        new Expect("x",                     new Long(0), LLVMExternalLinkage),
        new Expect("in_tnDef_exDef2nd",     new Long(9), LLVMInternalLinkage),
        new Expect("y",                     new Long(0), LLVMInternalLinkage),
        new Expect("in_tnDef2nd_exDecl", new Long(0), LLVMInternalLinkage),
        new Expect("z",                     new Long(0), LLVMInternalLinkage),
        new Expect("ex_tnDef_exDecl_exDef", new Long(1), LLVMExternalLinkage),
        new Expect("ex_tnDef_exDecl_exDecl", new Long(0), LLVMExternalLinkage),
        new Expect("in_tnDef_exDecl_exDef", new Long(9), LLVMInternalLinkage),
        new Expect("in_tnDef_exDecl_exDefNoEx", new Long(6), LLVMInternalLinkage),
        new Expect("ex_tnDef_exDef_exDecl", new Long(2), LLVMExternalLinkage),
        new Expect("in_tnDef_exDef_exDecl", new Long(4), LLVMInternalLinkage),
      },
      new String[]{
        "int ex_tnDef_exDecl;",
        "extern int ex_tnDef_exDecl;",
        "static int in_tnDef_exDecl;",
        "extern int in_tnDef_exDecl;",
        "int ex_tnDef_exDef;",
        "int ex_tnDef_exDef = 1;",
        "int ex_tnDef_exDef2nd;",
        "int x, ex_tnDef_exDef2nd = 7;",
        "static int in_tnDef_exDef;",
        "static int in_tnDef_exDef = 3;",
        "static int in_tnDef_exDef2nd;",
        "static int y, in_tnDef_exDef2nd = 9;",
        "static int z, in_tnDef2nd_exDecl;",
        "extern int in_tnDef2nd_exDecl;",
        "int ex_tnDef_exDecl_exDef;",
        "extern int ex_tnDef_exDecl_exDef;",
        "int ex_tnDef_exDecl_exDef = 1;",
        "int ex_tnDef_exDecl_exDecl;",
        "extern int ex_tnDef_exDecl_exDecl;",
        "extern int ex_tnDef_exDecl_exDecl;",
        "static int in_tnDef_exDecl_exDef;",
        "extern int in_tnDef_exDecl_exDef;",
        "static int in_tnDef_exDecl_exDef = 9;",
        "static int in_tnDef_exDecl_exDefNoEx;",
        "extern int in_tnDef_exDecl_exDefNoEx;",
        "int in_tnDef_exDecl_exDefNoEx= 6;",
        "int ex_tnDef_exDef_exDecl;",
        "int ex_tnDef_exDef_exDecl = 2;",
        "extern int ex_tnDef_exDef_exDecl;",
        "static int in_tnDef_exDef_exDecl;",
        "static int in_tnDef_exDef_exDecl = 4;",
        "extern int in_tnDef_exDef_exDecl;",
      });
  }

  // External definition first.
  @Test public void exDefFirst() throws IOException {
    checkDecls(
      new Expect[]{
        new Expect("ex_exDef_exDecl",      new Long(2), LLVMExternalLinkage),
        new Expect("in_exDef_exDecl",      new Long(7), LLVMInternalLinkage),
        new Expect("ex_exDef_tnDef",       new Long(8), LLVMExternalLinkage),
        new Expect("in_exDef_tnDef",       new Long(3), LLVMInternalLinkage),
      },
      new String[]{
        "int ex_exDef_exDecl = 2;",
        "extern int ex_exDef_exDecl;",
        "static int in_exDef_exDecl = 7;",
        "extern int in_exDef_exDecl;",
        "int ex_exDef_tnDef = 8;",
        "int ex_exDef_tnDef;",
        "static int in_exDef_tnDef = 3;",
        "static int in_exDef_tnDef;",
      });
  }

  // Block-scope and file-scope declarations with same names.
  @Test public void localDecl() throws IOException {
    checkDecls(
      new Expect[]{
        // Make sure file-scope declarations are not affected by block-scope
        // definitions of the same name, and make sure static block-scope
        // definitions are initialized correctly (zero if unspecified).
        new Expect("exDecl_inInitIn", null, LLVMExternalLinkage),
        new Expect("tnDef_inInitIn", new Long(0), LLVMExternalLinkage),
        new Expect("exDef_inInitIn", new Long(10), LLVMExternalLinkage),
        new Expect("exDecl_inIn", null, LLVMExternalLinkage),
        new Expect("tnDef_inIn", new Long(0), LLVMExternalLinkage),
        new Expect("exDef_inIn", new Long(9), LLVMExternalLinkage),
        new Expect("inTnDef_initIn", new Long(0), LLVMInternalLinkage),
        new Expect("inExDef_initIn", new Long(14), LLVMInternalLinkage),
        new Expect("inTnDef_inInitIn", new Long(0), LLVMInternalLinkage),
        new Expect("inExDef_inInitIn", new Long(23), LLVMInternalLinkage),

        // Exercise block-scope extern declarations.

        new Expect("exDeclOut_exDeclIn", null, LLVMExternalLinkage),
        new Expect("tnDef_exDeclIn", new Long(0), LLVMExternalLinkage),
        new Expect("exDef_exDeclIn", new Long(9), LLVMExternalLinkage),
        new Expect("inTnDef_exDeclIn", new Long(0), LLVMInternalLinkage),
        new Expect("inExDef_exDeclIn", new Long(8), LLVMInternalLinkage),

        new Expect("exDeclOut_shExDeclIn", null, LLVMExternalLinkage),
        new Expect("tnDef_shExDeclIn", new Long(0), LLVMExternalLinkage),
        new Expect("exDef_shExDeclIn", new Long(11), LLVMExternalLinkage),
        new Expect("inTnDef_shExDeclIn", new Long(0), LLVMInternalLinkage),
        new Expect("inExDef_shExDeclIn", new Long(1), LLVMInternalLinkage),

        new Expect("fn", null, LLVMExternalLinkage,
                   new Expect("inTnDef_initIn", new Long(6)),
                   new Expect("inExDef_initIn", new Long(2)),

                   new Expect("exDeclOut_shExDeclIn", null),
                   new Expect("tnDef_shExDeclIn", null),
                   new Expect("exDef_shExDeclIn", null),
                   new Expect("inTnDef_shExDeclIn", null),
                   new Expect("inExDef_shExDeclIn", null),
                   new Expect("shExDeclIn_exDeclOut", null),
                   new Expect("shExDeclIn_tnDef", null),
                   new Expect("shExDeclIn_exDef", null)),
        new Expect("fn.exDecl_inInitIn", new Long(4), LLVMInternalLinkage),
        new Expect("fn.tnDef_inInitIn", new Long(5), LLVMInternalLinkage),
        new Expect("fn.exDef_inInitIn", new Long(20), LLVMInternalLinkage),
        new Expect("fn.exDecl_inIn", new Long(0), LLVMInternalLinkage),
        new Expect("fn.tnDef_inIn", new Long(0), LLVMInternalLinkage),
        new Expect("fn.exDef_inIn", new Long(0), LLVMInternalLinkage),
        new Expect("fn.inTnDef_inInitIn", new Long(6), LLVMInternalLinkage),
        new Expect("fn.inExDef_inInitIn", new Long(2), LLVMInternalLinkage),

        new Expect("shExDeclIn_exDeclOut", null, LLVMExternalLinkage),
        new Expect("shExDeclIn_tnDef", new Long(0), LLVMExternalLinkage),
        new Expect("shExDeclIn_exDef", new Long(45), LLVMExternalLinkage),

        new Expect("exDeclInOnly", null, LLVMExternalLinkage),
        new Expect("exDeclInOnlyIncomplete", null, LLVMExternalLinkage),

        new Expect("exDeclIn_exDeclOut", null, LLVMExternalLinkage),
        new Expect("exDeclIn_tnDef", new Long(0), LLVMExternalLinkage),
        new Expect("exDeclIn_exDef", new Long(32), LLVMExternalLinkage),
      },
      new String[]{
        "extern int exDecl_inInitIn;",
        "int tnDef_inInitIn;",
        "int exDef_inInitIn = 10;",
        "extern int exDecl_inIn;",
        "int tnDef_inIn;",
        "int exDef_inIn = 9;",
        "static int inTnDef_initIn;",
        "static int inExDef_initIn = 14;",
        "static int inTnDef_inInitIn;",
        "static int inExDef_inInitIn = 23;",

        "extern int exDeclOut_exDeclIn;",
        "int tnDef_exDeclIn;",
        "int exDef_exDeclIn = 9;",
        "static int inTnDef_exDeclIn;",
        "static int inExDef_exDeclIn = 8;",

        "extern int exDeclOut_shExDeclIn;",
        "int tnDef_shExDeclIn;",
        "int exDef_shExDeclIn = 11;",
        "static int inTnDef_shExDeclIn;",
        "static int inExDef_shExDeclIn = 1;",

        "void fn() {",
        "  static int exDecl_inInitIn = 4;",
        "  static int tnDef_inInitIn = 5;",
        "  static int exDef_inInitIn = 20;",
        "  static int exDecl_inIn;",
        "  static int tnDef_inIn;",
        "  static int exDef_inIn;",
        "  int inTnDef_initIn = 6;",
        "  int inExDef_initIn = 2;",
        "  static int inTnDef_inInitIn = 6;",
        "  static int inExDef_inInitIn = 2;",

        "  extern int exDeclOut_exDeclIn;",
        "  extern int tnDef_exDeclIn;",
        "  extern int exDef_exDeclIn;",
        "  extern int inTnDef_exDeclIn;",
        "  extern int inExDef_exDeclIn;",

        "  int exDeclOut_shExDeclIn;",
        "  int tnDef_shExDeclIn;",
        "  int exDef_shExDeclIn;",
        "  int inTnDef_shExDeclIn;",
        "  int inExDef_shExDeclIn;",
        "  int shExDeclIn_exDeclOut;",
        "  int shExDeclIn_tnDef;",
        "  int shExDeclIn_exDef;",
        "  {",
        "    extern int exDeclOut_shExDeclIn;",
        "    extern int tnDef_shExDeclIn;",
        "    extern int exDef_shExDeclIn;",
        "    extern int inTnDef_shExDeclIn;",
        "    extern int inExDef_shExDeclIn;",

        "    extern int shExDeclIn_exDeclOut;",
        "    extern int shExDeclIn_tnDef;",
        "    extern int shExDeclIn_exDef;",
        "  }",

        "  extern int exDeclInOnly;",
        "  extern int exDeclInOnlyIncomplete;",

        "  extern int exDeclIn_exDeclOut;",
        "  extern int exDeclIn_tnDef;",
        "  extern int exDeclIn_exDef;",
        "}",

        "extern int shExDeclIn_exDeclOut;",
        "int shExDeclIn_tnDef;",
        "int shExDeclIn_exDef = 45;",

        "extern int exDeclIn_exDeclOut;",
        "int exDeclIn_tnDef;",
        "int exDeclIn_exDef = 32;",
      });
  }

  /**
   * TODO: Support for __thread was implemented in a rush. Needs more
   * thorough testing.
   */
  @Test public void threadLocal() throws IOException {
    checkDecls(
      new Expect[]{
        new Expect("ex", new Long(0), LLVMExternalLinkage, true),
        new Expect("in", new Long(0), LLVMInternalLinkage, true),
        new Expect("exLocal", null, LLVMExternalLinkage, true),
        new Expect("fn.inLocal", new Long(0), LLVMInternalLinkage, true),
      },
      new String[]{
        "__thread int ex;",
        "static __thread int in;",
        "void fn() {",
        "  extern __thread int exLocal;",
        "  static __thread int inLocal;",
        "}",
      });
  }
}