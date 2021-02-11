package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcPrimitiveFloatingType.SrcDoubleType;
import static openacc.codegen.llvmBackend.SrcPrimitiveFloatingType.SrcFloatType;
import static openacc.codegen.llvmBackend.SrcPrimitiveFloatingType.SrcLongDoubleType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SRC_SIZE_T_TYPE;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcBoolType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcCharType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcIntType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcLongLongType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcLongType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcShortType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcSignedCharType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedCharType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedIntType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedLongLongType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedLongType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedShortType;
import static openacc.codegen.llvmBackend.SrcPrimitiveNonNumericType.SrcVoidType;
import static org.jllvm.bindings.LLVMOpcode.LLVMStore;
import static org.jllvm.bindings.LLVMTypeKind.LLVMFunctionTypeKind;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.hamcrest.CoreMatchers;
import org.jllvm.LLVMArrayType;
import org.jllvm.LLVMBasicBlock;
import org.jllvm.LLVMConstant;
import org.jllvm.LLVMConstantArray;
import org.jllvm.LLVMConstantInteger;
import org.jllvm.LLVMContext;
import org.jllvm.LLVMExecutionEngine;
import org.jllvm.LLVMFunction;
import org.jllvm.LLVMFunctionType;
import org.jllvm.LLVMGenericValue;
import org.jllvm.LLVMGlobalValue;
import org.jllvm.LLVMGlobalVariable;
import org.jllvm.LLVMIdentifiedStructType;
import org.jllvm.LLVMInstruction;
import org.jllvm.LLVMIntegerType;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMPointerType;
import org.jllvm.LLVMRealType;
import org.jllvm.LLVMTargetData;
import org.jllvm.LLVMType;
import org.jllvm.LLVMValue;
import org.jllvm.LLVMVoidType;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Checks the ability to build correct LLVM types.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class BuildLLVMTest_Types extends BuildLLVMTest {
  private static class ExpectStruct {
    /***
     * @param tag
     *          is the struct tag
     * @param memberTypes
     *          is the array of member types, or it is null to indicate an
     *          opaque struct
     */
    public ExpectStruct(String tag, LLVMType... memberTypes) {
      this.tag = tag;
      this.memberTypes = memberTypes;
    }
    public final String tag;
    public final LLVMType[] memberTypes;
  }
  private static class ExpectTaglessStruct {
    /***
     * Same as {@link ExpectedStruct} except for a global variable whose type
     * is a tag-less struct with the specified member types.
     */
    public ExpectTaglessStruct(String varName, LLVMType... memberTypes) {
      this.varName = varName;
      this.memberTypes = memberTypes;
    }
    public final String varName;
    public final LLVMType[] memberTypes;
  }
  private static class Expect {
    public Expect(String name, LLVMType type) {
      this(name, type, (LLVMConstant)null);
    }
    public Expect(String name, LLVMType type, Integer init) {
      this(name, type,
           init == null
           ? null
           : LLVMConstantInteger.get(SrcIntType.getLLVMType(type.getContext()),
                                     init.longValue(), init.longValue() < 0));
    }
    public Expect(String name, LLVMType type, LLVMConstant init) {
      this.name = name;
      this.type = type;
      this.namedType = null;
      this.init = init;
      this.localVars = null;
    }
    public Expect(String name, LLVMFunctionType fnType, Expect... localVars) {
      this.name = name;
      this.type = fnType;
      this.namedType = null;
      this.init = null;
      this.localVars = localVars;
    }
    public Expect(String name, String namedType) {
      this.name = name;
      this.type = null;
      this.namedType = namedType;
      this.init = null;
      this.localVars = null;
    }
    public void checkType(LLVMType type, LLVMModule llvmModule) {
      checkType(null, type, llvmModule);
    }
    public void checkType(String enclosingFnName, LLVMType actualType,
                          LLVMModule llvmModule)
    {
      final String descr
        = (enclosingFnName == null ? "" : enclosingFnName + ".") + name;
      LLVMType expectedType = type;
      if (expectedType == null) {
        expectedType = llvmModule.getTypeByName(namedType);
        assertNotNull(descr + "'s expected type, " + namedType
                      + ", must exist",
                      expectedType.getInstance());
      }
      assertEquals(descr + "'s type",
                   LLVMPointerType.get(expectedType, 0), actualType);
    }
    public void checkInit(LLVMValue actualInit) {
      checkInit(null, actualInit);
    }
    public void checkInit(String enclosingFnName, LLVMValue actualInit) {
      if (init == null)
        return;
      final String descr
        = (enclosingFnName == null ? "" : enclosingFnName + ".") + name;
      assertNotNull(descr + " must be initialized", actualInit);
      assertEquals(descr + "'s initializer", init, actualInit);
    }
    public final String name;
    public final LLVMType type;
    public final String namedType;
    public final LLVMConstant init;
    public final Expect[] localVars;
  }

  private SimpleResult checkDecls(String[] decls) throws IOException {
    return checkDecls("", decls);
  }
  private SimpleResult checkDecls(String llvmTargetDataLayout, String[] decls)
    throws IOException
  {
    return buildLLVMSimple("", llvmTargetDataLayout, decls);
  }
  private void checkDecls(LLVMModule llvmModule, ExpectStruct[] expectStructs)
    throws IOException
  {
    for (ExpectStruct expect : expectStructs) {
      LLVMIdentifiedStructType type = llvmModule.getTypeByName(expect.tag);
      assertNotNull(expect.tag + " must exist", type.getInstance());
      checkStruct(expect.tag, type, expect.memberTypes);
    }
  }
  private void checkDecls(LLVMModule llvmModule,
                          ExpectTaglessStruct... expects) throws IOException
  {
    for (ExpectTaglessStruct expect : expects) {
      LLVMGlobalVariable var = llvmModule.getNamedGlobal(expect.varName);
      assertNotNull(expect.varName + " must exist", var.getInstance());
      assertTrue(expect.varName + "s type",
                 var.typeOf() instanceof LLVMPointerType);
      LLVMPointerType ptrType = (LLVMPointerType)var.typeOf();
      assertTrue(expect.varName + "'s type must be a pointer to a struct",
                 ptrType.getElementType() instanceof LLVMIdentifiedStructType);
      LLVMIdentifiedStructType type
        = (LLVMIdentifiedStructType)ptrType.getElementType();
      checkStruct(expect.varName + "'s type", type, expect.memberTypes);
    }
  }
  private void checkStruct(String descr, LLVMIdentifiedStructType type,
                           LLVMType... expectedMemberTypes)
  {
      assertFalse(descr + " must not be packed", type.isPacked());
      assertEquals(descr + "'s opaqueness",
                   expectedMemberTypes == null, type.isOpaque());
      LLVMType[] memberTypes = type.getElementTypes();
      assertEquals(descr + "'s number of members",
                   expectedMemberTypes == null ? 0 : expectedMemberTypes.length,
                   memberTypes.length);
      for (int i = 0; i < memberTypes.length; ++i)
        assertEquals(descr + "'s member " + i + " type",
                     expectedMemberTypes[i], memberTypes[i]);
    
  }
  private void checkDecls(LLVMModule llvmModule, Expect[] expects) {
    for (Expect expect : expects) {
      LLVMGlobalValue global;
      if (expect.type != null
          && expect.type.getTypeKind() == LLVMFunctionTypeKind)
      {
        global = llvmModule.getNamedFunction(expect.name);
        assertNotNull(expect.name + " must exist as a function",
                      global.getInstance());
      }
      else {
        global = llvmModule.getNamedGlobal(expect.name);
        assertNotNull(expect.name + " must exist as a global variable",
                      global.getInstance());
      }
      expect.checkType(global.typeOf(), llvmModule);
      if (global instanceof LLVMGlobalVariable)
        expect.checkInit(((LLVMGlobalVariable)global).getInitializer());
      if (expect.localVars != null) {
        LLVMBasicBlock entry = ((LLVMFunction)global).getEntryBasicBlock();
        Map<String, LLVMInstruction> valueTable = new HashMap<>();
        Map<LLVMValue, LLVMValue> initTable = new HashMap<>();
        for (LLVMInstruction insn = entry.getFirstInstruction();
             insn.getInstance() != null; insn = insn.getNextInstruction())
        {
          valueTable.put(insn.getValueName(), insn);
          if (insn.getInstructionOpcode() == LLVMStore) {
            if (initTable.get(insn.getOperand(1)) == null)
              initTable.put(insn.getOperand(1), insn.getOperand(0));
          }
        }
        for (Expect localVar : expect.localVars) {
          LLVMInstruction insn = valueTable.get(localVar.name);
          assertNotNull(global.getValueName() + " must have a local "
                        + localVar.name,
                        insn);
          localVar.checkType(global.getValueName(), insn.typeOf(),
                             llvmModule);
          localVar.checkInit(global.getValueName(), initTable.get(insn));
        }
      }
    }
    // If we accidentally generated a global multiple times, LLVM would
    // assign new names to later occurrences, so make sure we have exactly the
    // number of globals we expect.
    int nGlobals = 0;
    for (LLVMGlobalVariable global = llvmModule.getFirstGlobal();
         global.getInstance() != null;
         global = global.getNextGlobal())
      ++nGlobals;
    for (LLVMFunction global = llvmModule.getFirstFunction();
         global.getInstance() != null;
         global = global.getNextFunction())
      ++nGlobals;
    assertEquals("global declaration count", expects.length, nGlobals);
  }

  private LLVMFunctionType getFnTy(LLVMType retType) {
    return getFnTy(retType, true);
  }
  private LLVMFunctionType getFnTy(LLVMContext ctxt, LLVMType... paramTypes) {
    return getFnTy(LLVMVoidType.get(ctxt), false, paramTypes);
  }
  private LLVMFunctionType getFnTy(LLVMContext ctxt, boolean isVarArg,
                                   LLVMType... paramTypes)
  {
    return getFnTy(LLVMVoidType.get(ctxt), isVarArg, paramTypes);
  }
  private LLVMFunctionType getFnTy(LLVMType retType, boolean isVarArg,
                                   LLVMType... paramTypes)
  {
    return LLVMFunctionType.get(retType, isVarArg, paramTypes);
  }

  @BeforeClass public static void setup() {
    // Normally the BuildLLVM pass loads the jllvm native library. However,
    // these test methods use jllvm before running the BuildLLVM pass. In case
    // these test methods are run before any other test methods that run the
    // BuildLLVM pass, we must load the jllvm native library first.
    System.loadLibrary("jllvm");
  }

  @Test public void primitive() throws IOException {
    final SimpleResult simpleResult = checkDecls(
      new String[]{
        "char c0;",
        "signed char c1;",
        "char unsigned c2;",

        "short s0;",
        "short signed s1;",
        "short int s2;",
        "signed int short s3;",
        "unsigned short s4;",
        "int short unsigned s5;",

        "int i0;",
        "signed i1;",
        "signed int i2;",
        "unsigned i3;",
        "int unsigned i4;",

        "long l0;",
        "long signed l1;",
        "long int l2;",
        "int signed long l3;",
        "unsigned long l4;",
        "unsigned int long l5;",

        "long long ll0;",
        "long signed long ll1;",
        "int long long ll2;",
        "long long signed int ll3;",
        "unsigned long long ll4;",
        "long unsigned long int ll5;",

        "float f;",
        "double d;",
        "long double ld;",
        "_Bool b1;",
        "#include <stdbool.h>",
        "bool b2;",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    checkDecls(mod,
      new Expect[]{
        new Expect("c0", SrcCharType.getLLVMType(ctxt)),
        new Expect("c1", SrcSignedCharType.getLLVMType(ctxt)),
        new Expect("c2", SrcUnsignedCharType.getLLVMType(ctxt)),

        new Expect("s0", SrcShortType.getLLVMType(ctxt)),
        new Expect("s1", SrcShortType.getLLVMType(ctxt)),
        new Expect("s2", SrcShortType.getLLVMType(ctxt)),
        new Expect("s3", SrcShortType.getLLVMType(ctxt)),
        new Expect("s4", SrcUnsignedShortType.getLLVMType(ctxt)),
        new Expect("s5", SrcUnsignedShortType.getLLVMType(ctxt)),

        new Expect("i0", SrcIntType.getLLVMType(ctxt)),
        new Expect("i1", SrcIntType.getLLVMType(ctxt)),
        new Expect("i2", SrcIntType.getLLVMType(ctxt)),
        new Expect("i3", SrcUnsignedIntType.getLLVMType(ctxt)),
        new Expect("i4", SrcUnsignedIntType.getLLVMType(ctxt)),

        new Expect("l0", SrcLongType.getLLVMType(ctxt)),
        new Expect("l1", SrcLongType.getLLVMType(ctxt)),
        new Expect("l2", SrcLongType.getLLVMType(ctxt)),
        new Expect("l3", SrcLongType.getLLVMType(ctxt)),
        new Expect("l4", SrcUnsignedLongType.getLLVMType(ctxt)),
        new Expect("l5", SrcUnsignedLongType.getLLVMType(ctxt)),

        new Expect("ll0", SrcLongLongType.getLLVMType(ctxt)),
        new Expect("ll1", SrcLongLongType.getLLVMType(ctxt)),
        new Expect("ll2", SrcLongLongType.getLLVMType(ctxt)),
        new Expect("ll3", SrcLongLongType.getLLVMType(ctxt)),
        new Expect("ll4", SrcUnsignedLongLongType.getLLVMType(ctxt)),
        new Expect("ll5", SrcUnsignedLongLongType.getLLVMType(ctxt)),

        new Expect("f", SrcFloatType.getLLVMType(ctxt)),
        new Expect("d", SrcDoubleType.getLLVMType(ctxt)),
        new Expect("ld", SrcLongDoubleType.getLLVMType(ctxt)),
        new Expect("b1", SrcBoolType.getLLVMType(ctxt)),
        new Expect("b2", SrcBoolType.getLLVMType(ctxt)),
      });
  }

  @Test public void pointer() throws IOException {
    final SimpleResult simpleResult = checkDecls(
      new String[]{
        "void *pv;",
        "int *pi;",
        "float *pf;",
        "int **ppi;",
        "float ***pppi;",
        "float (*paf)[10];",
        "void (*pFn0)();",
        "void (*pFn1)(int i);",
        "void (*pFn2)(void(), void(*)(int), int[]);",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    checkDecls(mod,
      new Expect[]{
        new Expect("pv", LLVMPointerType.get(SrcVoidType.getLLVMTypeAsPointerTarget(ctxt), 0)),
        new Expect("pi", LLVMPointerType.get(SrcIntType.getLLVMType(ctxt), 0)),
        new Expect("pf", LLVMPointerType.get(SrcFloatType.getLLVMType(ctxt), 0)),
        new Expect("ppi", LLVMPointerType.get(
                            LLVMPointerType.get(SrcIntType.getLLVMType(ctxt), 0), 0)),
        new Expect("pppi", LLVMPointerType.get(
                             LLVMPointerType.get(
                               LLVMPointerType.get(SrcFloatType.getLLVMType(ctxt), 0),
                             0),
                           0)),
        new Expect("paf", LLVMPointerType.get(
                            LLVMArrayType.get(SrcFloatType.getLLVMType(ctxt), 10), 0)),
        new Expect("pFn0", LLVMPointerType.get(getFnTy(ctxt, true), 0)),
        new Expect("pFn1", LLVMPointerType.get(getFnTy(ctxt, SrcIntType.getLLVMType(ctxt)), 0)),
        new Expect("pFn2", LLVMPointerType.get(getFnTy(
                             ctxt,
                             LLVMPointerType.get(getFnTy(ctxt, true), 0),
                             LLVMPointerType.get(getFnTy(ctxt, SrcIntType.getLLVMType(ctxt)), 0),
                             LLVMPointerType.get(SrcIntType.getLLVMType(ctxt), 0)
                           ), 0)),
      });
  }

  @Test public void array() throws IOException {
    final SimpleResult simpleResult = checkDecls(
      new String[]{
        "int ai[2];",
        "float af[1];",
        "int aai[3][5];",
        "int aaai[7][8][2];",
        "int *api[6+5];", // dimension is expression not integer literal
        "int (*apai[11])[3];",
        "void (*apFn[9])(void);",
        "int (*incompleteArrayPtr)[];",
        "int sizeFromInit[] = {2, 5};",
        "int tnDef[];", // tentative def of incomplete array type has {0} as init
        "void fn() {",
        "  int sizeFromInitLocal[] = {9, 88, -7};",
        "  static int sizeFromInitLocalStatic[] = {1};",
        "}",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMIntegerType llvmIntType = SrcIntType.getLLVMType(ctxt);
    checkDecls(mod,
      new Expect[]{
        new Expect("ai", LLVMArrayType.get(llvmIntType, 2)),
        new Expect("af", LLVMArrayType.get(SrcFloatType.getLLVMType(ctxt), 1)),
        new Expect("aai", LLVMArrayType.get(
                            LLVMArrayType.get(llvmIntType, 5), 3)),
        new Expect("aaai", LLVMArrayType.get(
                             LLVMArrayType.get(
                               LLVMArrayType.get(llvmIntType, 2),
                             8),
                           7)),
        new Expect("api", LLVMArrayType.get(
                            LLVMPointerType.get(llvmIntType, 0), 11)),
        new Expect("apai", LLVMArrayType.get(
                             LLVMPointerType.get(
                               LLVMArrayType.get(llvmIntType, 3), 0),
                           11)),
        new Expect("apFn", LLVMArrayType.get(
                             LLVMPointerType.get(getFnTy(ctxt), 0),
                           9)),
        new Expect("incompleteArrayPtr",
                   LLVMPointerType.get(LLVMArrayType.get(llvmIntType, 0), 0)),
        new Expect("sizeFromInit", LLVMArrayType.get(llvmIntType, 2),
                   LLVMConstantArray.get(
                     llvmIntType,
                     new LLVMConstant[]{
                       LLVMConstantInteger.get(llvmIntType, 2, true),
                       LLVMConstantInteger.get(llvmIntType, 5, true),
                     })),
        new Expect("tnDef", LLVMArrayType.get(llvmIntType, 1),
                   LLVMConstantArray.constNull(LLVMArrayType.get(llvmIntType,
                                                                 1))),
        // The testing framework we have right now doesn't search for store
        // instructions that initialize elements of a local variable. It only
        // looks for store instructions for the variable itself, so there are
        // none for a local non-static variable of type array. That's ok as
        // we're still checking that the type has the right number of
        // elements, which came from the initializer.
        new Expect("fn", getFnTy(ctxt),
                   new Expect("sizeFromInitLocal",
                              LLVMArrayType.get(llvmIntType, 3))),
        new Expect("fn.sizeFromInitLocalStatic",
                   LLVMArrayType.get(llvmIntType, 1),
                   LLVMConstantArray.get(
                     llvmIntType,
                     new LLVMConstant[]{
                       LLVMConstantInteger.get(llvmIntType, 1, true),
                     })),
      });
  }

  @Test public void functionReturnType() throws IOException {
    final SimpleResult simpleResult = checkDecls(
      new String[]{
        "void v(void);",
        "int i(void);",
        "float f(void);",
        "void *pv(void);",
        "int *pi(void);",
        "int (*pFn(void))();",
        "int (*pa(void))[3];",
        "struct Incomplete is();",

        // Some pointer return types (at least void* and int* below) used to
        // fail for function definitions (but not for function prototypes)
        // because BuildLLVM called Procedure.getReturnType instead of
        // Procedure.getSpecifiers.
        //
        // pFnDef used to fail because the returned function pointer type was
        // erroneously assumed to specify zero parameters (not unspecified
        // parameters) simply because it was the return type on a function
        // definition (which cannot have unspecified parameters).
        "void vDef() {}",
        "int iDef() {}",
        "float fDef() {}",
        "void *pvDef() {}",
        "int *piDef() {}",
        "int (*pFnDef())() {}",
        "int (*paDef())[3] {}",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    LLVMFunctionType vType = getFnTy(LLVMVoidType.get(ctxt), false);
    LLVMFunctionType iType = getFnTy(SrcIntType.getLLVMType(ctxt), false);
    LLVMFunctionType fType = getFnTy(SrcFloatType.getLLVMType(ctxt), false);
    LLVMFunctionType pvType
      = getFnTy(LLVMPointerType.get(SrcVoidType.getLLVMTypeAsPointerTarget(ctxt),
                                     0),
                 false);
    LLVMFunctionType piType
      = getFnTy(LLVMPointerType.get(SrcIntType.getLLVMType(ctxt), 0), false);
    LLVMFunctionType pFnType
      = getFnTy(LLVMPointerType.get(getFnTy(SrcIntType.getLLVMType(ctxt)), 0),
                 false);
    LLVMFunctionType paType
      = getFnTy(LLVMPointerType.get(LLVMArrayType.get(
                                       SrcIntType.getLLVMType(ctxt), 3), 0),
                 false);
    final LLVMType structIncomplete = mod.getTypeByName("struct.Incomplete");
    final LLVMFunctionType isType = getFnTy(structIncomplete, true);
    checkDecls(mod,
      new Expect[]{
        new Expect("v", vType),
        new Expect("i", iType),
        new Expect("f", fType),
        new Expect("pv", pvType),
        new Expect("pi", piType),
        new Expect("pFn", pFnType),
        new Expect("pa", paType),
        new Expect("is", isType),

        new Expect("vDef", vType),
        new Expect("iDef", iType),
        new Expect("fDef", fType),
        new Expect("pvDef", pvType),
        new Expect("piDef", piType),
        new Expect("pFnDef", pFnType),
        new Expect("paDef", paType),
      });
  }

  @Test public void functionDefReturnTypeIncomplete() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "function definition return type has incomplete type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "struct T foo() {}",
      });
  }

  @Test public void functionParamType() throws IOException {
    final SimpleResult simpleResult = checkDecls(
      new String[]{
        "void i(int);",
        "void f(float);",
        "void i_f(int, float);",
        "void pv(void *);",
        "void pi(int *);",
        "void ai(int[3]);",
        "void a0i(int[]);",
        "void fn0(void());",
        "void fnV(void(void));",
        "void fn1(void(float));",
        "void fnRetI(int p());",
        "void pFnRetI(int(*)());",
        "void pa(int(*)[5]);",
        "void is(struct Incomplete);",

        // Check that symbols can be looked up from a parameter list. The
        // trouble here is that Cetus does not connect the ProcedureDeclarator
        // node or NestedDeclarator node to the logical parent Procedure node
        // for which it's a prototype. To exercise this case, these must be
        // function definitions not just prototypes.
        "typedef int T;",
        "void paramTypeLookup(T i) {}",
        "int (*paramTypeLookupNested(T i))[3] {}",
        "int global;",
        "void paramIdentifierLookup(int (*)[sizeof global]);",
        "int (*paramIdentifierLookupNested(int (*)[sizeof global]))[5];",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMType structIncomplete = mod.getTypeByName("struct.Incomplete");
    checkDecls(mod,
      new Expect[]{
        new Expect("i", getFnTy(ctxt, SrcIntType.getLLVMType(ctxt))),
        new Expect("f", getFnTy(ctxt, SrcFloatType.getLLVMType(ctxt))),
        new Expect("i_f", getFnTy(ctxt, SrcIntType.getLLVMType(ctxt), SrcFloatType.getLLVMType(ctxt))),
        new Expect("pv", getFnTy(ctxt, LLVMPointerType.get(SrcVoidType.getLLVMTypeAsPointerTarget(ctxt), 0))),
        new Expect("pi", getFnTy(ctxt, LLVMPointerType.get(SrcIntType.getLLVMType(ctxt), 0))),
        new Expect("ai", getFnTy(ctxt, LLVMPointerType.get(SrcIntType.getLLVMType(ctxt), 0))),
        new Expect("a0i", getFnTy(ctxt, LLVMPointerType.get(SrcIntType.getLLVMType(ctxt), 0))),
        new Expect("fn0", getFnTy(ctxt, LLVMPointerType.get(getFnTy(ctxt, true), 0))),
        new Expect("fnV", getFnTy(ctxt, LLVMPointerType.get(getFnTy(ctxt), 0))),
        new Expect("fn1", getFnTy(ctxt, LLVMPointerType.get(getFnTy(ctxt, SrcFloatType.getLLVMType(ctxt)), 0))),
        new Expect("fnRetI", getFnTy(ctxt, LLVMPointerType.get(getFnTy(SrcIntType.getLLVMType(ctxt)), 0))),
        new Expect("pFnRetI", getFnTy(ctxt, LLVMPointerType.get(getFnTy(SrcIntType.getLLVMType(ctxt)), 0))),
        new Expect("pa", getFnTy(ctxt, LLVMPointerType.get(LLVMArrayType.get(SrcIntType.getLLVMType(ctxt), 5), 0))),
        new Expect("is", getFnTy(ctxt, structIncomplete)),

        new Expect("paramTypeLookup",
                   LLVMFunctionType.get(LLVMVoidType.get(ctxt), false,
                                        SrcIntType.getLLVMType(ctxt))),
        new Expect("paramTypeLookupNested",
                   LLVMFunctionType.get(
                     LLVMPointerType.get(
                       LLVMArrayType.get(SrcIntType.getLLVMType(ctxt), 3), 0),
                     false, SrcIntType.getLLVMType(ctxt))),
        new Expect("global", SrcIntType.getLLVMType(ctxt)),
        new Expect("paramIdentifierLookup",
                   getFnTy(ctxt, LLVMPointerType.get(
                                   LLVMArrayType.get(
                                     SrcIntType.getLLVMType(ctxt),
                                     SrcIntType.getLLVMWidth()/8),
                                 0))),
        new Expect("paramIdentifierLookupNested",
                   LLVMFunctionType.get(
                     LLVMPointerType.get(
                       LLVMArrayType.get(SrcIntType.getLLVMType(ctxt), 5), 0),
                     false,
                     LLVMPointerType.get(
                       LLVMArrayType.get(
                         SrcIntType.getLLVMType(ctxt),
                         SrcIntType.getLLVMWidth()/8),
                     0))),
      });
  }

  @Test public void functionDefParamTypeIncomplete() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "function definition parameter 1 type has incomplete type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "struct T;",
        "void foo(struct T) {}",
      });
  }

  // Make sure the specifiers from previous declarators are cleared before
  // computing the types of later declarators in the same declaration.
  @Test public void multipleDeclarators() throws IOException {
    final SimpleResult simpleResult = checkDecls(
      new String[]{
        "int i, *p, j, a[3], k, f(int), h;",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    checkDecls(mod,
      new Expect[]{
        new Expect("i", SrcIntType.getLLVMType(ctxt)),
        new Expect("p", LLVMPointerType.get(SrcIntType.getLLVMType(ctxt), 0)),
        new Expect("j", SrcIntType.getLLVMType(ctxt)),
        new Expect("a", LLVMArrayType.get(SrcIntType.getLLVMType(ctxt), 3)),
        new Expect("k", SrcIntType.getLLVMType(ctxt)),
        new Expect("f", LLVMFunctionType.get(SrcIntType.getLLVMType(ctxt),
                                             false,
                                             SrcIntType.getLLVMType(ctxt))),
        new Expect("h", SrcIntType.getLLVMType(ctxt)),
      });
  }

  @Test public void structFileScope() throws IOException {
    final SimpleResult simpleResult = checkDecls(
      new String[]{
        "struct Def0Only { int i; };",
        "struct Def1Only { int *i; } def1OnlyIn;",
        "struct Def2Only { int i, j; float k; } def2OnlyIn0, *def2OnlyIn1;",
        "struct DefDecl { int *i; } defDeclIn;",
        "struct DefDecl;",
        "struct DefRef { int i[3]; };",
        "struct DefRef *defRefEx0, defRefEx1[3];",

        "struct DeclOnly;",
        "struct DeclDecl;",
        "struct DeclDecl;",
        "struct DeclDef;",
        "struct DeclDef { void (*f)(void); } declDef;",
        "struct DeclRef;",
        "struct DeclRef *declRef;",

        "struct RefOnlyA *refOnlyA0;",
        "struct RefOnlyA *refOnlyA1;",
        "struct RefOnlyB refOnlyB0();",
        "struct RefOnlyA *refOnlyA2;",
        "struct RefOnlyB *refOnlyB1;",
        "struct RefDef *refDef;",
        "struct RefDef { int x; };",
        "struct RefDecl *refDecl;",
        "struct RefDecl;",
      });
    final LLVMModule llvmModule = simpleResult.llvmModule;
    final LLVMContext ctxt = llvmModule.getContext();
    checkDecls(llvmModule,
      new ExpectStruct[]{
        new ExpectStruct("struct.Def0Only", SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("struct.Def1Only", LLVMPointerType.get(SrcIntType.getLLVMType(ctxt), 0)),
        new ExpectStruct("struct.Def2Only", SrcIntType.getLLVMType(ctxt), SrcIntType.getLLVMType(ctxt),
                         SrcFloatType.getLLVMType(ctxt)),
        new ExpectStruct("struct.DefDecl", LLVMPointerType.get(SrcIntType.getLLVMType(ctxt), 0)),
        new ExpectStruct("struct.DefRef", LLVMArrayType.get(SrcIntType.getLLVMType(ctxt), 3)),

        new ExpectStruct("struct.DeclOnly", (LLVMType[])null),
        new ExpectStruct("struct.DeclDecl", (LLVMType[])null),
        new ExpectStruct("struct.DeclDecl", (LLVMType[])null),
        new ExpectStruct("struct.DeclDef", LLVMPointerType.get(getFnTy(ctxt, false), 0)),
        new ExpectStruct("struct.DeclRef", (LLVMType[])null),

        new ExpectStruct("struct.RefOnlyA", (LLVMType[])null),
        new ExpectStruct("struct.RefOnlyB", (LLVMType[])null),
        new ExpectStruct("struct.RefDef", SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("struct.RefDecl", (LLVMType[])null),
      });
    LLVMType Def1Only = llvmModule.getTypeByName("struct.Def1Only");
    LLVMType Def2Only = llvmModule.getTypeByName("struct.Def2Only");
    LLVMType DefRef = llvmModule.getTypeByName("struct.DefRef");
    LLVMType DefDecl = llvmModule.getTypeByName("struct.DefDecl");
    LLVMType DeclRef = llvmModule.getTypeByName("struct.DeclRef");
    LLVMType RefOnlyA = llvmModule.getTypeByName("struct.RefOnlyA");
    LLVMType RefOnlyB = llvmModule.getTypeByName("struct.RefOnlyB");
    LLVMType RefDef = llvmModule.getTypeByName("struct.RefDef");
    LLVMType RefDecl = llvmModule.getTypeByName("struct.RefDecl");
    checkDecls(llvmModule,
      new Expect[]{
        new Expect("def1OnlyIn", Def1Only),
        new Expect("def2OnlyIn0", Def2Only),
        new Expect("def2OnlyIn1", LLVMPointerType.get(Def2Only, 0)),
        new Expect("defDeclIn", DefDecl),
        new Expect("defRefEx0", LLVMPointerType.get(DefRef, 0)),
        new Expect("defRefEx1", LLVMArrayType.get(DefRef, 3)),

        new Expect("declDef", "struct.DeclDef"),
        new Expect("declRef", LLVMPointerType.get(DeclRef, 0)),

        new Expect("refOnlyA0", LLVMPointerType.get(RefOnlyA, 0)),
        new Expect("refOnlyA1", LLVMPointerType.get(RefOnlyA, 0)),
        new Expect("refOnlyB0", getFnTy(RefOnlyB)),
        new Expect("refOnlyA2", LLVMPointerType.get(RefOnlyA, 0)),
        new Expect("refOnlyB1", LLVMPointerType.get(RefOnlyB, 0)),
        new Expect("refDef", LLVMPointerType.get(RefDef, 0)),
        new Expect("refDecl", LLVMPointerType.get(RefDecl, 0)),
      });
  }

  /**
   * TODO: Due to a Cetus bug, {@link BuildLLVM} currently resolves the scope
   * of a struct declaration/definition in a parameter list incorrectly as the
   * enclosing scope. See comments within
   * {@link BuildLLVM.Visitor#lookupSymbolTable} for details. This test
   * checks that {@link BuildLLVM} is consistent in this regard, but
   * ultimately this behavior needs to be corrected.
   */
  @Test public void structParam() throws IOException {
    final SimpleResult simpleResult = checkDecls(
      new String[]{
        "void fnDefInRefOut(struct DefInRefOut {int i;} p);",
        "struct DefInRefOut defInRefOut;",
        "void fnRefInDefOut(int i, struct RefInDefOut *p);",
        "struct RefInDefOut {int i;} refInDefOut;",

        "struct DefOutRefIn {int i;};",
        "void fnDefOutRefIn(struct DefOutRefIn p);",
        "struct DeclOutRefIn;",
        "void fnDeclOutRefIn(struct DeclOutRefIn p);",
        "struct RefOutRefIn *refOutRefIn;",
        "void fnRefOutRefIn(struct RefOutRefIn *p);",
        "struct RefOutDefIn *refOutDefIn;",
        "void fnRefOutDefIn(struct RefOutDefIn {int x;} p);",

        "void fnDefInRefOutFnDef(struct DefInRefOutFnDef {int i;} p) {}",
        "struct DefInRefOutFnDef defInRefOutFnDef;",
        "void fnRefInDefOutFnDef(struct RefInDefOutFnDef *p) {}",
        "struct RefInDefOutFnDef {int i;} refInDefOutFnDef;",
        "struct DefOutRefInFnDef {int i;};",
        "void fnDefOutRefInFnDef(struct DefOutRefInFnDef p) {}",
        "struct RefOutDefInFnDef *refOutDefInFnDef;",
        "void fnRefOutDefInFnDef(struct RefOutDefInFnDef {int x;} p);",
      });
    final LLVMModule llvmModule = simpleResult.llvmModule;
    final LLVMContext ctxt = llvmModule.getContext();
    checkDecls(llvmModule,
      new ExpectStruct[]{
        new ExpectStruct("struct.DefInRefOut", SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("struct.RefInDefOut", SrcIntType.getLLVMType(ctxt)),

        new ExpectStruct("struct.DefOutRefIn", SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("struct.DeclOutRefIn", (LLVMType[])null),
        new ExpectStruct("struct.RefOutRefIn", (LLVMType[])null),
        new ExpectStruct("struct.RefOutDefIn", SrcIntType.getLLVMType(ctxt)),

        new ExpectStruct("struct.DefInRefOutFnDef", SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("struct.RefInDefOutFnDef", SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("struct.DefOutRefInFnDef", SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("struct.RefOutDefInFnDef", SrcIntType.getLLVMType(ctxt)),
      });
    LLVMType DefInRefOut = llvmModule.getTypeByName("struct.DefInRefOut");
    LLVMType RefInDefOut = llvmModule.getTypeByName("struct.RefInDefOut");
    LLVMType DefOutRefIn = llvmModule.getTypeByName("struct.DefOutRefIn");
    LLVMType DeclOutRefIn = llvmModule.getTypeByName("struct.DeclOutRefIn");
    LLVMType RefOutRefIn = llvmModule.getTypeByName("struct.RefOutRefIn");
    LLVMType RefOutDefIn = llvmModule.getTypeByName("struct.RefOutDefIn");
    LLVMType DefInRefOutFnDef = llvmModule.getTypeByName("struct.DefInRefOutFnDef");
    LLVMType RefInDefOutFnDef = llvmModule.getTypeByName("struct.RefInDefOutFnDef");
    LLVMType DefOutRefInFnDef = llvmModule.getTypeByName("struct.DefOutRefInFnDef");
    LLVMType RefOutDefInFnDef = llvmModule.getTypeByName("struct.RefOutDefInFnDef");
    checkDecls(llvmModule,
      new Expect[]{
        new Expect("fnDefInRefOut", getFnTy(ctxt, DefInRefOut)),
        new Expect("defInRefOut", DefInRefOut),
        new Expect("fnRefInDefOut",
                   getFnTy(ctxt, SrcIntType.getLLVMType(ctxt),
                              LLVMPointerType.get(RefInDefOut, 0))),
        new Expect("refInDefOut", RefInDefOut),

        new Expect("fnDefOutRefIn", getFnTy(ctxt, DefOutRefIn)),
        new Expect("fnDeclOutRefIn", getFnTy(ctxt, DeclOutRefIn)),
        new Expect("fnRefOutRefIn",
                   getFnTy(ctxt, LLVMPointerType.get(RefOutRefIn, 0))),
        new Expect("refOutRefIn", LLVMPointerType.get(RefOutRefIn, 0)),
        new Expect("refOutDefIn", LLVMPointerType.get(RefOutDefIn, 0)),
        new Expect("fnRefOutDefIn", getFnTy(ctxt, RefOutDefIn)),

        new Expect("fnDefInRefOutFnDef", getFnTy(ctxt, DefInRefOutFnDef)),
        new Expect("defInRefOutFnDef", DefInRefOutFnDef),
        new Expect("fnRefInDefOutFnDef",
                    getFnTy(ctxt, LLVMPointerType.get(RefInDefOutFnDef, 0))),
        new Expect("refInDefOutFnDef", RefInDefOutFnDef),
        new Expect("fnDefOutRefInFnDef", getFnTy(ctxt, DefOutRefInFnDef)),
        new Expect("refOutDefInFnDef", LLVMPointerType.get(RefOutDefInFnDef, 0)),
        new Expect("fnRefOutDefInFnDef", getFnTy(ctxt, RefOutDefInFnDef)),
      });
  }

  @Test public void structMultiScope() throws IOException {
    final SimpleResult simpleResult = checkDecls(
      new String[]{
        "void fnDefInDefOut() { struct DefInDefOut { float x; } l; }",
        "struct DefInDefOut { double x; } defInDefOut;",
        "struct DefOutDefIn { int x; } defOutDefIn;",
        "void fnDefOutDefIn() { struct DefOutDefIn { float x; } l; }",

        "void fnDefInDeclOut() { struct DefInDeclOut { int x; } l; }",
        "struct DefInDeclOut;",
        "struct DeclOutDefIn;",
        "void fnDeclOutDefIn() { struct DeclOutDefIn { int x; } l; }",
        "void fnDeclInDefOut() { struct DeclInDefOut; }",
        "struct DeclInDefOut { int x; } declInDefOut;",
        "struct DefOutDeclIn { int x; } defOutDeclIn;",
        "void fnDefOutDeclIn() { struct DefOutDeclIn; }",

        "void fnDefInRefOut() { struct DefInRefOut { int x; } l; }",
        "struct DefInRefOut *defInRefOut;",
        "struct RefOutDefIn *refOutDefIn;",
        "void fnRefOutDefIn() { struct RefOutDefIn { int x; } l; }",
        "void fnRefInDefOut() { struct RefInDefOut *l; }",
        "struct RefInDefOut { int x; } refInDefOut;",
        "struct DefOutRefIn { int x; } defOutRefIn;",
        "void fnDefOutRefIn() { struct DefOutRefIn l; }",

        "void fnDeclInDeclOut() { struct DeclInDeclOut; }",
        "struct DeclInDeclOut;",
        "struct DeclOutDeclIn;",
        "void fnDeclOutDeclIn() { struct DeclOutDeclIn; }",

        "void fnDeclInRefOut() { struct DeclInRefOut; }",
        "struct DeclInRefOut *declInRefOut;",
        "struct RefOutDeclIn *refOutDeclIn;",
        "void fnRefOutDeclIn() { struct RefOutDeclIn; }",
        "void fnRefInDeclOut() { struct RefInDeclOut *l; }",
        "struct RefInDeclOut;",
        "struct DeclOutRefIn;",
        "void fnDeclOutRefIn() { struct DeclOutRefIn *l; }",

        "void fnRefInRefOut() { struct RefInRefOut *l; }",
        "struct RefInRefOut *refInRefOut;",
        "struct RefOutRefIn *refOutRefIn;",
        "void fnRefOutRefIn() { struct RefOutRefIn *l; }",

        "struct DefDefDef { int x; } defDefDef;",
        "struct DefRefDef { int x; } defRefDef;",
        "struct DeclDeclDecl;",
        "struct DeclDeclRef;",
        "struct RefRefRef *refRefRef;",
        "void fnMany() {",
        "  struct DefDefDef { float x; } defDefDef;",
        "  struct DefRefDef defRefDef;",
        "  struct DeclDeclDecl;",
        "  struct DeclDeclRef;",
        "  struct RefRefRef *refRefRef;",
        "  {",
        "    struct DefDefDef { double x; } defDefDef;",
        "    struct DefRefDef { double x; } defRefDef;",
        "    struct DeclDeclDecl;",
        "    struct DeclDeclRef *declDeclRef;",
        "    struct RefRefRef *refRefRef;",
        "  }",
        "}",
      });
    final LLVMModule llvmModule = simpleResult.llvmModule;
    final LLVMContext ctxt = llvmModule.getContext();
    checkDecls(llvmModule,
      new ExpectStruct[]{
        new ExpectStruct("struct.DefInDefOut", SrcFloatType.getLLVMType(ctxt)),
        new ExpectStruct("struct.DefInDefOut.0", SrcDoubleType.getLLVMType(ctxt)),
        new ExpectStruct("struct.DefOutDefIn", SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("struct.DefOutDefIn.1", SrcFloatType.getLLVMType(ctxt)),

        new ExpectStruct("struct.DefInDeclOut", SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("struct.DefInDeclOut.2", (LLVMType[])null),
        new ExpectStruct("struct.DeclOutDefIn", (LLVMType[])null),
        new ExpectStruct("struct.DeclOutDefIn.3", SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("struct.DeclInDefOut", (LLVMType[])null),
        new ExpectStruct("struct.DeclInDefOut.4", SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("struct.DefOutDeclIn", SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("struct.DefOutDeclIn.5", (LLVMType[])null),

        new ExpectStruct("struct.DefInRefOut", SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("struct.DefInRefOut.6", (LLVMType[])null),
        new ExpectStruct("struct.RefOutDefIn", (LLVMType[])null),
        new ExpectStruct("struct.RefOutDefIn.7", SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("struct.RefInDefOut", (LLVMType[])null),
        new ExpectStruct("struct.RefInDefOut.8", SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("struct.DefOutRefIn", SrcIntType.getLLVMType(ctxt)),

        new ExpectStruct("struct.DeclInDeclOut", (LLVMType[])null),
        new ExpectStruct("struct.DeclInDeclOut.9", (LLVMType[])null),
        new ExpectStruct("struct.DeclOutDeclIn", (LLVMType[])null),
        new ExpectStruct("struct.DeclOutDeclIn.10", (LLVMType[])null),

        new ExpectStruct("struct.DeclInRefOut", (LLVMType[])null),
        new ExpectStruct("struct.DeclInRefOut.11", (LLVMType[])null),
        new ExpectStruct("struct.RefOutDeclIn", (LLVMType[])null),
        new ExpectStruct("struct.RefOutDeclIn.12", (LLVMType[])null),
        new ExpectStruct("struct.RefInDeclOut", (LLVMType[])null),
        new ExpectStruct("struct.RefInDeclOut.13", (LLVMType[])null),
        new ExpectStruct("struct.DeclOutRefIn", (LLVMType[])null),

        new ExpectStruct("struct.RefInRefOut", (LLVMType[])null),
        new ExpectStruct("struct.RefInRefOut.14", (LLVMType[])null),
        new ExpectStruct("struct.RefOutRefIn", (LLVMType[])null),

        new ExpectStruct("struct.DefDefDef", SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("struct.DefRefDef", SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("struct.DeclDeclDecl", (LLVMType[])null),
        new ExpectStruct("struct.DeclDeclRef", (LLVMType[])null),
        new ExpectStruct("struct.RefRefRef", (LLVMType[])null),
        new ExpectStruct("struct.DefDefDef.15", SrcFloatType.getLLVMType(ctxt)),
        new ExpectStruct("struct.DeclDeclDecl.16", (LLVMType[])null),
        new ExpectStruct("struct.DeclDeclRef.17", (LLVMType[])null),
        new ExpectStruct("struct.DefDefDef.18", SrcDoubleType.getLLVMType(ctxt)),
        new ExpectStruct("struct.DefRefDef.19", SrcDoubleType.getLLVMType(ctxt)),
        new ExpectStruct("struct.DeclDeclDecl.20", (LLVMType[])null),
      });
    LLVMType DefInRefOut6 = llvmModule.getTypeByName("struct.DefInRefOut.6");
    LLVMType RefOutDefIn = llvmModule.getTypeByName("struct.RefOutDefIn");
    LLVMType RefInDefOut = llvmModule.getTypeByName("struct.RefInDefOut");
    LLVMType DeclInRefOut11 = llvmModule.getTypeByName("struct.DeclInRefOut.11");
    LLVMType RefOutDeclIn = llvmModule.getTypeByName("struct.RefOutDeclIn");
    LLVMType RefInDeclOut = llvmModule.getTypeByName("struct.RefInDeclOut");
    LLVMType DeclOutRefIn = llvmModule.getTypeByName("struct.DeclOutRefIn");
    LLVMType RefInRefOut = llvmModule.getTypeByName("struct.RefInRefOut");
    LLVMType RefInRefOut14 = llvmModule.getTypeByName("struct.RefInRefOut.14");
    LLVMType RefOutRefIn = llvmModule.getTypeByName("struct.RefOutRefIn");
    LLVMType RefRefRef = llvmModule.getTypeByName("struct.RefRefRef");
    LLVMType DeclDeclRef17 = llvmModule.getTypeByName("struct.DeclDeclRef.17");
    checkDecls(llvmModule,
      new Expect[]{
        new Expect("fnDefInDefOut", getFnTy(ctxt),
                   new Expect("l", "struct.DefInDefOut")),
        new Expect("defInDefOut", "struct.DefInDefOut.0"),
        new Expect("defOutDefIn", "struct.DefOutDefIn"),
        new Expect("fnDefOutDefIn", getFnTy(ctxt),
                   new Expect("l", "struct.DefOutDefIn.1")),

        new Expect("fnDefInDeclOut", getFnTy(ctxt),
                   new Expect("l", "struct.DefInDeclOut")),
        new Expect("fnDeclOutDefIn", getFnTy(ctxt),
                   new Expect("l", "struct.DeclOutDefIn.3")),
        new Expect("fnDeclInDefOut", getFnTy(ctxt)),
        new Expect("declInDefOut", "struct.DeclInDefOut.4"),
        new Expect("defOutDeclIn", "struct.DefOutDeclIn"),
        new Expect("fnDefOutDeclIn", getFnTy(ctxt)),

        new Expect("fnDefInRefOut", getFnTy(ctxt),
                   new Expect("l", "struct.DefInRefOut")),
        new Expect("defInRefOut", LLVMPointerType.get(DefInRefOut6, 0)),
        new Expect("refOutDefIn", LLVMPointerType.get(RefOutDefIn, 0)),
        new Expect("fnRefOutDefIn", getFnTy(ctxt),
                   new Expect("l", "struct.RefOutDefIn.7")),
        new Expect("fnRefInDefOut", getFnTy(ctxt),
                   new Expect("l", LLVMPointerType.get(RefInDefOut, 0))),
        new Expect("refInDefOut", "struct.RefInDefOut.8"),
        new Expect("defOutRefIn", "struct.DefOutRefIn"),
        new Expect("fnDefOutRefIn", getFnTy(ctxt),
                   new Expect("l", "struct.DefOutRefIn")),

        new Expect("fnDeclOutDeclIn", getFnTy(ctxt)),
        new Expect("fnDeclInDeclOut", getFnTy(ctxt)),

        new Expect("fnDeclInRefOut", getFnTy(ctxt)),
        new Expect("declInRefOut", LLVMPointerType.get(DeclInRefOut11, 0)),
        new Expect("refOutDeclIn", LLVMPointerType.get(RefOutDeclIn, 0)),
        new Expect("fnRefOutDeclIn", getFnTy(ctxt)),
        new Expect("fnRefInDeclOut", getFnTy(ctxt),
                   new Expect("l", LLVMPointerType.get(RefInDeclOut, 0))),
        new Expect("fnDeclOutRefIn", getFnTy(ctxt),
                   new Expect("l", LLVMPointerType.get(DeclOutRefIn, 0))),
        new Expect("fnRefInRefOut", getFnTy(ctxt),
                   new Expect("l", LLVMPointerType.get(RefInRefOut, 0))),
        new Expect("refInRefOut", LLVMPointerType.get(RefInRefOut14, 0)),
        new Expect("refOutRefIn", LLVMPointerType.get(RefOutRefIn, 0)),
        new Expect("fnRefOutRefIn", getFnTy(ctxt),
                   new Expect("l", LLVMPointerType.get(RefOutRefIn, 0))),

        new Expect("defDefDef", "struct.DefDefDef"),
        new Expect("defRefDef", "struct.DefRefDef"),
        new Expect("refRefRef", LLVMPointerType.get(RefRefRef, 0)),
        new Expect("fnMany", getFnTy(ctxt),
                   new Expect("defDefDef", "struct.DefDefDef.15"),
                   new Expect("defRefDef", "struct.DefRefDef"),
                   new Expect("refRefRef", LLVMPointerType.get(RefRefRef, 0)),
                   new Expect("defDefDef1", "struct.DefDefDef.18"),
                   new Expect("defRefDef2", "struct.DefRefDef.19"),
                   new Expect("declDeclRef", LLVMPointerType.get(DeclDeclRef17, 0)),
                   new Expect("refRefRef3", LLVMPointerType.get(RefRefRef, 0))),
      });
  }

  @Test public void structNested() throws IOException {
    final SimpleResult simpleResult = checkDecls(
      new String[]{
        "struct Outer {", // defines file-scope Outer
        "  double x;",
        "  struct Inner {", // defines file-scope Inner
        "    int x;",
        "    float y;",
        "  } inner;",
        "  struct Ref *ref;", // implicitly declares file-scope Ref
        "  struct Outer *outer;", // recursive
        "  int y;",
        "} outer;",
        "struct Inner inner;", // refers to file-scope Inner
        "struct Ref *ref;", // refers to file-scope Ref
        "void foo() {",
        "  struct Outer {", // defines block-scope Outer
        "    struct Inner {", // defines block-scope Inner
        "      int x;",
        "    } inner;",
        "    struct Outer *x;", // recursive: doesn't refer to file-scope Outer
        "  } outer;",
        "  struct Inner inner;", // refers to block-scope Inner
        "  struct Ref *ref;", // refers to file-scope Ref
        "}",
        "void bar() {",
        "  struct Outer {", // defines block-scope Outer
        "    struct Inner inner;", // refers to file-scope Inner
        "    struct Ref *ref;", // refers to file-scope Ref
        "    int x;",
        "  } outer;",
        "  struct Inner inner;", // refers to file-scope Inner
        "  struct Ref *ref;", // refers to file-scope Ref
        "}",
      });
    final LLVMModule llvmModule = simpleResult.llvmModule;
    final LLVMContext ctxt = llvmModule.getContext();
    LLVMType Inner = llvmModule.getTypeByName("struct.Inner");
    LLVMType Outer = llvmModule.getTypeByName("struct.Outer");
    LLVMType Ref = llvmModule.getTypeByName("struct.Ref");
    LLVMType Inner0 = llvmModule.getTypeByName("struct.Inner.0");
    LLVMType Outer1 = llvmModule.getTypeByName("struct.Outer.1");
    checkDecls(llvmModule,
      new ExpectStruct[]{
        new ExpectStruct("struct.Inner", SrcIntType.getLLVMType(ctxt),
                         SrcFloatType.getLLVMType(ctxt)),
        new ExpectStruct("struct.Outer", SrcDoubleType.getLLVMType(ctxt),
                         Inner, LLVMPointerType.get(Ref, 0),
                         LLVMPointerType.get(Outer, 0),
                         SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("struct.Ref", (LLVMType[])null),
        new ExpectStruct("struct.Inner.0", SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("struct.Outer.1", Inner0,
                         LLVMPointerType.get(Outer1, 0)),
        new ExpectStruct("struct.Outer.2", Inner, LLVMPointerType.get(Ref, 0),
                         SrcIntType.getLLVMType(ctxt)),
      });
    checkDecls(llvmModule,
      new Expect[]{
        new Expect("outer", "struct.Outer"),
        new Expect("inner", "struct.Inner"),
        new Expect("ref", LLVMPointerType.get(Ref, 0)),
        new Expect("foo", getFnTy(ctxt),
                   new Expect("outer", "struct.Outer.1"),
                   new Expect("inner", "struct.Inner.0"),
                   new Expect("ref", LLVMPointerType.get(Ref, 0))),
        new Expect("bar", getFnTy(ctxt),
                   new Expect("outer", "struct.Outer.2"),
                   new Expect("inner", "struct.Inner"),
                   new Expect("ref", LLVMPointerType.get(Ref, 0))),
      });
  }

  @Test public void structFlexibleArray() throws IOException {
    final SimpleResult simpleResult = checkDecls(
      new String[]{
        "struct T { int i; int a[]; } t;",
        "struct S { float f; struct T arr[]; } s;",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMType T = mod.getTypeByName("struct.T");
    checkDecls(mod,
      new ExpectStruct[]{
        new ExpectStruct("struct.T", SrcIntType.getLLVMType(ctxt),
                         LLVMArrayType.get(SrcIntType.getLLVMType(ctxt), 0)),
        new ExpectStruct("struct.S", SrcFloatType.getLLVMType(ctxt),
                                     LLVMArrayType.get(T, 0)),
      });
    checkDecls(mod,
      new Expect[]{
        new Expect("t", "struct.T"),
        new Expect("s", "struct.S"),
      });
  }

  /**
   * Cetus appears to always add tags to tag-less structs, so
   * {@link BuildLLVM} requires no special handler for that case. Testing is
   * challenging because it's hard to predict the tags. So far, I usually get
   * {@code named_structAnonymous_c_3} and {@code named_structAnonymous_c_11},
   * in this test, but I don't know how likely those are to change as Cetus
   * evolves.
   */
  @Test public void structTagless() throws IOException {
    final SimpleResult simpleResult = checkDecls(
      new String[]{
        "struct {",
        "  int i;",
        "  struct {",
        "    float f;",
        "  } inner;",
        "} x;",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMGlobalVariable x = mod.getNamedGlobal("x");
    assertNotNull("x must exist", x.getInstance());
    assertTrue("x's type",
               x.typeOf() instanceof LLVMPointerType);
    LLVMPointerType xPtrType = (LLVMPointerType)x.typeOf();
    assertTrue("x's type must be a pointer to a struct",
               xPtrType.getElementType() instanceof LLVMIdentifiedStructType);
    LLVMIdentifiedStructType xType
      = (LLVMIdentifiedStructType)xPtrType.getElementType();
    LLVMType[] xMembers = xType.getElementTypes();
    assertEquals("x's type's number of members", 2, xMembers.length);
    assertEquals("x's member 0 type", SrcIntType.getLLVMType(ctxt), xMembers[0]);
    assertTrue("x's member 1 type must be a struct",
               xMembers[1] instanceof LLVMIdentifiedStructType);
    LLVMIdentifiedStructType member1Type = (LLVMIdentifiedStructType)xMembers[1];
    LLVMType[] member1Members = member1Type.getElementTypes();
    assertEquals("x's member 1's type's number of members",
                 1, member1Members.length);
    assertEquals("x's member 1's type's member 1 type",
                 member1Members[0], SrcFloatType.getLLVMType(ctxt));
  }

  @Test public void structMemberFnType() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "struct member has function type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "struct T { void f(void); };",
      });
  }

  @Test public void structMemberVoidType() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "struct member has incomplete type but is not flexible array member"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "struct T { void a; };",
      });
  }

  @Test public void structMemberIncompleteStruct() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "struct member has incomplete type but is not flexible array member"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "struct S1 { struct S2 i; };",
      });
  }

  /**
   * Use of bit-fields is checked in
   * {@link BuildLLVMTest_OtherExpressions#memberAccess} and
   * {@link BuildLLVMTest_OtherExpressions#assignment}.
   */
  @Test public void bitFields() throws IOException {
    final SimpleResult simpleResult = checkDecls(
      new String[]{
        "#include <stddef.h>",

        "struct ObviousDeclarators {",
        "  signed char i:1;",
        "  signed char pad:2;",
        "  int j:5;",
        "  signed char k:1;",
        "};",
        // Multiple declarators within a declaration, and unnamed declarators
        // are easily overlooked.
        "struct ObfuscatedDeclarators {",
        "  signed char i:1, :2;",
        "  int j:5;",
        "  signed char k:1;",
        "};",
        // Show, by contrast, what effect the padding had.
        "struct NoPadding {",
        "  signed char i:1;",
        "  int j:5;",
        "  signed char k:1;",
        "};",
        // A zero-width bit-field has more effect.
        "struct ZeroWidth {",
        "  signed char i:1, :0;",
        "  int j:5;",
        "  signed char k:1;",
        "};",

        // Make sure a bit-field doesn't pack with a non-bit-field.
        "struct NonBitFieldPacking {",
        "  int i : 16;", // could be mistaken for previous bit-field
        "  int j;",      // not a bit-field
        "  int k : 16;", // there's nothing to pack it with
        "  int l : 16;", // packs
        "};",

        // Make sure signed can packed with unsigned. TODO: Also check that
        // signedness is maintained; put that in OtherExpressions.
        "struct Signedness {",
        "  int            i : 16;",
        "  unsigned       u : 16;",
        "  unsigned char uc : 4;",
        "  signed   char sc : 4;",
        "};",

        // Integer types are allowed, and various widths smaller than the
        // specified integer type should be fine.
        "enum E { E1, E2 };",
        "struct IntegerTypes {",
        "  _Bool b:1;",
        "  short unsigned int sui:15;",
        "  short int si:16;",
        "  int i:32;",
        "  unsigned int ui:30;",
        "  enum E e:3;",
        "  long l:33;",
        "  long unsigned lu:33;",
        "  long long ll:33;",
        "  long unsigned long lul:33;",
        "};",

        // Make sure BuildLLVM can handle a sizeof expression containing a
        // bit-field width expression.  The former requires computing sizes
        // of expressions, but the latter requires computing values of
        // expressions, so exprMode should have to be pushed and popped at the
        // latter. Actually, because Cetus pulls struct definitions out of
        // expressions, it doesn't matter, but we keep this check in case that
        // behavior should change.
        "#define T struct {int i:5;}",
        "int inSizeof = sizeof ((T*)0) == sizeof(void*);",

        // Make sure width doesn't have to be simply an integer literal. Cetus
        // used to drop the bit-field width expression if Cetus couldn't
        // simplify it to an integer literal, but Cetus would have to know the
        // size of an int to do that here.
        "struct WidthExpr { int i : sizeof(int), j : 1; };",

        // Otherwise, bit-fields in unions overlap like any fields.
        "union Union { char f1:5; char f2:3; char f3:8; };",

        // Unnamed bit-fields in unions are omitted.
        "union UnionUnnamed { int : 5; char c1; int : 0; unsigned char c2; };",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    checkDecls(mod,
      new ExpectStruct[]{
        new ExpectStruct("struct.ObviousDeclarators",
                         SrcSignedCharType.getLLVMType(ctxt),
                         SrcSignedCharType.getLLVMType(ctxt)),
        new ExpectStruct("struct.ObfuscatedDeclarators",
                         SrcSignedCharType.getLLVMType(ctxt),
                         SrcSignedCharType.getLLVMType(ctxt)),
        new ExpectStruct("struct.NoPadding",
                         SrcSignedCharType.getLLVMType(ctxt)),
        new ExpectStruct("struct.ZeroWidth",
                         SrcSignedCharType.getLLVMType(ctxt),
                         SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("struct.NonBitFieldPacking",
                         SrcIntType.getLLVMType(ctxt),
                         SrcIntType.getLLVMType(ctxt),
                         SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("struct.Signedness",
                         SrcIntType.getLLVMType(ctxt),
                         SrcUnsignedCharType.getLLVMType(ctxt)),
        new ExpectStruct("struct.IntegerTypes",
                         SrcBoolType.getLLVMType(ctxt),
                         SrcUnsignedShortType.getLLVMType(ctxt),
                         SrcShortType.getLLVMType(ctxt),
                         SrcIntType.getLLVMType(ctxt),
                         SrcUnsignedIntType.getLLVMType(ctxt),
                         SrcEnumType.COMPATIBLE_INTEGER_TYPE.getLLVMType(ctxt),
                         SrcLongType.getLLVMType(ctxt),
                         SrcUnsignedLongType.getLLVMType(ctxt),
                         SrcLongLongType.getLLVMType(ctxt),
                         SrcUnsignedLongLongType.getLLVMType(ctxt)),
        new ExpectStruct("struct.WidthExpr",
                         SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("union.Union",
                         SrcCharType.getLLVMType(ctxt)),
        new ExpectStruct("union.UnionUnnamed",
                         SrcCharType.getLLVMType(ctxt)),
      });
    checkDecls(mod,
      new Expect[]{
        new Expect("inSizeof", SrcIntType.getLLVMType(ctxt), 1),
      });
  }

  /**
   * {@link BuildLLVM}'s handling of declarations/definitions of a union is
   * exactly the same as for a struct except that (1) the generated LLVM
   * struct has a tag that starts with "{@code union.}" instead of
   * "{@code struct.}", and (2) the members are altered to select the largest
   * member size and alignment. Thus, we test those aspects of unions
   * thoroughly and assume the testing of structs is mostly sufficient for the
   * other aspects of unions.
   */
  @Test public void union() throws IOException {
    final SimpleResult simpleResult = checkDecls(
      new String[]{
        "union Def { int x; } def;",
        "union Decl;",
        "union Ref *ref;",
        "void fnDef() { union Def { float f; } def; }",
        "union Pad41 { char arr[1]; float f;  } pad41;",
        "union Pad42 { int i; char arr[2];    } pad42;",
        "union Pad43 { char arr[3]; int i;    } pad43;",
        "union Pad44 { float f; char arr[4];  } pad44;",
        "union Pad45 { float f; char arr[5];  } pad45;",
        "union Pad46 { char arr[6]; float f;  } pad46;",
        "union Pad47 { int i; char arr[7];    } pad47;",
        "union Pad48 { char arr[8]; int i;    } pad48;",
        "union Pad49 { int i; char arr[9];    } pad49;",
        "union Pad87 { double d; char arr[7]; } pad87;",
        "union Pad88 { char arr[8]; double d; } pad88;",
        "union Pad89 { char arr[9]; double d; } pad89;",
        "union Mid { char c; double d; float f; } mid;",
        "union Recur { union Recur *p; } recur;",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMType padLLVMType = SrcUnionType.getPadLLVMType(ctxt);
    final LLVMType unionRecur = mod.getTypeByName("union.Recur");
    checkDecls(mod,
      new ExpectStruct[]{
        new ExpectStruct("union.Def", SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("union.Decl", (LLVMType[])null),
        new ExpectStruct("union.Ref", (LLVMType[])null),
        new ExpectStruct("union.Def.0", SrcFloatType.getLLVMType(ctxt)),
        new ExpectStruct("union.Pad41",
                         LLVMArrayType.get(SrcCharType.getLLVMType(ctxt), 1),
                         LLVMArrayType.get(padLLVMType, 3)),
        new ExpectStruct("union.Pad42", SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("union.Pad43",
                         LLVMArrayType.get(SrcCharType.getLLVMType(ctxt), 3),
                         LLVMArrayType.get(padLLVMType, 1)),
        new ExpectStruct("union.Pad44", SrcFloatType.getLLVMType(ctxt)),
        new ExpectStruct("union.Pad45",
                         SrcFloatType.getLLVMType(ctxt),
                         LLVMArrayType.get(padLLVMType, 4)),
        new ExpectStruct("union.Pad46",
                         LLVMArrayType.get(SrcCharType.getLLVMType(ctxt), 6),
                         LLVMArrayType.get(padLLVMType, 2)),
        new ExpectStruct("union.Pad47",
                         SrcIntType.getLLVMType(ctxt),
                         LLVMArrayType.get(padLLVMType, 4)),
        new ExpectStruct("union.Pad48",
                         LLVMArrayType.get(SrcCharType.getLLVMType(ctxt), 8)),
        new ExpectStruct("union.Pad49",
                         SrcIntType.getLLVMType(ctxt),
                         LLVMArrayType.get(padLLVMType, 8)),
        new ExpectStruct("union.Pad87", SrcDoubleType.getLLVMType(ctxt)),
        new ExpectStruct("union.Pad88",
                         LLVMArrayType.get(SrcCharType.getLLVMType(ctxt), 8)),
        new ExpectStruct("union.Pad89",
                         LLVMArrayType.get(SrcCharType.getLLVMType(ctxt), 9),
                         LLVMArrayType.get(padLLVMType, 7)),
        new ExpectStruct("union.Mid",
                         SrcCharType.getLLVMType(ctxt),
                         LLVMArrayType.get(padLLVMType, 7)),
        new ExpectStruct("union.Recur",
                         LLVMPointerType.get(unionRecur, 0)),
      });
    LLVMType Ref = simpleResult.llvmModule.getTypeByName("union.Ref");
    checkDecls(simpleResult.llvmModule,
      new Expect[]{
        new Expect("def", "union.Def"),
        new Expect("ref", LLVMPointerType.get(Ref, 0)),
        new Expect("fnDef", getFnTy(ctxt), new Expect("def", "union.Def.0")),
        new Expect("pad41", "union.Pad41"),
        new Expect("pad42", "union.Pad42"),
        new Expect("pad43", "union.Pad43"),
        new Expect("pad44", "union.Pad44"),
        new Expect("pad45", "union.Pad45"),
        new Expect("pad46", "union.Pad46"),
        new Expect("pad47", "union.Pad47"),
        new Expect("pad48", "union.Pad48"),
        new Expect("pad49", "union.Pad49"),
        new Expect("pad87", "union.Pad87"),
        new Expect("pad88", "union.Pad88"),
        new Expect("pad89", "union.Pad89"),
        new Expect("mid", "union.Mid"),
        new Expect("recur", "union.Recur"),
      });
  }

  /**
   * Check that specifying a different target data layout does influence union
   * computations.
   */
  @Test public void unionTargetDataLayout() throws IOException {
    final SimpleResult simpleResult = checkDecls(
      "i" + SrcIntType.getLLVMWidth() + ":64-f32:64",
      new String[]{
        "union Pad40 { float f;  }              pad40;",
        "union Pad41 { char arr[1]; float f;  } pad41;",
        "union Pad42 { int i; char arr[2];    } pad42;",
        "union Pad43 { char arr[3]; int i;    } pad43;",
        "union Pad44 { float f; char arr[4];  } pad44;",
        "union Pad45 { float f; char arr[5];  } pad45;",
        "union Pad46 { char arr[6]; float f;  } pad46;",
        "union Pad47 { int i; char arr[7];    } pad47;",
        "union Pad48 { char arr[8]; int i;    } pad48;",
        "union Pad49 { int i; char arr[9];    } pad49;",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMType padLLVMType = SrcUnionType.getPadLLVMType(ctxt);
    checkDecls(mod,
      new ExpectStruct[]{
        new ExpectStruct("union.Pad40", SrcFloatType.getLLVMType(ctxt),
                         LLVMArrayType.get(padLLVMType, 4)),
        new ExpectStruct("union.Pad41",
                         LLVMArrayType.get(SrcCharType.getLLVMType(ctxt), 1),
                         LLVMArrayType.get(padLLVMType, 7)),
        new ExpectStruct("union.Pad42", SrcIntType.getLLVMType(ctxt),
                         LLVMArrayType.get(padLLVMType, 4)),
        new ExpectStruct("union.Pad43",
                         LLVMArrayType.get(SrcCharType.getLLVMType(ctxt), 3),
                         LLVMArrayType.get(padLLVMType, 5)),
        new ExpectStruct("union.Pad44", SrcFloatType.getLLVMType(ctxt),
                         LLVMArrayType.get(padLLVMType, 4)),
        new ExpectStruct("union.Pad45", SrcFloatType.getLLVMType(ctxt),
                         LLVMArrayType.get(padLLVMType, 4)),
        new ExpectStruct("union.Pad46",
                         LLVMArrayType.get(SrcCharType.getLLVMType(ctxt), 6),
                         LLVMArrayType.get(padLLVMType, 2)),
        new ExpectStruct("union.Pad47", SrcIntType.getLLVMType(ctxt),
                         LLVMArrayType.get(padLLVMType, 4)),
        new ExpectStruct("union.Pad48",
                         LLVMArrayType.get(SrcCharType.getLLVMType(ctxt), 8)),
        new ExpectStruct("union.Pad49", SrcIntType.getLLVMType(ctxt),
                         LLVMArrayType.get(padLLVMType, 12)),
      });
    checkDecls(mod,
      new Expect[]{
        new Expect("pad40", "union.Pad40"),
        new Expect("pad41", "union.Pad41"),
        new Expect("pad42", "union.Pad42"),
        new Expect("pad43", "union.Pad43"),
        new Expect("pad44", "union.Pad44"),
        new Expect("pad45", "union.Pad45"),
        new Expect("pad46", "union.Pad46"),
        new Expect("pad47", "union.Pad47"),
        new Expect("pad48", "union.Pad48"),
        new Expect("pad49", "union.Pad49"),
      });
  }

  /**
   * Check that, as for tag-less structs, Cetus does add tags to tag-less
   * unions, but leave {@link #structTagless} to test this case
   * thoroughly.
   */
  @Test public void unionTagless() throws IOException {
    final SimpleResult simpleResult
      = checkDecls(new String[]{"union { int i; } x;"});
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    checkDecls(mod, new ExpectTaglessStruct("x", SrcIntType.getLLVMType(ctxt)));
  }

  @Test public void unionMemberIncompleteType() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "union member has incomplete type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "union U { struct S i; };",
      });
  }

  @Test public void unionMemberFlexibleArray() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "union member has incomplete type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "union U { int i; int a[]; };",
      });
  }

  /**
   * Use of members from anonymous structures/unions is checked in
   * {@link BuildLLVMTest_OtherExpressions#memberAccess} and
   * {@link BuildLLVMTest_OtherExpressions#offsetofOperator}.
   */
  @Test public void anonymousStructUnion() throws IOException {
    // In C11, an anonymous structure/union should have no tag, but Cetus
    // would give it a tag that's hard to predict, so we just go ahead and
    // give it a tag deterministically. gcc permits tags if -fms-extensions.
    final SimpleResult simpleResult = checkDecls(
      new String[]{
        "struct StructSimple {",
        "  union UnionAnonymous {",
        "    int i;",
        "    int j;",
        "  };",
        "  int k;",
        "} structSimple;",
        "union UnionSimple {",
        "  struct StructAnonymous {",
        "    int i;",
        "    int j;",
        "  };",
        "  int k;",
        "} unionSimple;",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    LLVMType unionAnonymous = mod.getTypeByName("union.UnionAnonymous");
    LLVMType structAnonymous = mod.getTypeByName("struct.StructAnonymous");
    checkDecls(mod,
      new ExpectStruct[]{
        new ExpectStruct("union.UnionAnonymous",
                         SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("struct.StructSimple", unionAnonymous,
                         SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("struct.StructAnonymous",
                         SrcIntType.getLLVMType(ctxt),
                         SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("union.UnionSimple", structAnonymous),
      });
  }

  @Test public void typedef() throws IOException {
    final SimpleResult simpleResult = checkDecls(
      new String[]{
        "typedef int I;",
        "typedef void *P;",
        "typedef void F(int);",
        // typedef name same as tag.
        // struct/union def in typedef.
        "typedef struct S { int i; } S;",
        // typedef name different than tag.
        "typedef union Opaque O;",
        // typedef in terms of typedef name.
        // array in terms of typedef name.
        // struct/union def outside typedef.
        "typedef S T[3];",

        "I ig;",
        "P pg;",
        "F fg;", // VariableDeclarator behaves like ProcedureDeclarator
        "S sg;",
        "O *og;", // pointer to typedef name
        "T tg;",

        "struct Members {",
        "  I im;",
        "  P pm;",
        "  F *fm;",
        "  S sm;",
        "  O *om;",
        "  T tm;",
        "} members;",

        "typedef float OutTInT;",
        "OutTInT outTInT;",
        "typedef float OutTInV;",
        "OutTInV outTInV;",

        // F and T as parameters are converted to pointers.
        // op cannot be incomplete when attached to function definition.
        "void fnDecl(I ip, P pp, F fp, S sp, O op, T tp);",
        "void fnDef(I ip, P pp, F fp, S sp, O *op, T tp) {",
        "  I il;",
        "  P pl;",
        "  F *fl;", // TODO: we don't yet handle local prototypes.
        "  S sl;",
        "  O *ol;",
        "  T tl;",

        // BUG0 exercises multiple typedefs in one declaration.
        // Also, BUG0 and BUG1 work around Cetus bugs that are exercised in
        // {@link #typedefShadow} but that have been fixed since these test
        // cases were originally written.
        "  typedef double BUG0, OutTInT;",
        "  OutTInT outTInT;",
        "  BUG0 bug0;",
        "  int BUG1, OutTInV;",

        "  typedef double InTOutT;",
        "  InTOutT inTOutT;",
        "  typedef float InTOutV;",
        "  InTOutV inTOutV;",
        "}",

        "OutTInT outTInTAfter;",
        "OutTInV outTInVAfter;",

        "typedef float InTOutT;",
        "InTOutT inTOutT;",
        "int InTOutV;",
      });
    final LLVMModule llvmModule = simpleResult.llvmModule;
    final LLVMContext ctxt = llvmModule.getContext();
    LLVMType I = SrcIntType.getLLVMType(ctxt);
    LLVMType P = LLVMPointerType.get(SrcVoidType.getLLVMTypeAsPointerTarget(ctxt), 0);
    LLVMType F = getFnTy(ctxt, SrcIntType.getLLVMType(ctxt));
    LLVMType Fp = LLVMPointerType.get(F, 0);
    LLVMType S = llvmModule.getTypeByName("struct.S");
    LLVMType O = llvmModule.getTypeByName("union.Opaque");
    LLVMType Op = LLVMPointerType.get(O, 0);
    LLVMType T = LLVMArrayType.get(S, 3);
    LLVMType Tep = LLVMPointerType.get(S, 0);
    checkDecls(llvmModule,
      new ExpectStruct[]{
        new ExpectStruct("struct.S", SrcIntType.getLLVMType(ctxt)),
        new ExpectStruct("union.Opaque", (LLVMType[])null),
        new ExpectStruct("struct.Members", I, P, Fp, S, Op, T),
      });
    checkDecls(llvmModule,
      new Expect[]{
        new Expect("ig", I),
        new Expect("pg", P),
        new Expect("fg", F),
        new Expect("sg", S),
        new Expect("og", Op),
        new Expect("tg", T),
        new Expect("members", "struct.Members"),
        new Expect("outTInT", SrcFloatType.getLLVMType(ctxt)),
        new Expect("outTInV", SrcFloatType.getLLVMType(ctxt)),
        new Expect("fnDecl", getFnTy(ctxt, I, P, Fp, S, O, Tep)),
        new Expect("fnDef", getFnTy(ctxt, I, P, Fp, S, Op, Tep),
          new Expect("il", I),
          new Expect("pl", P),
          new Expect("fl", Fp),
          new Expect("sl", S),
          new Expect("ol", Op),
          new Expect("tl", T),

          new Expect("outTInT", SrcDoubleType.getLLVMType(ctxt)),
          new Expect("bug0", SrcDoubleType.getLLVMType(ctxt)),
          new Expect("OutTInV", SrcIntType.getLLVMType(ctxt)),
          new Expect("BUG1", SrcIntType.getLLVMType(ctxt)),

          new Expect("inTOutT", SrcDoubleType.getLLVMType(ctxt)),
          new Expect("inTOutV", SrcFloatType.getLLVMType(ctxt))
        ),
        new Expect("outTInTAfter", SrcFloatType.getLLVMType(ctxt)),
        new Expect("outTInVAfter", SrcFloatType.getLLVMType(ctxt)),
        new Expect("inTOutT", SrcFloatType.getLLVMType(ctxt)),
        new Expect("InTOutV", SrcIntType.getLLVMType(ctxt)),
      });
  }

  /**
   * See {@link #structTagless} for why this is tough to test.
   */
  @Test public void typedefTaglessStructOrUnion() throws IOException {
    final SimpleResult simpleResult
      = checkDecls(
          new String[]{
            "typedef struct { int i; } S; S s;",
            "typedef union { int i; } U; U u;",
          });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    checkDecls(mod,
               new ExpectTaglessStruct("s", SrcIntType.getLLVMType(ctxt)),
               new ExpectTaglessStruct("u", SrcIntType.getLLVMType(ctxt)));
  }

  /**
   * Due to a parsing bug (since fixed), Cetus used to mishandle some
   * declarations that attempt to shadow typedef names. It didn't understand
   * that, after a type specifier has already appeared in a declaration, any
   * additional identifier should be parsed as a declarator even if it was
   * previously defined as a typedef name. In that case, Cetus quietly omitted
   * the declaration altogether from the tree. A workaround was to add a dummy
   * declarator name and then a comma before the desired declarator, as shown
   * in the {@code BUG0} and {@code BUG1} examples in {@link #typedef}. Adding
   * an initializer also worked in the case of a variable.
   */
  // Originally failed as:
  // @XFail(exception=AssertionError.class,
  //        message="fn.outTInT's type expected:<double*> but was:<float*>")
  @Test public void typedefShadow() throws IOException {
    final SimpleResult simpleResult = checkDecls(
      new String[]{
        "typedef float OutTInT;",
        "OutTInT outTInT;",
        "typedef float OutTInV;",
        "OutTInV outTInV;",
        "void fn() {",
        "  typedef double OutTInT;", // bug: omitted from tree
        "  OutTInT outTInT;", // bug: Cetus thinks OutTInT is still float
        "  int OutTInV;", // bug: omitted from tree
        "}",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    checkDecls(mod,
      new Expect[]{
        new Expect("outTInT", SrcFloatType.getLLVMType(ctxt)),
        new Expect("outTInV", SrcFloatType.getLLVMType(ctxt)),
        new Expect("fn", getFnTy(ctxt),
          new Expect("outTInT", SrcDoubleType.getLLVMType(ctxt)),
          new Expect("OutTInV", SrcIntType.getLLVMType(ctxt))
        ),
      });
  }

  /**
   * Previously, if a typedef name was used in a scope, declared later in the
   * same scope, but also declared earlier in an enclosing scope, BuildLLVM
   * would fail an assertion when trying to resolve the typedef name at the
   * use. This check ensures that it resolves to the enclosing scope by
   * providing an initializer that's compatible with the earlier typedef name
   * but not the later typedef name.
   */
  @Test public void typedefWrongScope() throws IOException {
    buildLLVMSimple(
      "", "",
      new String[]{
        "struct S1 { int i; } s1;",
        "struct S2 { int i; };",
        "typedef struct S1 T;",
        "void fn() {",
        "  T t = s1;",
        "  typedef struct S2 T;",
        "}",
      });
  }

  @Test public void enumeration() throws IOException {
    final SimpleResult simpleResult = checkDecls(
      new String[]{
        // Exercise various numbers of enumerators and various numbers of
        // variables declared inside and outside an enumeration declaration.
        "enum E0 {E0I};",
        "enum E0 e0a = E0I;",
        "enum E1 {E1I, E1J} e1a;",
        "enum E1 e1b = E1I, e1c = E1J;",
        "enum E2 {E2I, E2J, E2K} e2a = E2I, e2b = E2J;",
        "enum E2 e2c = E2K, e2d = E2I, e2e = E2J;",

        // Tag-less enumeration.
        "enum {TLI, TLJ} tla = TLI, tlb = TLJ;",

        // Exercise various enumerator initializers.
        "enum I0 {I0A = 2, I0B,      I0C    } i0a = I0A, i0b = I0B, i0c = I0C;",
        "enum I1 {I1A,     I1B = -1, I1C    } i1a = I1A, i1b = I1B, i1c = I1C;",
        "enum I2 {I2A = 5, I2B = 9,  I2C    } i2a = I2A, i2b = I2B, i2c = I2C;",
        "enum I3 {I3A,     I3B = 8,  I3C = 8} i3a = I3A, i3b = I3B, i3c = I3C;",

        // Nested enumerations.
        "enum N { NA = sizeof(enum {NB}), NC } na = NA, nb = NB, nc = NC;",

        // enum typedef.
        "typedef enum I0 ET0;",
        "ET0 et0 = I0A;",
        "typedef enum {ET1A = -3} ET1;",
        "ET1 et1a = ET1A;",

        // Enumeration scopes.
        "enum DefDef {DefDefA=0, DefDefB} defDefA0=DefDefA, defDefB0=DefDefB;",
        "enum DefRef {DefRefA=5, DefRefB} defRefA0=DefRefA, defRefB0=DefRefB;",
        "void fn() {",
        "  enum DefDef {DefDefA=10, DefDefB} defDefA0=DefDefA, defDefB0=DefDefB;",
        "  enum DefRef defRefA=DefRefA, defRefB=DefRefB;",
        "}",
        "enum DefDef defDefA1=DefDefA, defDefB1=DefDefB;",
        "enum DefRef defRefA1=DefRefA, defRefB1=DefRefB;",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMIntegerType ENUM_TYPE_LLVM_TYPE
      = SrcEnumType.COMPATIBLE_INTEGER_TYPE.getLLVMType(ctxt);
    checkDecls(mod,
      new Expect[]{
        new Expect("e0a", ENUM_TYPE_LLVM_TYPE, 0),

        new Expect("e1a", ENUM_TYPE_LLVM_TYPE, 0),
        new Expect("e1b", ENUM_TYPE_LLVM_TYPE, 0),
        new Expect("e1c", ENUM_TYPE_LLVM_TYPE, 1),

        new Expect("e2a", ENUM_TYPE_LLVM_TYPE, 0),
        new Expect("e2b", ENUM_TYPE_LLVM_TYPE, 1),
        new Expect("e2c", ENUM_TYPE_LLVM_TYPE, 2),
        new Expect("e2d", ENUM_TYPE_LLVM_TYPE, 0),
        new Expect("e2e", ENUM_TYPE_LLVM_TYPE, 1),

        new Expect("tla", ENUM_TYPE_LLVM_TYPE, 0),
        new Expect("tlb", ENUM_TYPE_LLVM_TYPE, 1),

        new Expect("i0a", ENUM_TYPE_LLVM_TYPE, 2),
        new Expect("i0b", ENUM_TYPE_LLVM_TYPE, 3),
        new Expect("i0c", ENUM_TYPE_LLVM_TYPE, 4),

        new Expect("i1a", ENUM_TYPE_LLVM_TYPE, 0),
        new Expect("i1b", ENUM_TYPE_LLVM_TYPE, -1),
        new Expect("i1c", ENUM_TYPE_LLVM_TYPE, 0),

        new Expect("i2a", ENUM_TYPE_LLVM_TYPE, 5),
        new Expect("i2b", ENUM_TYPE_LLVM_TYPE, 9),
        new Expect("i2c", ENUM_TYPE_LLVM_TYPE, 10),

        new Expect("i3a", ENUM_TYPE_LLVM_TYPE, 0),
        new Expect("i3b", ENUM_TYPE_LLVM_TYPE, 8),
        new Expect("i3c", ENUM_TYPE_LLVM_TYPE, 8),

        new Expect("na", ENUM_TYPE_LLVM_TYPE,
                         (int)ENUM_TYPE_LLVM_TYPE.getWidth()/8),
        new Expect("nb", ENUM_TYPE_LLVM_TYPE, 0),
        new Expect("nc", ENUM_TYPE_LLVM_TYPE, 5),

        new Expect("et0", ENUM_TYPE_LLVM_TYPE, 2),
        new Expect("et1a", ENUM_TYPE_LLVM_TYPE, -3),

        new Expect("defDefA0", ENUM_TYPE_LLVM_TYPE, 0),
        new Expect("defDefB0", ENUM_TYPE_LLVM_TYPE, 1),
        new Expect("defRefA0", ENUM_TYPE_LLVM_TYPE, 5),
        new Expect("defRefB0", ENUM_TYPE_LLVM_TYPE, 6),
        new Expect("fn", getFnTy(ctxt),
                    new Expect("defDefA0", ENUM_TYPE_LLVM_TYPE, 10),
                    new Expect("defDefB0", ENUM_TYPE_LLVM_TYPE, 11),
                    new Expect("defRefA", ENUM_TYPE_LLVM_TYPE, 5),
                    new Expect("defRefB", ENUM_TYPE_LLVM_TYPE, 6)),
        new Expect("defDefA1", ENUM_TYPE_LLVM_TYPE, 0),
        new Expect("defDefB1", ENUM_TYPE_LLVM_TYPE, 1),
        new Expect("defRefA1", ENUM_TYPE_LLVM_TYPE, 5),
        new Expect("defRefB1", ENUM_TYPE_LLVM_TYPE, 6),
      });
  }

  /**
   * TODO: Due to a Cetus bug, {@link BuildLLVM} currently resolves the scope
   * of an enum declaration in a parameter list incorrectly as the enclosing
   * scope. This test checks that {@link BuildLLVM} is consistent in this
   * regard, but ultimately Cetus's behavior needs to be fixed. Once it is
   * fixed, the current {@link BuildLLVM} implementation should not need to
   * change as it just relies on the symbol tables constructed by Cetus in
   * order to resolve an enum's scope. See
   * {@link BuildLLVM.Visitor#lookupSymbolTable} for further discussion.
   */
  @Test public void enumerationParam() throws IOException {
    final SimpleResult simpleResult = checkDecls(
      new String[]{
        "void fnDef(enum E {I=5} e) {",
        "  enum E i = I;",
        "}",
        "enum E i = I;",

        "void fnDecl(enum F {J=6} f);",
        "enum F j = J;",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMIntegerType ENUM_TYPE_LLVM_TYPE
      = SrcEnumType.COMPATIBLE_INTEGER_TYPE.getLLVMType(ctxt);
    checkDecls(mod,
      new Expect[]{
        new Expect("fnDef", getFnTy(ctxt, ENUM_TYPE_LLVM_TYPE),
                    new Expect("i", ENUM_TYPE_LLVM_TYPE, 5)),
        new Expect("i", ENUM_TYPE_LLVM_TYPE, 5),

        new Expect("fnDecl", getFnTy(ctxt, ENUM_TYPE_LLVM_TYPE)),
        new Expect("j", ENUM_TYPE_LLVM_TYPE, 6),
      });
  }

  /**
   * Previously, if an enum tag was used in a scope, defined later in the
   * same scope, but also defined earlier in an enclosing scope, BuildLLVM
   * would fail an assertion when trying to resolve the enum tag at the use.
   * This check ensures that it resolves to the enclosing scope by making sure
   * the resolved type isn't compatible with the later enum definition.
   * Currently, I see no other way to check the resolution.
   */
  @Test public void enumerationWrongScope() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "warning treated as error:"
      + " initialization requires conversion between pointer types with"
      + " incompatible target types without explicit cast: enum types are"
      + " incompatible because they are not the same enum: enum E"
      + " {A=<i32<0>>} and enum E {A=<i32<0>>}"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "enum E {A};",
        "void fn() {",
        "  enum E *e1;",
        "  enum E {A} *e2 = e1;",
        "}",
      });
  }

  @Test public void functionCompositeType() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = checkDecls(
      new String[]{
        "#include <stddef.h>",

        // -129 so that the compatible integer type is likely int. At the time
        // this was written, BuildLLVM always chose int, but other compilers
        // (clang 3.5.1) sometimes choose unsigned when possible, so BuildLLVM
        // may do so too some day.
        "enum E {A = -129};",

        // param type list unspecified always
        "void v0_v0();",
        "void v0_v0();",
        "int i0_e0();",
        "enum E i0_e0();",
        "enum E e0_i0();",
        "int e0_i0();",
        "int (*paui0_pa5e0())[];",
        "enum E (*paui0_pa5e0())[5];",
        "int (*pa5i0_paue0())[5];",
        "enum E (*pa5i0_paue0())[];",

        // param type list unspecified sometimes
        "void v0_vv();",
        "void v0_vv(void);",
        "void vv_v0(void);",
        "void vv_v0();",
        "int (*pa3iv_paue0(void))[3];",
        "enum E (*pa3iv_paue0())[];",
        // each param type must be compatible with itself after default
        // argument promotions, and the function cannot be variadic
        "int ii_e0(int i);",
        "enum E ii_e0();",
        "enum E e0_id();",
        "int e0_id(double);",
        "void v0_vldli();",
        "void v0_vldli(long double, long int);",

        // param type list specified multiple times
        "void vv_vv(void);",
        "void vv_vv(void);",
        "int (*pauiv_pa9ev(void))[];",
        "enum E (*pauiv_pa9ev(void))[9];",
        "enum E ei_ii(int);",
        "int ei_ii(int);",
        "void vi_ve(int);",
        "void vi_ve(enum E);",
        "void vpaue_vpa2i(enum E (*)[]);",
        "void vpaue_vpa2i(int (*)[2]);",
        "void vpa2i_vpaue(int (*)[2]);",
        "void vpa2i_vpaue(enum E (*)[]);",
        "void viv_vev(int, ...);",
        "void viv_vev(enum E, ...);",
        "void vci_vce(char, int);",
        "void vci_vce(char, enum E);",
        "void vipvv_vepvv(int, void *, ...);",
        "void vipvv_vepvv(enum E, void *, ...);",
        "void vepvfn0_vipvfni(enum E, void());",
        "void vepvfn0_vipvfni(int, void(int));",

        // change from composite doesn't break fn def accessing ret type or
        // param type

        // return type changes at definition
        "size_t size;",
        "size_t getSize() { return size; }",
        "char (*defRet(void))[];",
        "char (*defRet(void))[5] {",
        "  size_t s = sizeof *defRet();",
        "  size = s;",
        "  return (char(*)[5])0;",
        "}",
        "char (*defRet(void))[];",

        // param type changes at definition
        "size_t defParam(char (*)[]);",
        "size_t defParam(char (*p)[3]) { return sizeof *p; }",
        "size_t defParam(char (*)[]);",
        "size_t call_defParam() { return defParam(0); }",
        
        // return and param type change after definition
        "int a9i[9];",
        "int (*get_a9i())[9] { return &a9i; }",
        "int (*defRetParam(int (*p)[]))[] { return p; }",
        "int (*defRetParam(int (*p)[9]))[9];",
        "int (*call_defRetParam())[9] { return defRetParam(&a9i); }",
        "size_t sizeof_defRetParam() { return sizeof *defRetParam(0); }",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    checkDecls(mod,
      new Expect[]{
        new Expect("v0_v0", getFnTy(SrcVoidType.getLLVMType(ctxt), true)),
        new Expect("i0_e0", getFnTy(SrcIntType.getLLVMType(ctxt), true)),
        new Expect("e0_i0", getFnTy(SrcIntType.getLLVMType(ctxt), true)),
        new Expect(
          "paui0_pa5e0",
          getFnTy(LLVMPointerType.get(
                    LLVMArrayType.get(SrcIntType.getLLVMType(ctxt), 5), 0),
                  true)),
        new Expect(
          "pa5i0_paue0",
          getFnTy(LLVMPointerType.get(
                    LLVMArrayType.get(SrcIntType.getLLVMType(ctxt), 5), 0),
                  true)),
        new Expect("v0_vv", getFnTy(SrcVoidType.getLLVMType(ctxt), false)),
        new Expect("vv_v0", getFnTy(SrcVoidType.getLLVMType(ctxt), false)),
        new Expect(
          "pa3iv_paue0",
          getFnTy(LLVMPointerType.get(
                    LLVMArrayType.get(SrcIntType.getLLVMType(ctxt), 3), 0),
                  false)),
        new Expect("ii_e0", getFnTy(SrcIntType.getLLVMType(ctxt), false,
                                    SrcIntType.getLLVMType(ctxt))),
        new Expect("e0_id", getFnTy(SrcIntType.getLLVMType(ctxt), false,
                                    SrcDoubleType.getLLVMType(ctxt))),
        new Expect("v0_vldli", getFnTy(SrcVoidType.getLLVMType(ctxt), false,
                                       SrcLongDoubleType.getLLVMType(ctxt),
                                       SrcLongType.getLLVMType(ctxt))),
        new Expect("vv_vv", getFnTy(SrcVoidType.getLLVMType(ctxt), false)),
        new Expect(
          "pauiv_pa9ev",
          getFnTy(LLVMPointerType.get(
                    LLVMArrayType.get(SrcIntType.getLLVMType(ctxt), 9), 0),
                  false)),
        new Expect("ei_ii", getFnTy(SrcIntType.getLLVMType(ctxt), false,
                                    SrcIntType.getLLVMType(ctxt))),
        new Expect("vi_ve", getFnTy(SrcVoidType.getLLVMType(ctxt), false,
                                    SrcIntType.getLLVMType(ctxt))),
        new Expect(
          "vpaue_vpa2i",
          getFnTy(SrcVoidType.getLLVMType(ctxt), false,
                  LLVMPointerType.get(
                    LLVMArrayType.get(SrcIntType.getLLVMType(ctxt), 2), 0))),
        new Expect(
          "vpa2i_vpaue",
          getFnTy(SrcVoidType.getLLVMType(ctxt), false,
                  LLVMPointerType.get(
                    LLVMArrayType.get(SrcIntType.getLLVMType(ctxt), 2), 0))),
        new Expect("viv_vev", getFnTy(SrcVoidType.getLLVMType(ctxt), true,
                                      SrcIntType.getLLVMType(ctxt))),
        new Expect("vci_vce", getFnTy(SrcVoidType.getLLVMType(ctxt), false,
                                      SrcCharType.getLLVMType(ctxt),
                                      SrcIntType.getLLVMType(ctxt))),
        new Expect(
          "vipvv_vepvv",
          getFnTy(SrcVoidType.getLLVMType(ctxt), true,
                  SrcIntType.getLLVMType(ctxt),
                  LLVMPointerType.get(
                    SrcVoidType.getLLVMTypeAsPointerTarget(ctxt), 0))),
        new Expect(
          "vepvfn0_vipvfni",
          getFnTy(SrcVoidType.getLLVMType(ctxt), false,
                  SrcIntType.getLLVMType(ctxt),
                  LLVMPointerType.get(
                    getFnTy(SrcVoidType.getLLVMType(ctxt), false,
                            SrcIntType.getLLVMType(ctxt)),
                    0))),

        new Expect("size", SRC_SIZE_T_TYPE.getLLVMType(ctxt)),
        new Expect("getSize",
                   getFnTy(SRC_SIZE_T_TYPE.getLLVMType(ctxt), false)),
        new Expect(
          "defRet",
          getFnTy(LLVMPointerType.get(
                    LLVMArrayType.get(SrcCharType.getLLVMType(ctxt), 5),
                    0),
                  false)),
        new Expect(
          "defParam",
          getFnTy(SRC_SIZE_T_TYPE.getLLVMType(ctxt), false,
                  LLVMPointerType.get(
                    LLVMArrayType.get(SrcCharType.getLLVMType(ctxt), 3),
                    0))),
        new Expect("call_defParam", getFnTy(SRC_SIZE_T_TYPE.getLLVMType(ctxt),
                                            false)),
        new Expect("a9i", LLVMArrayType.get(SrcIntType.getLLVMType(ctxt),
                                                9)),
        new Expect(
          "get_a9i",
          getFnTy(LLVMPointerType.get(
                    LLVMArrayType.get(SrcIntType.getLLVMType(ctxt), 9),
                    0),
                  false)),
        new Expect(
          "defRetParam",
          getFnTy(LLVMPointerType.get(
                    LLVMArrayType.get(SrcIntType.getLLVMType(ctxt), 9),
                    0),
                  false,
                  LLVMPointerType.get(
                    LLVMArrayType.get(SrcIntType.getLLVMType(ctxt), 9),
                    0))),
        new Expect(
          "call_defRetParam",
          getFnTy(LLVMPointerType.get(
                    LLVMArrayType.get(SrcIntType.getLLVMType(ctxt), 9),
                    0),
                  false)),
        new Expect("sizeof_defRetParam",
                   getFnTy(SRC_SIZE_T_TYPE.getLLVMType(ctxt), false)),
      });
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    runFn(exec, mod, "defRet");
    checkIntFn(exec, mod, "getSize", 5*SrcCharType.getLLVMWidth()/8);
    checkIntFn(exec, mod, "call_defParam", 3*SrcCharType.getLLVMWidth()/8);
    final LLVMGenericValue a9i = runFn(exec, mod, "get_a9i");
    checkPointerFn(exec, mod, "call_defRetParam", a9i);
    checkIntFn(exec, mod, "sizeof_defRetParam",
               9*SrcIntType.getLLVMWidth()/8);
    exec.dispose();
  }

  @Test public void functionCompositeTypeLocal() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final String[] srcLines
      = new String[]{
        "#include <stddef.h>",

        "size_t localOuter1Size, localInnerSize, localOuter2Size;",
        "size_t get_localOuter1Size() { return localOuter1Size; }",
        "size_t get_localInnerSize() { return localInnerSize; }",
        "size_t get_localOuter2Size() { return localOuter2Size; }",
        "int a9i[9];",

        "int (*fn())[];",
        "void caller1() {",
        "  int (*global)[8] = fn(&a9i);", // param/ret size unspecified, so compatible

        "  int (*fn(int (*p)[]))[2];",
        "  fn(&a9i);", // param size unspecified, so compatible
        "  size_t localOuter1Size_ = sizeof *fn(0);", // ret size specified
        "  localOuter1Size = localOuter1Size_;",

        "  {",
        "    int (*fn(int (*p)[3]))[];",
        "//INCOMPATIBLE:    fn(&a9i);", // param size specified, so incompatible
        "    size_t localInnerSize_ = sizeof *fn(0);", // ret size specified
        "    localInnerSize = localInnerSize_;",
        "  }",

        "  fn(&a9i);", // param size unspecified, so compatible
        "  size_t localOuter2Size_ = sizeof *fn(0);", // ret size specified
        "  localOuter2Size = localOuter2Size_;",
        "}",

        "void caller2() {",
        "  int (*global)[8] = fn(&a9i);", // param/ret size unspecified, so compatible
        "}",

        "int (*fn(void *))[] { return (int(*)[])0; }",

        // Block-scope function declarations must shadow other block-scope
        // symbols.
        "#define PI 3.14159",
        "double sin(double);",
        "double localShadow() {",
        "  int sin;",
        "  int cos;",
        "  {",
        "    double sin(double);", // prior file-scope and block-scope decls
        "    double cos(double);", // prior block-scope decl only
        "    return sin(PI/2) + cos(0);",
        "  }",
        "}",
      };

    // Make sure it's fine with warnings treated as errors.
    final File file = writeTmpFile("-clean.c", srcLines);
    final BuildLLVM buildLLVM = buildLLVM("", "", true, file);
    final LLVMModule mod = buildLLVM.getLLVMModules()[0];
    final LLVMContext ctxt = mod.getContext();
    final LLVMIntegerType sizeTLLVMType = SRC_SIZE_T_TYPE.getLLVMType(ctxt);
    final LLVMIntegerType intLLVMType = SrcIntType.getLLVMType(ctxt);
    final LLVMRealType doubleLLVMType = SrcDoubleType.getLLVMType(ctxt);
    final LLVMPointerType ptrVoidLLVMType
      = LLVMPointerType.get(SrcVoidType.getLLVMTypeAsPointerTarget(ctxt), 0);
    checkDecls(mod,
      new Expect[]{
        new Expect("localOuter1Size", sizeTLLVMType),
        new Expect("localInnerSize", sizeTLLVMType),
        new Expect("localOuter2Size", sizeTLLVMType),
        new Expect("get_localOuter1Size", getFnTy(sizeTLLVMType, false)),
        new Expect("get_localInnerSize", getFnTy(sizeTLLVMType, false)),
        new Expect("get_localOuter2Size", getFnTy(sizeTLLVMType, false)),
        new Expect("a9i", LLVMArrayType.get(intLLVMType, 9)),
        new Expect(
          "fn",
          getFnTy(LLVMPointerType.get(LLVMArrayType.get(intLLVMType, 0), 0),
                  false, ptrVoidLLVMType)),
        new Expect("caller1", getFnTy(ctxt)),
        new Expect("caller2", getFnTy(ctxt)),
        new Expect("sin", getFnTy(doubleLLVMType, false, doubleLLVMType)),
        new Expect("localShadow", getFnTy(doubleLLVMType, false)),
        new Expect("cos", getFnTy(doubleLLVMType, false, doubleLLVMType)),
      });
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    runFn(exec, mod, "caller1");
    checkIntFn(exec, mod, "get_localOuter1Size",
               2*SrcIntType.getLLVMWidth()/8);
    checkIntFn(exec, mod, "get_localInnerSize",
               2*SrcIntType.getLLVMWidth()/8);
    checkIntFn(exec, mod, "get_localOuter2Size",
               2*SrcIntType.getLLVMWidth()/8);
    checkDoubleFn(exec, mod, "localShadow", 2);
    exec.dispose();

    // Uncomment the pointer incompatibility, and try again with warnings
    // not treated as errors.
    for (int i = 0; i < srcLines.length; ++i)
      srcLines[i] = srcLines[i].replaceFirst("^//INCOMPATIBLE:", "");
    buildLLVM("", "", false, writeTmpFile("-warning.c", srcLines));

    // Finally, treat the pointer incompatibility as an error.
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "warning treated as error:"
      + " argument 1 to \"fn\" requires conversion between pointer types"
      + " with incompatible target types without explicit cast: array types"
      + " are incompatible because they have different sizes"));
    buildLLVM("", "", true, writeTmpFile("-error.c", srcLines));
  }

  /**
   * {@link #functionCompositeType} exercises various kinds of composite
   * type computations. Here we focus on just file-scope variable issues.
   */
  @Test public void varCompositeType() throws IOException {
    final SimpleResult simpleResult = checkDecls(
      new String[]{
        "#include <stddef.h>",

        "int arrSize[];",
        "int arrSize[3];",

        // The second declaration of initBitCast here alters the type,
        // requiring the old initializer to be bitcast to the new type. Without
        // that bitcast, LLVM complains about a type mismatch. Computing the
        // size of initBitCast requires the second declaration to actually
        // alter the type.
        "int (*initBitCast)[] = &arrSize;",
        "int (*initBitCast)[3];",
        "size_t initBitCastSize = sizeof *initBitCast;", // requires second type

        // The composite type must be computed before the initialization. If
        // it were the other way around, these types would be incompatible
        // because they would have different sizes.
        "int compositeBeforeInit[3];",
        "int compositeBeforeInit[] = {1, 2};", // 3 elements, last is zero
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMGlobalVariable arrSize = mod.getNamedGlobal("arrSize");
    final LLVMIntegerType intLLVMType = SrcIntType.getLLVMType(ctxt);
    checkDecls(mod,
      new Expect[]{
        new Expect("arrSize",
                   LLVMArrayType.get(intLLVMType, 3)),
        new Expect("initBitCast",
                   LLVMPointerType.get(LLVMArrayType.get(intLLVMType, 3), 0),
                   arrSize),
        new Expect("initBitCastSize", SRC_SIZE_T_TYPE.getLLVMType(ctxt),
                   LLVMConstantInteger.get(SRC_SIZE_T_TYPE.getLLVMType(ctxt),
                                           3*SrcIntType.getLLVMWidth()/8,
                                           false)),
        new Expect("compositeBeforeInit",
                   LLVMArrayType.get(intLLVMType, 3),
                   LLVMConstantArray.get(intLLVMType,
                     LLVMConstantInteger.get(intLLVMType, 1, true),
                     LLVMConstantInteger.get(intLLVMType, 2, true),
                     LLVMConstantInteger.get(intLLVMType, 0, true))),
      });
  }

  @Test public void varCompositeTypeLocal() throws Exception {
    LLVMTargetData.initializeNativeTarget();
    final SimpleResult simpleResult = checkDecls(
      new String[]{
        "#include <stddef.h>",
        "size_t getPtrSize() { return sizeof(void*); }",

        "size_t localOuter1Size;",
        "size_t localInnerSize1, localInnerSize2;",
        "size_t localOuter2Size;",
        "size_t get_localOuter1Size() { return localOuter1Size; }",
        "size_t get_localInnerSize1() { return localInnerSize1; }",
        "size_t get_localInnerSize2() { return localInnerSize2; }",
        "size_t get_localOuter2Size() { return localOuter2Size; }",

        "int (*(*var)[])[];",
        "void user1() {",
        "  int (*(*global)[8])[9] = var;", // both sizes unspecified, so compatible

        "  extern int (*(*var)[2])[];",
        "  int (*(*localOuter1)[2])[9] = var;", // second size unspecified, so compatible
        "  size_t localOuter1Size_ = sizeof *var;", // first size specified
        "  localOuter1Size = localOuter1Size_;",

        "  {",
        "    extern int (*(*var)[])[3];",
        "    size_t localInnerSize1_ = sizeof *var;", // first size specified
        "    localInnerSize1 = localInnerSize1_;",
        "    size_t localInnerSize2_ = sizeof ***var;", // second size specified
        "    localInnerSize2 = localInnerSize2_;",
        "  }",

        "  int (*(*localOuter2)[2])[9] = var;", // second size unspecified, so compatible
        "  size_t localOuter2Size_ = sizeof *var;", // first size specified
        "  localOuter2Size = localOuter2Size_;",
        "}",

        "void user2() {",
        "  int (*(*global)[8])[9] = var;", // both sizes unspecified, so compatible
        "}",

        // Block-scope extern declarations must shadow other block-scope
        // symbols.
        "int localShadow_i = 5;",
        "int localShadow() {",
        "  struct S {int i;} localShadow_i;",
        "  struct S localShadow_j;",
        "  {",
        "    extern int localShadow_i;", // prior file-scope and block-scope decls
        "    extern int localShadow_j;", // prior block-scope decl only
        "    return localShadow_i + localShadow_j;",
        "  }",
        "}",
        "int localShadow_j = 6;",
      });
    final LLVMModule mod = simpleResult.llvmModule;
    final LLVMContext ctxt = mod.getContext();
    final LLVMIntegerType sizeTLLVMType = SRC_SIZE_T_TYPE.getLLVMType(ctxt);
    final LLVMIntegerType intLLVMType = SrcIntType.getLLVMType(ctxt);
    checkDecls(mod,
      new Expect[]{
        new Expect("getPtrSize", getFnTy(sizeTLLVMType, false)),
        new Expect("localOuter1Size", sizeTLLVMType),
        new Expect("localInnerSize1", sizeTLLVMType),
        new Expect("localInnerSize2", sizeTLLVMType),
        new Expect("localOuter2Size", sizeTLLVMType),
        new Expect("get_localOuter1Size", getFnTy(sizeTLLVMType, false)),
        new Expect("get_localInnerSize1", getFnTy(sizeTLLVMType, false)),
        new Expect("get_localInnerSize2", getFnTy(sizeTLLVMType, false)),
        new Expect("get_localOuter2Size", getFnTy(sizeTLLVMType, false)),
        new Expect(
          "var",
          LLVMPointerType.get(
            LLVMArrayType.get(
              LLVMPointerType.get(LLVMArrayType.get(intLLVMType, 0), 0),
              0),
            0)),
        new Expect("user1", getFnTy(ctxt)),
        new Expect("user2", getFnTy(ctxt)),
        new Expect("localShadow_i", SrcIntType.getLLVMType(ctxt),
                    getConstInt(5, ctxt)),
        new Expect("localShadow_j", SrcIntType.getLLVMType(ctxt),
                    getConstInt(6, ctxt)),
        new Expect("localShadow", getFnTy(SrcIntType.getLLVMType(ctxt),
                                          false)),
      });
    final LLVMExecutionEngine exec = new LLVMExecutionEngine(mod);
    runFn(exec, mod, "user1");
    final long ptrSize
      = runFn(exec, mod, "getPtrSize").toInt(false).longValue();
    checkIntFn(exec, mod, "get_localOuter1Size", 2*ptrSize);
    checkIntFn(exec, mod, "get_localInnerSize1", 2*ptrSize);
    checkIntFn(exec, mod, "get_localInnerSize2", 3*SrcIntType.getLLVMWidth()/8);
    checkIntFn(exec, mod, "get_localOuter2Size", 2*ptrSize);
    checkIntFn(exec, mod, "localShadow", 11);
    exec.dispose();
  }

  @Test public void localVarVoidType() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "variable \"foo\" has void type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "void fn() {",
        "  void foo = 5;",
        "}",
      });
  }

  @Test public void localVarIncompleteType() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "variable \"foo\" has incomplete type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "void fn() {",
        "  struct T foo;",
        "}",
      });
  }

  @Test public void localStaticVarVoidType() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "variable \"t\" has void type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "void fn() {",
        "  static void t = 5;;",
        "}",
      });
  }

  @Test public void localStaticVarIncompleteType() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "variable \"t\" has incomplete type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "void fn() {",
        "  static struct T t;",
        "}",
      });
  }

  @Test public void globalVarIncompleteType() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "variable \"t\" has incomplete type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "union T t;",
      });
  }
}