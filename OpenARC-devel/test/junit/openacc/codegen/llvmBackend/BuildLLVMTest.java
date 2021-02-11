package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcPrimitiveFloatingType.SrcDoubleType;
import static openacc.codegen.llvmBackend.SrcPrimitiveFloatingType.SrcFloatType;
import static openacc.codegen.llvmBackend.SrcPrimitiveFloatingType.SrcLongDoubleType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SRC_SIZE_T_TYPE;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcBoolType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcCharType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedCharType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcLongType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcSignedCharType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcIntType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedIntType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcUnsignedLongType;
import static openacc.codegen.llvmBackend.SrcPrimitiveNonNumericType.SrcVoidType;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.File;
import java.io.IOException;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;

import openacc.exec.ACC2GPUDriver;
import openacc.test.JUnitTest;

import org.jllvm.LLVMConstant;
import org.jllvm.LLVMConstantInteger;
import org.jllvm.LLVMConstantReal;
import org.jllvm.LLVMContext;
import org.jllvm.LLVMExecutionEngine;
import org.jllvm.LLVMFunction;
import org.jllvm.LLVMGenericInt;
import org.jllvm.LLVMGenericReal;
import org.jllvm.LLVMGenericValue;
import org.jllvm.LLVMGlobalVariable;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMPointerType;
import org.jllvm.bindings.SWIGTYPE_p_void;

/**
 * Super class for all JUnit test cases for the {@link BuildLLVM} pass.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public abstract class BuildLLVMTest extends JUnitTest {
  /**
   * We want an ACC2GPUDriver, but we need access to its constructor, which
   * is protected.  This gives us a public constructor.
   */
  protected static class Driver extends ACC2GPUDriver {}

  public static class SimpleResult {
    public SimpleResult(BuildLLVM buildLLVM, LLVMModule llvmModule) {
      this.buildLLVM = buildLLVM;
      this.llvmModule = llvmModule;
    }
    public final BuildLLVM buildLLVM;
    public final LLVMModule llvmModule;
  }

  /**
   * Translate a single translation unit to an {@link LLVMModule} using
   * {@link BuildLLVM}.
   * 
   * <p>
   * Calls {@link #fail} if the resulting {@link LLVMModule} does not contain
   * a translation unit of the same name as the temporary file used to store
   * it.
   * </p>
   * 
   * @param llvmTargetTriple
   *          the target triple string for LLVM; empty string for LLVM default
   * @param llvmTargetDataLayout
   *          the target data layout string for LLVM; empty string for LLVM
   *          default
   * @param opts
   *          additional command-line options for OpenARC
   * @param warningsAsErrors
   *          whether to report warnings as errors
   * @param srcLines
   *          the source content of the translation unit, line by line
   * @return the result of the translation. Be sure to maintain a reference
   *         to the returned {@link BuildLLVM} until you're done using the
   *         returned {@link LLVMModule} (see {@link BuildLLVM#getLLVMModules}
   *         for details).
   * @throws IOException
   *           if there's a problem writing to a temporary file
   */
  protected SimpleResult buildLLVMSimple(String llvmTargetTriple,
                                         String llvmTargetDataLayout,
                                         String[] opts,
                                         boolean warningsAsErrors,
                                         String... srcLines) throws IOException
  {
    final File srcFile = writeTmpFile(".c", srcLines);
    final BuildLLVM buildLLVM = buildLLVM(llvmTargetTriple,
                                          llvmTargetDataLayout, opts,
                                          warningsAsErrors, srcFile);
    final LLVMModule[] modules = buildLLVM.getLLVMModules();
    final String[] moduleIDs = buildLLVM.getLLVMModuleIdentifiers();
    for (int i = 0; i < modules.length; ++i) {
      if (moduleIDs[i].equals(srcFile.getAbsolutePath()))
        return new SimpleResult(buildLLVM, modules[i]);
    }
    fail("no LLVMModule found for original source file");
    return null; // suppress compiler warning
  }

  /**
   * Same as
   * {@link #buildLLVMSimple(String, String, String[], boolean, String...)
   * except set {@code warningsAsErrors} to true.
   */
  protected SimpleResult buildLLVMSimple(String llvmTargetTriple,
                                         String llvmTargetDataLayout,
                                         String[] opts,
                                         String... srcLines) throws IOException
  {
    return buildLLVMSimple(llvmTargetTriple, llvmTargetDataLayout, opts,
                           true, srcLines);
  }

  /**
   * Same as
   * {@link #buildLLVMSimple(String, String, String[], boolean, String...)
   * except set {@code opts} to empty array.
   */
  protected SimpleResult buildLLVMSimple(String llvmTargetTriple,
                                         String llvmTargetDataLayout,
                                         boolean warningsAsErrors,
                                         String... srcLines) throws IOException
  {
    return buildLLVMSimple(llvmTargetTriple, llvmTargetDataLayout,
                           new String[0], warningsAsErrors, srcLines);
  }

  /**
   * Same as {@link #buildLLVMSimple(String, String, String[], String...)
   * except set {@code opts} to empty array.
   */
  protected SimpleResult buildLLVMSimple(String llvmTargetTriple,
                                         String llvmTargetDataLayout,
                                         String... srcLines) throws IOException
  {
    return buildLLVMSimple(llvmTargetTriple, llvmTargetDataLayout,
                           new String[0], srcLines);
  }

  /**
   * Translate multiple translation units to {@link LLVMModule}s using
   * {@link BuildLLVM}.
   * 
   * @param llvmTargetTriple
   *          the target triple string for LLVM; empty string for LLVM default
   * @param llvmTargetDataLayout
   *          the target data layout string for LLVM; empty string for LLVM
   *          default
   * @param opts
   *          additional command-line options for OpenARC. Keep in mind that
   *          any {@code -addIncludePath} will override the one already
   *          specified for NVL-C include files, so you might wish to
   *          generate absolute file names into include directives in your
   *          test source instead.
   * @param warningsAsErrors
   *          whether to report warnings as errors
   * @param srcFiles
   *          the source files, each of which should be created by
   *          {@link #writeTmpFile} or {@link #writeFileInTmpDir}
   * @return the {@link BuildLLVM} object storing the result of the
   *         translation. Before trying to execute any of the contained code,
   *         the caller should probably link all the contained LLVM modules
   *         into one module using {@link LLVMModule#linkModule}
   */
  protected BuildLLVM buildLLVM(String llvmTargetTriple,
                                String llvmTargetDataLayout, String[] opts,
                                boolean warningsAsErrors, File... srcFiles)
    throws IOException
  {
    final List<String> cmdLine = new ArrayList<>();
    cmdLine.add("-emitLLVM=" + llvmTargetTriple + ";" + llvmTargetDataLayout);
    cmdLine.add("-outdir=" + mkTmpDir(".cetus_output", true));
    cmdLine.add("-disableLineLLVM");
    cmdLine.add("-debugLLVM");
    if (warningsAsErrors)
      cmdLine.add("-WerrorLLVM");
    cmdLine.add("-addIncludePath="
                + new File(new File(getTopDir(), "nvl"), "include"));
    for (String opt : opts)
      cmdLine.add(opt);
    for (int i = 0; i < srcFiles.length; ++i)
      cmdLine.add(srcFiles[i].getAbsolutePath());
    final String[] cmdLineArray = new String[cmdLine.size()];
    cmdLine.toArray(cmdLineArray);
    final Driver driver = new Driver();
    driver.run(cmdLineArray);
    return (BuildLLVM)driver.getBuildLLVM();
  }

  /**
   * Same as {@link #buildLLVM(String, String, String[], boolean, File...)}
   * but without additional OpenARC command-line options.
   */
  protected BuildLLVM buildLLVM(String llvmTargetTriple,
                                String llvmTargetDataLayout,
                                boolean warningsAsErrors, File... srcFiles)
    throws IOException
  {
    return buildLLVM(llvmTargetTriple, llvmTargetDataLayout,
                     new String[0], warningsAsErrors, srcFiles);
  }

  protected SimpleResult buildLLVMSimple(String llvmTargetTriple,
                                         String llvmTargetDataLayout,
                                         boolean warningsAsErrors,
                                         String[]... srcLineArrays)
    throws IOException
  {
    int nlines = 0;
    for (String[] array : srcLineArrays)
      nlines += array.length;
    final String[] srcLines = new String[nlines];
    int i = 0;
    for (String[] array : srcLineArrays) {
      for (int j = 0; j < array.length; ++j)
        srcLines[i++] = array[j];
    }
    return buildLLVMSimple(llvmTargetTriple, llvmTargetDataLayout,
                           warningsAsErrors, srcLines);
  }

  protected static String formatFnCall(String fn, LLVMGenericValue... args) {
    final StringBuilder res = new StringBuilder(fn);
    res.append("(");
    for (int i = 0; i < args.length; ++i) {
      if (i > 0) res.append(", ");
      res.append(args[i]);
    }
    res.append(")");
    return res.toString();
  }
  protected static LLVMGenericValue runFn(LLVMExecutionEngine exec,
                                          LLVMModule llvmModule, String fn,
                                          LLVMGenericValue... args)
  {
    final LLVMFunction llvmFunction = llvmModule.getNamedFunction(fn);
    System.err.println("running LLVM function: "+formatFnCall(fn, args));
    assertNotNull(fn + " must exist as a function",
                  llvmFunction.getInstance());
    return exec.runFunction(llvmFunction, args);
  }
  protected static LLVMGenericValue runFn(LLVMExecutionEngine exec,
                                          LLVMModule llvmModule, String fn,
                                          long firstArg, long... args)
  {
    // firstArg is necessary so that, when all arguments are omitted, calls
    // are not ambiguous between this method and the previous one.
    return runFn(exec, llvmModule, fn, getIntsGeneric(llvmModule.getContext(),
                                                      firstArg, args));
  }

  protected static void checkIntFn(LLVMExecutionEngine exec,
                                   LLVMModule llvmModule, long expectedVal,
                                   boolean isSigned, String fn,
                                   LLVMGenericValue... args)
  {
    final LLVMGenericValue val = runFn(exec, llvmModule, fn, args);
    assertEquals(formatFnCall(fn, args), expectedVal,
                 val.toInt(isSigned).longValue());
  }
  protected static void checkIntFn(LLVMExecutionEngine exec,
                                   LLVMModule llvmModule, long expectedVal,
                                   String fn, LLVMGenericValue... args)
  {
    checkIntFn(exec, llvmModule, expectedVal, expectedVal < 0, fn, args);
  }
  protected static void checkIntFn(LLVMExecutionEngine exec,
                                   LLVMModule llvmModule, long expectedVal,
                                   String fn, long firstArg, long... args)
  {
    // firstArg is necessary so that, when all arguments are omitted, calls
    // are not ambiguous between this method and the previous one.
    checkIntFn(exec, llvmModule, expectedVal, expectedVal < 0, fn,
               getIntsGeneric(llvmModule.getContext(), firstArg, args));
  }
  protected static void checkIntFn(LLVMExecutionEngine exec,
                                   LLVMModule llvmModule, String fn,
                                   long expectedVal)
  {
    checkIntFn(exec, llvmModule, expectedVal, expectedVal < 0, fn);
  }
  protected static void checkIntFn(LLVMExecutionEngine exec,
                                   LLVMModule llvmModule, String fn,
                                   LLVMGenericValue arg,
                                   long expectedVal)
  {
    checkIntFn(exec, llvmModule, expectedVal, expectedVal < 0, fn, arg);
  }

  protected static void checkFloatFn(LLVMExecutionEngine exec,
                                     LLVMModule llvmModule, double expectedVal,
                                     String fn, LLVMGenericValue... args)
  {
    final LLVMContext ctxt = llvmModule.getContext();
    final LLVMGenericValue val = runFn(exec, llvmModule, fn, args);
    assertEquals(formatFnCall(fn, args), expectedVal,
                 val.toFloat(SrcFloatType.getLLVMType(ctxt)),
                 expectedVal == 0. ? 1e-6 : 1e-6*Math.abs(expectedVal));
  }
  protected static void checkFloatFn(LLVMExecutionEngine exec,
                                     LLVMModule llvmModule,
                                     double expectedVal, String fn,
                                     double firstArg, double... args)
  {
    // firstArg is necessary so that, when all arguments are omitted, calls
    // are not ambiguous between this method and the previous one.
    checkFloatFn(exec, llvmModule, expectedVal, fn,
                 getFloatsGeneric(llvmModule.getContext(), firstArg, args));
  }
  protected static void checkFloatFn(LLVMExecutionEngine exec,
                                     LLVMModule llvmModule, String fn,
                                     double expectedVal)
  {
    checkFloatFn(exec, llvmModule, expectedVal, fn);
  }
  protected static void checkFloatFn(LLVMExecutionEngine exec,
                                     LLVMModule llvmModule, String fn,
                                     LLVMGenericValue arg,
                                     double expectedVal)
  {
    checkFloatFn(exec, llvmModule, expectedVal, fn, arg);
  }

  protected static void checkDoubleFn(LLVMExecutionEngine exec,
                                      LLVMModule llvmModule, double expectedVal,
                                      String fn, LLVMGenericValue... args)
  {
    final LLVMContext ctxt = llvmModule.getContext();
    final LLVMGenericValue val = runFn(exec, llvmModule, fn, args);
    assertEquals(formatFnCall(fn, args), expectedVal,
                 val.toFloat(SrcDoubleType.getLLVMType(ctxt)),
                 expectedVal == 0. ? 1e-6 : 1e-6*Math.abs(expectedVal));
  }
  protected static void checkDoubleFn(LLVMExecutionEngine exec,
                                      LLVMModule llvmModule,
                                      double expectedVal, String fn,
                                      double firstArg, double... args)
  {
    // firstArg is necessary so that, when all arguments are omitted, calls
    // are not ambiguous between this method and the previous one.
    checkDoubleFn(exec, llvmModule, expectedVal, fn,
                  getDoublesGeneric(llvmModule.getContext(), firstArg, args));
  }
  protected static void checkDoubleFn(LLVMExecutionEngine exec,
                                      LLVMModule llvmModule, String fn,
                                      double expectedVal)
  {
    checkDoubleFn(exec, llvmModule, expectedVal, fn);
  }
  protected static void checkDoubleFn(LLVMExecutionEngine exec,
                                      LLVMModule llvmModule, String fn,
                                      LLVMGenericValue arg,
                                      double expectedVal)
  {
    checkDoubleFn(exec, llvmModule, expectedVal, fn, arg);
  }

  // We don't have a long double version of the above because LLVM otherwise
  // complains that LLVMGenericValueToFloat supports only float and double.

  /**
   * The purpose of this class is to allow us to compare the address stored in
   * a {@link SWIGTYPE_p_void}.
   */
  public static class Ptr extends SWIGTYPE_p_void {
    public Ptr(long cptr) {
      super(cptr, false);
    }
    public Ptr(SWIGTYPE_p_void o) {
      super(getCPtr(o), false);
    }
    @Override
    public boolean equals(Object obj) {
      return getCPtr(this) == getCPtr((Ptr)obj);
    }
    @Override
    public String toString() {
      return "ptr<" + getCPtr(this) + ">";
    }
  }
  protected static void checkPointerFn(LLVMExecutionEngine exec,
                                       LLVMModule llvmModule, String fn,
                                       LLVMGenericValue expectedVal)
  {
    checkPointerFn(exec, llvmModule, fn, null, expectedVal);
  }
  protected static void checkPointerFn(LLVMExecutionEngine exec,
                                       LLVMModule llvmModule, String fn,
                                       LLVMGenericValue arg,
                                       LLVMGenericValue expectedVal)
  {
    final LLVMContext ctxt = llvmModule.getContext();
    final LLVMGenericValue val
      = arg != null
        ? runFn(exec, llvmModule, fn, new LLVMGenericValue[]{arg})
        : runFn(exec, llvmModule, fn, new LLVMGenericValue[]{});
    final LLVMPointerType ptrType
      = LLVMPointerType.get(SrcVoidType.getLLVMTypeAsPointerTarget(ctxt), 0);
    assertEquals(fn + "(" + (arg == null ? "" : arg.toString()) + ")",
                 new Ptr(expectedVal.toPointer(ptrType)),
                 new Ptr(val.toPointer(ptrType)));
  }

  protected static void checkGlobalVar(LLVMModule llvmModule, String name,
                                       SrcType type)
  {
    final LLVMGlobalVariable var = llvmModule.getNamedGlobal(name);
    assertNotNull(name + " must exist as a global variable",
                  var.getInstance());
    assertNotNull(name + "'s initializer must exist", !var.isDeclaration());
    assertEquals(name + "'s type",
                 SrcPointerType.get(type).getLLVMType(llvmModule.getContext()),
                 var.typeOf());
  }
  protected static LLVMConstant checkGlobalVarGetInit(LLVMModule llvmModule,
                                                      String name)
  {
    final LLVMGlobalVariable var = llvmModule.getNamedGlobal(name);
    assertNotNull(name + " must exist as a global variable",
                  var.getInstance());
    assertNotNull(name + "'s initializer must exist", !var.isDeclaration());
    return var.getInitializer();
  }
  protected static void checkGlobalVar(LLVMModule llvmModule, String name,
                                       LLVMConstant expectedInit)
  {
    final LLVMConstant init = checkGlobalVarGetInit(llvmModule, name);
    // Checking the type separately from the value sometimes improves the
    // helpfulness of error messages. That is, instead of something like
    //
    //   var initializer's value expected:<org.jllvm.LLVMConstantPointer@750fd2df>
    //   but was:<org.jllvm.LLVMConstantPointer@61dd6a83>
    //
    // you get something like
    //
    //   var initializer's type expected:<i32 addrspace(1)*> but was:<i32*>
    assertEquals(name + " initializer's type",
                 expectedInit.typeOf(), init.typeOf());
    assertEquals(name + " initializer's value", expectedInit, init);
  }
  protected static void checkIntGlobalVar(LLVMModule llvmModule, String name,
                                          long expectedInitVal)
  {
    final LLVMConstant init = checkGlobalVarGetInit(llvmModule, name);
    assertTrue(name + " initializer's value must be a constant integer",
               init instanceof LLVMConstantInteger);
    final LLVMConstantInteger initInt = (LLVMConstantInteger)init;
    assertEquals(name + " initializer's value", expectedInitVal,
                 expectedInitVal < 0 ? initInt.getSExtValue()
                                     : initInt.getZExtValue().longValue());
  }
  protected static void checkFloatGlobalVar(LLVMModule llvmModule,
                                            String name, double expectedVal)
  {
    final LLVMConstant init = checkGlobalVarGetInit(llvmModule, name);
    assertEquals(
      name + " initializer's value",
      LLVMConstantReal.get(SrcFloatType.getLLVMType(llvmModule.getContext()),
                           expectedVal),
      init);
  }
  protected static void checkDoubleGlobalVar(LLVMModule llvmModule,
                                             String name, double expectedVal)
  {
    final LLVMConstant init = checkGlobalVarGetInit(llvmModule, name);
    assertEquals(
      name + " initializer's value",
      LLVMConstantReal.get(SrcDoubleType.getLLVMType(llvmModule.getContext()),
                           expectedVal),
      init);
  }

  protected static LLVMConstantInteger getConstBool(long val,
                                                    LLVMContext ctxt)
  {
    return LLVMConstantInteger.get(SrcBoolType.getLLVMType(ctxt), val, false);
  }
  protected static LLVMConstantInteger getConstSignedChar(long val,
                                                          LLVMContext ctxt)
  {
    return LLVMConstantInteger.get(SrcSignedCharType.getLLVMType(ctxt), val,
                                   false);
  }
  protected static LLVMConstantInteger getConstUnsigned(long val,
                                                        LLVMContext ctxt)
  {
    return LLVMConstantInteger.get(SrcUnsignedIntType.getLLVMType(ctxt), val,
                                   false);
  }
  protected static LLVMConstantInteger getConstInt(long val,
                                                   LLVMContext ctxt)
  {
    return LLVMConstantInteger.get(SrcIntType.getLLVMType(ctxt), val, true);
  }
  protected static LLVMConstantInteger getSizeTInteger(long nbytes,
                                                       LLVMContext ctxt)
  {
    return LLVMConstantInteger.get(SRC_SIZE_T_TYPE.getLLVMType(ctxt), nbytes,
                                   false);
  }
  protected static LLVMConstantReal getConstFloat(double val,
                                                  LLVMContext ctxt)
  {
    return LLVMConstantReal.get(SrcFloatType.getLLVMType(ctxt), val);
  }
  protected static LLVMConstantReal getConstDouble(double val,
                                                   LLVMContext ctxt)
  {
    return LLVMConstantReal.get(SrcDoubleType.getLLVMType(ctxt), val);
  }
  protected static LLVMConstantReal getConstLongDouble(double val,
                                                       LLVMContext ctxt)
  {
    return LLVMConstantReal.get(SrcLongDoubleType.getLLVMType(ctxt), val);
  }

  protected static LLVMGenericInt getIntegerGeneric(SrcIntegerType srcType,
                                                    long val, LLVMContext ctxt)
  {
    BigInteger bigVal = BigInteger.valueOf(val);
    if (val < 0)
      bigVal = BigInteger.ONE.shiftLeft((int)srcType.getWidth()).add(bigVal);
    return new LLVMGenericInt(srcType.getLLVMType(ctxt), bigVal,
                              srcType.isSigned());
  }
  protected static LLVMGenericInt getBoolGeneric(long val, LLVMContext ctxt) {
    return getIntegerGeneric(SrcBoolType, val, ctxt);
  }
  protected static LLVMGenericInt getUnsignedCharGeneric(long val, LLVMContext ctxt) {
    return getIntegerGeneric(SrcUnsignedCharType, val, ctxt);
  }
  protected static LLVMGenericInt getSignedCharGeneric(long val, LLVMContext ctxt) {
    return getIntegerGeneric(SrcSignedCharType, val, ctxt);
  }
  protected static LLVMGenericInt getCharGeneric(long val, LLVMContext ctxt) {
    return getIntegerGeneric(SrcCharType, val, ctxt);
  }
  protected static LLVMGenericInt getIntGeneric(long val, LLVMContext ctxt) {
    return getIntegerGeneric(SrcIntType, val, ctxt);
  }
  protected static LLVMGenericInt[] getIntsGeneric(LLVMContext ctxt,
                                                   long firstVal, long... vals)
  {
    final LLVMGenericInt[] valsGen = new LLVMGenericInt[1+vals.length];
    for (int i = 0; i < 1+vals.length; ++i) {
      final long val = i == 0 ? firstVal : vals[i-1];
      valsGen[i] = getIntGeneric(val, ctxt);
    }
    return valsGen;
  }
  protected static LLVMGenericInt getUnsignedIntGeneric(long val,
                                                        LLVMContext ctxt)
  {
    return getIntegerGeneric(SrcUnsignedIntType, val, ctxt);
  }
  protected static LLVMGenericInt getUnsignedLongGeneric(long val,
                                                         LLVMContext ctxt)
  {
    return getIntegerGeneric(SrcUnsignedLongType, val, ctxt);
  }
  protected static LLVMGenericInt getLongGeneric(long val, LLVMContext ctxt) {
    return getIntegerGeneric(SrcLongType, val, ctxt);
  }

  protected static LLVMGenericReal getGenericReal(
    SrcPrimitiveFloatingType srcType, double val, LLVMContext ctxt)
  {
    return new LLVMGenericReal(srcType.getLLVMType(ctxt), val);
  }
  protected static LLVMGenericReal getFloatGeneric(
    double val, LLVMContext ctxt)
  {
    return getGenericReal(SrcFloatType, val, ctxt);
  }
  protected static LLVMGenericReal getDoubleGeneric(
    double val, LLVMContext ctxt)
  {
    return getGenericReal(SrcDoubleType, val, ctxt);
  }
  protected static LLVMGenericReal getLongDoubleGeneric(
    double val, LLVMContext ctxt)
  {
    return getGenericReal(SrcLongDoubleType, val, ctxt);
  }
  protected static LLVMGenericReal[] getFloatsGeneric(
    LLVMContext ctxt, double firstVal, double... vals)
  {
    final LLVMGenericReal[] valsGen = new LLVMGenericReal[1+vals.length];
    for (int i = 0; i < 1+vals.length; ++i) {
      final double val = i == 0 ? firstVal : vals[i-1];
      valsGen[i] = getFloatGeneric(val, ctxt);
    }
    return valsGen;
  }
  protected static LLVMGenericReal[] getDoublesGeneric(
    LLVMContext ctxt, double firstVal, double... vals)
  {
    final LLVMGenericReal[] valsGen = new LLVMGenericReal[1+vals.length];
    for (int i = 0; i < 1+vals.length; ++i) {
      final double val = i == 0 ? firstVal : vals[i-1];
      valsGen[i] = getDoubleGeneric(val, ctxt);
    }
    return valsGen;
  }
  protected static LLVMGenericReal[] getLongDoublesGeneric(
    LLVMContext ctxt, double firstVal, double... vals)
  {
    final LLVMGenericReal[] valsGen = new LLVMGenericReal[1+vals.length];
    for (int i = 0; i < 1+vals.length; ++i) {
      final double val = i == 0 ? firstVal : vals[i-1];
      valsGen[i] = getLongDoubleGeneric(val, ctxt);
    }
    return valsGen;
  }

  private static String typeIDToType(String tid) {
    String res = "";
    for (int i = 0; i < tid.length(); ++i) {
      if (i > 0) res+=" ";
      switch (tid.charAt(i)) {
      case 'v': res += "void";     break;
      case 'l': res += "long";     break;
      case 'd': res += "double";   break;
      case 'f': res += "float";    break;
      case 'u': res += "unsigned"; break;
      case 's': res += "signed";   break;
      case 'i': res += "int";      break;
      case 'c': res += "char";     break;
      case 'b': res += "_Bool";    break;
      case 'p':
        res += typeIDToType(tid.substring(i+1));
        res += "*";
        return res;
      default: throw new IllegalStateException();
      }
    }
    return res;
  }
  private static String operatorToName(String oper) {
    final String name;
    switch (oper) {
    case "*":  name = "mul"; break;
    case "/":  name = "div"; break;
    case "%":  name = "rem"; break;
    case "+":  name = "add"; break;
    case "-":  name = "sub"; break;
    case "<<": name = "shl"; break;
    case ">>": name = "shr"; break;
    case "<":  name = "lt"; break;
    case ">":  name = "gt"; break;
    case "<=": name = "le"; break;
    case ">=": name = "ge"; break;
    case "==": name = "eq"; break;
    case "!=": name = "ne"; break;
    case "&":  name = "bitwiseAnd"; break;
    case "^":  name = "bitwiseXor"; break;
    case "|":  name = "bitwiseOr";  break;
    case "&&": name = "logicalAnd"; break;
    case "||": name = "logicalOr";  break;
    case "?:": name = "cond"; break;
    default: throw new IllegalStateException();
    }
    return name;
  }

  /**
   * This can be combined with {@link #checkBinaryArith} to generate checks
   * for binary operators. Types are specified in the form understood by
   * {@link #typeIDtoType}. Operands must be constant expressions and are
   * cast/assigned to those types before being supplied to the binary
   * operator.
   * 
   * <p>
   * For the specified operand types and operand values, this checks the
   * approximate result value as well as the result type's sizeof and
   * signedness. It checks those in constant-expression context and in
   * non-constant expression context (operand values are stored in variables,
   * which are used as the operands so a constant expression cannot be
   * formed). For signedness checks to be effective, the result value must be
   * a large enough positive number or a negative number so that the value's
   * sign will be reversed if signedness of the result type is reversed.
   * </p>
   * 
   * <p>
   * These methods are also used by {@link BuildLLVMTest_OtherExpressions}.
   * </p>
   */
  protected static String genBinaryArith(String oper, String tid1, String tid2,
                                         String op1, String op2)
  {
    // We store the actual result values in doubles even when testing
    // integers. For negative or large positive actual result values, that
    // allows us to determine if the actual result integer type is signed
    // (converting an integer to a floating type maintains the sign based on
    // the integer type). Alternatively, we could store the result value in an
    // integer type wider than the expected result integer type and check
    // whether sign extension occurs, but that strategy is impossible if the
    // expected result integer type is already the widest integer type, and it
    // fails if the actual result integer type is already the widest integer
    // type regardless of the expected result integer type.
    // 
    // By storing the result in a double, we often get only an approximate
    // actual result value (integer to floating conversion can be lossy).
    // That's OK as we're not trying to check LLVM's instructions. Instead,
    // we're checking BuildLLVM's ability to compute types correctly and
    // generate the correct LLVM instructions, and incorrect types or
    // instructions usually produce either significantly different values or
    // identical values not just approximately correct values.
    // 
    // We choose double over long double simply because, as of LLVM 3.2,
    // LLVMExecutionEngine.runFunction doesn't support returning long double.
    return genTernaryArith(oper, null, tid1, tid2, null, op1, op2, null);
  }

  protected static String genTernaryArith(
    String oper1, String oper2, String tid1, String tid2, String tid3,
    String op1, String op2, String op3)
  {
    return genTernary(oper1, oper2, "d", tid1, tid2, tid3, op1, op2, op3);
  }

  protected static String genBinaryPtr(String oper, String tidr,
                                       String tid1, String tid2,
                                       String op1, String op2)
  {
    return genTernaryPtr(oper, null, tidr, tid1, tid2, null, op1, op2, null);
  }

  protected static String genTernaryPtr(String oper1, String oper2,
                                        String tidr,
                                        String tid1, String tid2, String tid3,
                                        String op1, String op2, String op3)
  {
    assert(tidr.charAt(0) == 'p');
    assert((oper2==null) == (tid3==null));
    assert((oper2==null) == (op3==null));
    final String name = operatorToName(oper1 + (oper2==null ? "" : oper2))
                        + "_"+tid1 + "_"+tid2 + (oper2==null ? "" : "_"+tid3);
    return
      genTernary(oper1, oper2, tidr, tid1, tid2, tid3, op1, op2, op3)
      + "double cvd_"+name+"() { return (double)*cv_"+name+"_; }\n"
      + "double vd_"+name+"() { return (double)*v_"+name+"(); }\n"
      ;
  }

  private static String genTernary(String oper1, String oper2, String tidr,
                                   String tid1, String tid2, String tid3,
                                   String op1, String op2, String op3)
  {
    assert((oper2==null) == (tid3==null));
    assert((oper2==null) == (op3==null));
    final String name = operatorToName(oper1 + (oper2==null ? "" : oper2))
                        + "_"+tid1 + "_"+tid2 + (oper2==null ? "" : "_"+tid3);
    final String tr = typeIDToType(tidr);
    final String t1 = typeIDToType(tid1);
    final String t2 = typeIDToType(tid2);
    final String t3 = oper2==null ? null : typeIDToType(tid3);
    final String c_call
      = "("+t1+")("+op1+") "+oper1+" ("+t2+")("+op2+")"
        + (oper2==null ? "" : " "+oper2+" ("+t3+")("+op3+")");
    final String opDecls
      = t1+" op1="+op1+"; "+t2+" op2="+op2+";"
        + (oper2==null ? "" : " "+t3+" op3="+op3+";");
    final String call
      = "op1 "+oper1+" op2" + (oper2==null ? "" : " "+oper2+"op3");
    return
        "size_t cs_"+name+" = sizeof(" + c_call + ");\n"
      + tr+" cv_"+name+"_ = " + c_call + ";\n"
      + tr+" cv_"+name+"() { return cv_"+name+"_; }\n"
      + "size_t s_"+name+"() {\n"
      + "  "+opDecls+"\n"
      + "  return sizeof("+call+");\n"
      + "}\n"
      + tr+" v_"+name+"(){\n"
      + "  "+opDecls+"\n"
      + "  return "+call+";\n"
      + "}\n"
      ;
  }

  /** See {@link #genBinaryArith}. */
  protected static void checkBinaryArith(
    LLVMExecutionEngine exec, LLVMModule mod,
    String oper, String tid1, String tid2, long size, double val)
  {
    checkTernaryArith(exec, mod, oper, null, tid1, tid2, null, size, val);
  }

  protected static void checkTernaryArith(
    LLVMExecutionEngine exec, LLVMModule mod,
    String oper1, String oper2,
    String tid1, String tid2, String tid3,
    long size, double val)
  {
    checkTernary(exec, mod, oper1, oper2, "d", tid1, tid2, tid3, size, val,
                 "v");
  }

  protected static void checkBinaryPtr(
    LLVMExecutionEngine exec, LLVMModule mod,
    String oper, String tidr, String tid1, String tid2, long size, double val)
  {
    checkTernaryPtr(exec, mod, oper, null, tidr, tid1, tid2, null, size, val);
  }

  protected static void checkTernaryPtr(
    LLVMExecutionEngine exec, LLVMModule mod,
    String oper1, String oper2, String tidr,
    String tid1, String tid2, String tid3, long size, double val)
  {
    checkTernary(exec, mod, oper1, oper2, tidr, tid1, tid2, tid3, size, val,
                 "vd");
  }

  private static void checkTernary(
    LLVMExecutionEngine exec, LLVMModule mod,
    String oper1, String oper2, String tidr,
    String tid1, String tid2, String tid3,
    long size, double val, String valName)
  {
    assert((oper2==null) == (tid3==null));
    final String name = operatorToName(oper1 + (oper2==null ? "" : oper2))
                        + "_"+tid1 + "_"+tid2 + (oper2==null ? "" : "_"+tid3);
    checkGlobalVar(mod, "cs_"+name, getSizeTInteger(size, mod.getContext()));
    checkDoubleFn(exec, mod, "c"+valName+"_"+name, val);
    checkIntFn(exec, mod, "s_"+name, size);
    checkDoubleFn(exec, mod, valName+"_"+name, val);
  }
}
