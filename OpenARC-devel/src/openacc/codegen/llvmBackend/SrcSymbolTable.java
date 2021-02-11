package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcParamAndReturnType.SRC_PARAMRET_CONST_NVL_HEAP_PTR;
import static openacc.codegen.llvmBackend.SrcParamAndReturnType.SRC_PARAMRET_NVL_HEAP_PTR;
import static openacc.codegen.llvmBackend.SrcParamType.SRC_PARAM_DISCARD;
import static openacc.codegen.llvmBackend.SrcParamType.SRC_PARAM_INT_GT1;
import static openacc.codegen.llvmBackend.SrcParamType.SRC_PARAM_METADATA_NULL_PTR_FROM_RET_TYPE;
import static openacc.codegen.llvmBackend.SrcParamType.SRC_PARAM_METADATA_NULL_PTR_FROM_ARG_TYPE;
import static openacc.codegen.llvmBackend.SrcParamType.SRC_PARAM_MPI_GROUP;
import static openacc.codegen.llvmBackend.SrcParamType.SRC_PARAM_NULL_VOID_PTR;
import static openacc.codegen.llvmBackend.SrcParamType.SRC_PARAM_PTR_TO_NVL;
import static openacc.codegen.llvmBackend.SrcParamType.SRC_PARAM_TYPE_CHECKSUM_FROM_PTR_ARG;
import static openacc.codegen.llvmBackend.SrcParamType.SRC_PARAM_TYPE_CHECKSUM_FROM_RET_TYPE;
import static openacc.codegen.llvmBackend.SrcParamType.SRC_PARAM_VA_LIST_LVALUE;
import static openacc.codegen.llvmBackend.SrcParamType.SRC_PARAM_VA_LIST_RVALUE;
import static openacc.codegen.llvmBackend.SrcPrimitiveFloatingType.SrcDoubleType;
import static openacc.codegen.llvmBackend.SrcPrimitiveFloatingType.SrcFloatType;
import static openacc.codegen.llvmBackend.SrcPrimitiveFloatingType.SrcLongDoubleType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SRC_UINT16_T_TYPE;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SRC_UINT32_T_TYPE;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SRC_UINT64_T_TYPE;
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
import static openacc.codegen.llvmBackend.SrcReturnType.SRC_RET_BARE_PTR_TYPE_FROM_ARG;
import static openacc.codegen.llvmBackend.SrcReturnType.SRC_RET_NVL_PTR_TYPE_FROM_TYPE_ARG;
import static openacc.codegen.llvmBackend.SrcReturnType.SRC_RET_NVM_PTR_FROM_ARG;
import static openacc.codegen.llvmBackend.SrcReturnType.SRC_RET_NVM_PTR_TYPE_FROM_TYPE_ARG;
import static openacc.codegen.llvmBackend.SrcReturnType.SRC_RET_TYPE_IS_TYPE_ARG;
import static org.jllvm.bindings.LLVMLinkage.LLVMInternalLinkage;

import java.io.UnsupportedEncodingException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import openacc.codegen.llvmBackend.SrcFunctionBuiltin.Instruction;
import openacc.codegen.llvmBackend.SrcQualifiedType.SrcTypeQualifier;

import org.jllvm.LLVMArrayType;
import org.jllvm.LLVMBasicBlock;
import org.jllvm.LLVMConstant;
import org.jllvm.LLVMConstantArray;
import org.jllvm.LLVMConstantExpression;
import org.jllvm.LLVMConstantInteger;
import org.jllvm.LLVMContext;
import org.jllvm.LLVMFunction;
import org.jllvm.LLVMGetElementPointerInstruction;
import org.jllvm.LLVMGlobalValue;
import org.jllvm.LLVMGlobalVariable;
import org.jllvm.LLVMIdentifiedStructType;
import org.jllvm.LLVMInstructionBuilder;
import org.jllvm.LLVMIntegerType;
import org.jllvm.LLVMModule;
import org.jllvm.LLVMStackAllocation;
import org.jllvm.LLVMType;
import org.jllvm.LLVMValue;
import org.jllvm.bindings.LLVMLinkage;

import cetus.hir.Case;
import cetus.hir.ClassDeclaration;
import cetus.hir.Declaration;
import cetus.hir.Declarator;
import cetus.hir.Default;
import cetus.hir.Enumeration;
import cetus.hir.Label;
import cetus.hir.NestedDeclarator;
import cetus.hir.Procedure;
import cetus.hir.ProcedureDeclarator;
import cetus.hir.Program;
import cetus.hir.Specifier;
import cetus.hir.Symbol;
import cetus.hir.SymbolTable;
import cetus.hir.SymbolTools;
import cetus.hir.TranslationUnit;
import cetus.hir.Traversable;
import cetus.hir.UserSpecifier;

/**
 * The LLVM backend's class for looking up and indexing symbols used in the
 * C source.
 * 
 * TODO: The methods of this class need documentation.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public class SrcSymbolTable {
  /**
   * The algorithm used to compute type checksums. We're not concerned about
   * security, so MD5 is probably fine. The number of bytes in the checksum
   * (16 bytes) must match the number of bytes expected by the NVL runtime.
   */
  public static final String TYPE_CHECKSUM_ALGORITHM = "MD5";
  /**
   * The encoding used to convert a type compatibility string to a byte
   * array before computing the type checksum. We choose UTF-8 arbitrarily.
   * Maybe there's a better choice, but the important thing is that we
   * consistently choose the same encoding.
   */
  public static final String TYPE_COMPATSTR_ENCODING = "UTF-8";

  /**
   * Map in which the key is a node that has no parent in the Cetus IR, and
   * the value is the node's logical parent. An entry is added when such a
   * parent-less node is visited and before its descendant nodes are
   * visited. {@link #lookupSymbolTable} uses this map to facilitate correct
   * symbol lookup from such descendant nodes.
   */
  private final Map<Traversable, Traversable> parentFixupMap
    = new IdentityHashMap<>();

  /** Cache of built-in types used in the C source. */
  private final BuiltinTypeTable builtinTypeTable = new BuiltinTypeTable();
  /**
   * Map from the (topmost) {@link Declarator} for a typedef in the source
   * code to the {@link SrcType} to which it resolves.
   */
  private final Map<Declarator, SrcType> typedefMap = new IdentityHashMap<>();
  /**
   * Map from an {@link Enumeration} in the source code to the
   * {@link SrcEnumType} for it.
   */
  private final Map<Enumeration, SrcEnumType> enumMap
    = new IdentityHashMap<>();
  /**
   * Map from a named struct or union in the source code to its
   * {@link SrcStructOrUnionType}. Accessed only by
   * {@link #getSrcStructOrUnionType} and {@link #checkNVMStoredStructs}.
   */
  private final Map<StructOrUnionKey, SrcStructOrUnionType> structOrUnionMap
    = new HashMap<>();

  /**
   * Map from a function builtin name to its {@link SrcFunctionBuiltin}.
   */
  private final Map<String, SrcFunctionBuiltin> srcFunctionBuiltinMap
    = new HashMap<>();
  /**
   * Map in which the key is the (topmost) {@link Declarator} either for a
   * local variable or for a local declaration of a global variable or
   * function, and the value is its lvalue or its function designator. The
   * lvalue contains a {@link LLVMStackAllocation} or a
   * {@link LLVMGlobalVariable}, or the function designator contains a
   * {@link LLVMFunction}. However, a {@link LLVMGlobalVariable} or
   * {@link LLVMFunction} might be wrapped in a constant-expression bitcast
   * to a local composite type.
   */
  private Map<Declarator, ValueAndType> localMap = new IdentityHashMap<>();
  /**
   * Each element in the array corresponds to a translation unit of the same
   * index. Each is a map from a file-scope variable's or static local
   * variable's LLVM name (a static local variable's name is different in
   * LLVM than in the source code) to its lvalue, which contain its
   * {@link LLVMGlobalVariable}.
   */
  private Map<String, ValueAndType> globalVarMaps[] = null;
  /**
   * Each element in the array corresponds to a translation unit of the same
   * index. Each is a set of all file-scope variables (by LLVM name, which
   * is the same as the source name) for which at least one tentative
   * definition but no external definition has been seen so far.
   */
  private Set<String> globalVarTentativeDefSets[] = null;
  /**
   * Each element in the array corresponds to a translation unit of the same
   * index. Each is a map from a function name to its function designator
   * (see {@link ValueAndType#ValueAndType} preconditions for how that's
   * encoded).
   */
  private Map<String, ValueAndType> functionMaps[] = null;
  /**
   * Each element in the array corresponds to a translation unit of the same
   * index. Each is a map from a function name to its {@link InlineDefState}.
   * If there is no entry for a function name, then either the function is
   * internally linked or no file-scope function declaration/definition has
   * yet been seen for that name in this translation unit (block-scope
   * declarations don't affect whether a function's definition is an inline
   * definition).
   */
  private Map<String, InlineDefState> functionInlineDefStateMaps[] = null;

  /**
   * Map from each {@link Label} node to the basic block to which it refers.
   */
  private Map<Label, LLVMBasicBlock> labelMap = new HashMap<>();
  /**
   * Map from each {@link Case} node to the basic block to which it refers.
   */
  private Map<Case, LLVMBasicBlock> caseMap = new HashMap<>();
  /**
   * Map from each {@link Default} node to the basic block to which it
   * refers.
   */
  private Map<Default, LLVMBasicBlock> defaultMap = new HashMap<>();

  /**
   * Each element in the array corresponds to a translation unit of the same
   * index. Each element is a map. Each key is one of that translation
   * unit's {@link SrcType}s for which a checksum must be computed. Each
   * value is the LLVM global variable, of type {@code i8[]}, that should be
   * initialized with that checksum
   */
  private Map<SrcType, LLVMGlobalVariable> typeChecksumMap[] = null;
  /** Should be accessed only by {@link #getMessageDigest}. */
  private MessageDigest messageDigest = null;
  /**
   * Get the {@link MessageDigest} used to compute the checksums in
   * {@link typeChecksumMap}. We don't construct it unless it's needed
   * because its construction could throw an exception if the chosen
   * algorithm isn't supported.
   */
  private MessageDigest getMessageDigest() {
    if (messageDigest != null)
      return messageDigest;
    try {
      messageDigest = MessageDigest.getInstance(TYPE_CHECKSUM_ALGORITHM);
    }
    catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException(
        "failure accessing "+TYPE_CHECKSUM_ALGORITHM
        +" algorithm for type checksum",
        e);
    }
    return messageDigest;
  }
  /**
   * Whether debugging output has been enabled for type checksum
   * computations.
   */
  private final boolean debugTypeChecksums;

  /**
   * Whether a function's declarations so far in a translation unit indicate
   * that any definition of it in that translation will be an inline
   * definition (and thus a definition must appear in the translation unit)
   * instead of an external definition. See ISO C99 sec. 6.7.4p6.
   */
  public static enum InlineDefState {
    /**
     * C99 semantics so far specify an inline definition. Can be overridden
     * by any other state.
     */
    INLINE_C99_SO_FAR,
    /**
     * C99 semantics specify no inline definition. Can be overridden by any
     * GNU state only.
     */
    INLINE_C99_NOT,
    /**
     * GNU semantics specify an inline definition. Cannot be overridden by
     * any other state.
     */
    INLINE_GNU,
    /**
     * GNU semantics specify no inline definition. Cannot be overridden by
     * any other state.
     */
    INLINE_GNU_NOT,
  };

  @SuppressWarnings("unchecked")
  private void allocateModuleArrays(int nModules) {
    globalVarMaps = new Map[nModules];
    globalVarTentativeDefSets = new Set[nModules];
    functionMaps = new Map[nModules];
    functionInlineDefStateMaps = new Map[nModules];
    typeChecksumMap = new Map[nModules];
  }

  public SrcSymbolTable(LLVMContext llvmContext, int nModules,
                        boolean debugTypeChecksums)
  {
    this.debugTypeChecksums = debugTypeChecksums;

    allocateModuleArrays(nModules);
    for (int i = 0; i < nModules; ++i) {
      globalVarMaps[i] = new HashMap<>();
      globalVarTentativeDefSets[i] = new HashSet<>();
      functionMaps[i] = new HashMap<>();
      functionInlineDefStateMaps[i] = new HashMap<>();
      typeChecksumMap[i] = new IdentityHashMap<>();
    }

    final SrcType cv
      = SrcQualifiedType.get(SrcVoidType, SrcTypeQualifier.CONST);
    final SrcType cc
      = SrcQualifiedType.get(SrcCharType, SrcTypeQualifier.CONST);
    final SrcPointerType pv = SrcPointerType.get(SrcVoidType);
    final SrcPointerType pc = SrcPointerType.get(SrcCharType);
    final SrcPointerType pcv = SrcPointerType.get(cv);
    final SrcPointerType pcc = SrcPointerType.get(cc);
    final SrcType rpv
      = SrcQualifiedType.get(pv, SrcTypeQualifier.RESTRICT);
    final SrcType rpcv
      = SrcQualifiedType.get(pcv, SrcTypeQualifier.RESTRICT);
    final SrcType rpc
      = SrcQualifiedType.get(pc, SrcTypeQualifier.RESTRICT);
    final SrcType rpcc
      = SrcQualifiedType.get(pcc, SrcTypeQualifier.RESTRICT);

    // va_*
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_va_start", "llvm.va_start", SrcVoidType,
      false, SRC_PARAM_VA_LIST_LVALUE, SRC_PARAM_DISCARD));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_va_end", "llvm.va_end", SrcVoidType,
      false, SRC_PARAM_VA_LIST_LVALUE));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_va_copy", "llvm.va_copy", SrcVoidType,
      false, SRC_PARAM_VA_LIST_LVALUE, SRC_PARAM_VA_LIST_LVALUE));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_va_arg", Instruction.VA_ARG, SRC_RET_TYPE_IS_TYPE_ARG,
      false, SRC_PARAM_VA_LIST_LVALUE));

    // fabs*
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_fabsf",
      "llvm.fabs.f" + SrcFloatType.getLLVMType(llvmContext)
                      .getPrimitiveSizeInBits(),
      SrcFloatType, false, SrcFloatType));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_fabs",
      "llvm.fabs.f" + SrcDoubleType.getLLVMType(llvmContext)
                      .getPrimitiveSizeInBits(),
      SrcDoubleType, false, SrcDoubleType));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_fabsl",
      "llvm.fabs.f" + SrcLongDoubleType.getLLVMType(llvmContext)
                      .getPrimitiveSizeInBits(),
      SrcLongDoubleType, false, SrcLongDoubleType));

    // inf* and huge_val*
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_inff",
      new ValueAndType(SrcFloatType.getInfinity(llvmContext),
                       SrcFloatType, false)));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_inf",
      new ValueAndType(SrcDoubleType.getInfinity(llvmContext),
                       SrcDoubleType, false)));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_infl",
      new ValueAndType(SrcLongDoubleType.getInfinity(llvmContext),
                       SrcLongDoubleType, false)));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_huge_valf",
      new ValueAndType(SrcFloatType.getInfinity(llvmContext),
                       SrcFloatType, false)));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_huge_val",
      new ValueAndType(SrcDoubleType.getInfinity(llvmContext),
                       SrcDoubleType, false)));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_huge_vall",
      new ValueAndType(SrcLongDoubleType.getInfinity(llvmContext),
                       SrcLongDoubleType, false)));

    // object_size and *_chk
    //
    // These are listed at:
    // https://gcc.gnu.org/onlinedocs/gcc-4.9.2/gcc/Object-Size-Checking.html#Object-Size-Checking
    //
    // TODO: We're cheating by translating all *_chk directly to the
    // corresponding standard library functions that do not perform any
    // object size checking. Perhaps library functions that do perform
    // object size checking are standard enough that we should just
    // translate to them instead. See:
    //
    // http://refspecs.linux-foundation.org/LSB_4.0.0/LSB-Core-generic/LSB-Core-generic/libc---memcpy-chk-1.html
    //
    // Here are some related LLVM intrinsics, but I'm not sure yet when to
    // use them:
    //
    // http://llvm.org/docs/LangRef.html#standard-c-library-intrinsics
    //
    // TODO: These need to checked in the test suite. For now, only SPEC
    // CPU 2006 is exercising some of them, but we don't always run it
    // because it's slow.
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_object_size",
      "llvm.objectsize.i" + SRC_SIZE_T_TYPE.getLLVMWidth(), SRC_SIZE_T_TYPE,
      false, pv, SRC_PARAM_INT_GT1));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin___memcpy_chk", "memcpy", pv,
      false, rpv, rpcv, SRC_SIZE_T_TYPE, SRC_PARAM_DISCARD));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin___memmove_chk", "memmove", pv,
      false, pv, pcv, SRC_SIZE_T_TYPE, SRC_PARAM_DISCARD));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin___memset_chk", "memset", pv,
      false, pv, SrcIntType, SRC_SIZE_T_TYPE, SRC_PARAM_DISCARD));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin___strcpy_chk", "strcpy", pc,
      false, rpc, rpcc, SRC_PARAM_DISCARD));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin___stpcpy_chk", "stpcpy", pc,
      false, pc, pcc, SRC_PARAM_DISCARD));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin___strncpy_chk", "strncpy", pc,
      false, rpc, rpcc, SRC_SIZE_T_TYPE, SRC_PARAM_DISCARD));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin___strcat_chk", "strcat", pc,
      false, rpc, rpcc, SRC_PARAM_DISCARD));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin___strncat_chk", "strncat", pc,
      false, rpc, rpcc, SRC_SIZE_T_TYPE, SRC_PARAM_DISCARD));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin___sprintf_chk", "sprintf", SrcIntType,
      true, rpc, SRC_PARAM_DISCARD, SRC_PARAM_DISCARD, rpcc));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin___snprintf_chk", "snprintf", SrcIntType,
      true, rpc, SRC_SIZE_T_TYPE, SRC_PARAM_DISCARD, SRC_PARAM_DISCARD,
      rpcc));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin___vsprintf_chk", "vsprintf", SrcIntType,
      false, rpc, SRC_PARAM_DISCARD, SRC_PARAM_DISCARD, rpcc,
      SRC_PARAM_VA_LIST_RVALUE));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin___vsnprintf_chk", "vsnprintf", SrcIntType,
      false, rpc, SRC_SIZE_T_TYPE, SRC_PARAM_DISCARD, SRC_PARAM_DISCARD,
      rpcc, SRC_PARAM_VA_LIST_RVALUE));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin___printf_chk", "printf", SrcIntType,
      true, SRC_PARAM_DISCARD, rpcc));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin___vprintf_chk", "vprintf", SrcIntType,
      false, SRC_PARAM_DISCARD, rpcc, SRC_PARAM_VA_LIST_RVALUE));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin___fprintf_chk", "fprintf", SrcIntType,
      true, rpv, SRC_PARAM_DISCARD, rpcc));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin___vfprintf_chk", "vfprintf", SrcIntType,
      false, rpv, SRC_PARAM_DISCARD, rpcc, SRC_PARAM_VA_LIST_RVALUE));

    // alloca
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_alloca", Instruction.ALLOCA, pv,
      false, SRC_SIZE_T_TYPE));

    // expect
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_expect", "llvm.expect.i"+SrcLongType.getLLVMWidth(),
      SrcLongType, false, SrcLongType,
      new SrcParamConstExprType(SrcLongType)));

    // bswap*
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_bswap16", "llvm.bswap.i16",
      SRC_UINT16_T_TYPE, false, SRC_UINT16_T_TYPE));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_bswap32", "llvm.bswap.i32",
      SRC_UINT32_T_TYPE, false, SRC_UINT32_T_TYPE));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_bswap64", "llvm.bswap.i64",
      SRC_UINT64_T_TYPE, false, SRC_UINT64_T_TYPE));

    // nvl_*
    //
    // We don't know the size of mode_t on the target (I've seen 16 bits and
    // 32 bits), but unsigned long is probably more than big enough.
    // llvm.nvl.create is overloaded, so it can handle any integer type here,
    // but the NVL runtime must also accept unsigned long, which it can then
    // pass into system functions as a mode_t.
    //
    // Likewise, __builtin_nvl_create*, __builtin_nvl_recover*, and
    // __builtin_nvl_alloc_nv are overloaded on their other integer
    // parameters. We use size_t here, so the NVL runtime must also accept
    // size_t.
    //
    // Finally, __builtin_nvl_alloc_length is overloaded on its integer
    // return type. We use size_t here, the so the NVL runtime must also
    // return size_t.
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_nvl_create", "llvm.nvl.create.local",
      SRC_PARAMRET_NVL_HEAP_PTR, false,
      new SrcParamType[]{pcc, SRC_SIZE_T_TYPE, SrcUnsignedLongType},
      false, new boolean[]{false, true, true}, null));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_nvl_recover", "llvm.nvl.recover.local",
      SRC_PARAMRET_NVL_HEAP_PTR, false,
      new SrcParamType[]{pcc, SRC_SIZE_T_TYPE, SrcUnsignedLongType},
      false, new boolean[]{false, true, true}, null));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_nvl_open", "llvm.nvl.open.local",
      SRC_PARAMRET_NVL_HEAP_PTR, false, pcc));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_nvl_create_mpi", "llvm.nvl.create",
      SRC_PARAMRET_NVL_HEAP_PTR, false,
      new SrcParamType[]{pcc, SRC_SIZE_T_TYPE, SrcUnsignedLongType,
                         SRC_PARAM_MPI_GROUP},
      false, new boolean[]{false, true, true, false}, null));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_nvl_recover_mpi", "llvm.nvl.recover",
      SRC_PARAMRET_NVL_HEAP_PTR, false,
      new SrcParamType[]{pcc, SRC_SIZE_T_TYPE, SrcUnsignedLongType,
                         SRC_PARAM_MPI_GROUP},
      false, new boolean[]{false, true, true, false}, null));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_nvl_open_mpi", "llvm.nvl.open",
      SRC_PARAMRET_NVL_HEAP_PTR, false, pcc, SRC_PARAM_MPI_GROUP));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_nvl_close", "llvm.nvl.close",
      SrcVoidType, false, SRC_PARAMRET_NVL_HEAP_PTR));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_nvl_get_name", "llvm.nvl.get_name",
      pcc, false, SRC_PARAMRET_CONST_NVL_HEAP_PTR));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_nvl_set_root", "llvm.nvl.set_root",
      SrcVoidType, false, SRC_PARAMRET_NVL_HEAP_PTR,
      SRC_PARAM_PTR_TO_NVL, SRC_PARAM_TYPE_CHECKSUM_FROM_PTR_ARG));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_nvl_get_root", "llvm.nvl.get_root",
      SRC_RET_NVL_PTR_TYPE_FROM_TYPE_ARG, false,
      SRC_PARAMRET_CONST_NVL_HEAP_PTR,
      SRC_PARAM_TYPE_CHECKSUM_FROM_RET_TYPE));
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_nvl_alloc_nv", "llvm.nvl.alloc",
      SRC_RET_NVM_PTR_TYPE_FROM_TYPE_ARG, false,
      new SrcParamType[]{SRC_PARAMRET_NVL_HEAP_PTR, SRC_SIZE_T_TYPE,
                         SRC_SIZE_T_TYPE,
                         SRC_PARAM_METADATA_NULL_PTR_FROM_RET_TYPE,
                         SRC_PARAM_NULL_VOID_PTR},
      false, new boolean[]{false, true, true, false, false}, null));
    // TODO: Handle weak pointers.
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_nvl_alloc_length", "llvm.nvl.alloc.length.v2nv",
      SRC_SIZE_T_TYPE, false,
      new SrcParamType[]{SRC_PARAM_PTR_TO_NVL},
      true, new boolean[]{false}, null));
    // TODO: The nvl_bare_hack function is a temporary hack to let us expose
    // direct pointers. We should instead create a more careful nvl_bare
    // construct as follows:
    //
    //   nvl_bare (T1 *p1 = p1_nvl; T2 *p2 = p2_nvl;) {
    //     // scope of p1 and p2
    //   }
    //
    // p1 would have to have the same type as p1_nvl except for removal of
    // the target type's nvl qualifier (if the target type is an array, then
    // the nvl qualifier would have to be removed from the array). p1 would
    // be implicitly const, so that the compiler could assume it stores the
    // same address throughout the following block. There would be an
    // implicit vrefs inc/dec on p1_nvl before/after the block. There would
    // be an implicit persist call and write lock on the target object if
    // there's an enclosing transaction. T1 would have to be a plain C type
    // (thus, as for our current nvl_bare_hack function, p1_nvl cannot be a
    // pointer to a weak pointer). That is, it could not contain pointers
    // (necessarily to NVM) because they would be incorrectly treated as
    // V-to-NV pointers not NV-to-NV pointers for the sake of load, store,
    // ref counting, etc. nvl_alloc_nv could not be called within the block
    // because it might have to resize the heap, remap it, and thus
    // invalidate bare pointers. nvl_bare would clearly have some overhead,
    // but the main purpose is to amortize pointer overhead by using the
    // bare pointer repeatedly in the block. It's also useful for, for
    // example, calling printf on an NVM-stored char*, etc., so we might
    // want a read-only version, perhaps indicated by a const qualifier on
    // T1.
    //
    // (Of course, all of this would also apply to p2 and other declarations
    // in parens.)
    //
    // Also see related todo below for nvl_persist_hack function.
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_nvl_bare_hack", "llvm.nvl.bare",
      SRC_RET_BARE_PTR_TYPE_FROM_ARG, false, SRC_PARAM_PTR_TO_NVL));
    // TODO: The nvl_persist_hack function is a temporary hack to let users
    // persist an NVM store that is hidden from the compiler because the
    // store is via a pointer obtained from nvl_bare_hack. When we implement
    // the more careful nvl_bare construct described in the todo above, the
    // compiler will be able to track bare pointers and call persist
    // automatically. We haven't bothered to handle weak pointers yet.
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_nvl_persist_hack", "llvm.nvl.persist.v2nv",
      SrcVoidType, false,
      new SrcParamType[]{
        SRC_PARAM_PTR_TO_NVL, SRC_PARAM_METADATA_NULL_PTR_FROM_ARG_TYPE,
        SRC_SIZE_T_TYPE},
      false, new boolean[]{false, false, true}, null));
    // TODO: The nvl_nv2nv_to_v2nv_hack function is a temporary hack to let
    // users convert an NV-to-NV pointer to a V-to-NV pointer because it has
    // been loaded via a bare pointer. The first arg is the pointer to be
    // converted. The second arg is the NVM pointer (before conversion to a
    // bare pointer) via which it was loaded and it must not be a null
    // pointer. We haven't bothered to handle weak pointers yet. When we
    // implement the more careful nvl_bare construct described in the todo
    // above, the compiler will be able to track bare pointers to call this
    // automatically as needed.
    //
    // Be careful to call this immediately upon loading the pointer and
    // don't use the NV-to-NV form of the pointer in any other way.
    // That is, the call to nvl_nv2nv_to_v2nv_hack is the only way the
    // compiler can determine that the loaded pointer is actually in
    // NV-to-NV form. Thus, the compiler detects the case where the only
    // user of a loaded NVM pointer is nvl_nv2nv_to_v2nv_hack and then
    // suppresses V-to-NV automatic reference counting for the loaded
    // pointer until after the conversion. There is currently no way to
    // safely load a struct via a bare pointer if that struct contains an
    // NVM pointer. Instead, load the NVM pointer from the struct and call
    // nvl_nv2nv_to_v2nv_hack on the loaded pointer.
    addFunctionBuiltin(new SrcFunctionBuiltin(
      "__builtin_nvl_nv2nv_to_v2nv_hack",
      "llvm.nvl.nv2nv.to.v2nv",
      SRC_RET_NVM_PTR_FROM_ARG, false,
      SRC_PARAM_PTR_TO_NVL, SRC_PARAM_PTR_TO_NVL));
  }

  public void addParentFixup(Traversable node, Traversable parent) {
    parentFixupMap.put(node, parent);
  }

  public BuiltinTypeTable getBuiltinTypeTable() {
    return builtinTypeTable;
  }
  public void addTypedef(Declarator declarator, SrcType srcType) {
    typedefMap.put(declarator, srcType);
  }
  public void addEnum(Enumeration e, SrcEnumType t) {
    enumMap.put(e, t);
  }
  public SrcEnumType getEnum(Enumeration e) {
    return enumMap.get(e);
  }

  /**
   * Find or create the {@link SrcStructOrUnionType}, which stores a
   * {@link LLVMIdentifiedStructType}, for a named struct or union upon
   * encountering a definition, declaration, or reference for it.
   * 
   * <p>
   * See {@link StructOrUnionKey} for a discussion of the difficulty of
   * properly distinguishing different named struct or unions in the source.
   * </p>
   * 
   * <p>
   * The LLVM name for the resulting {@link LLVMIdentifiedStructType} is
   * based on the struct or union tag from the source code. However,
   * {@link LLVMIdentifiedStructType}s are stored in the {@link LLVMContext},
   * so they are shared among {@link LLVMModule}s (see the "Detailed
   * Description" at http://llvm.org/doxygen/classllvm_1_1StructType.html),
   * and the same struct or union tag might be used for different structs
   * and unions across multiple {@link TranslationUnit}s and across multiple
   * scopes within a {@link TranslationUnit}. Fortunately, LLVM appends a
   * suffix to the requested name of a {@link LLVMIdentifiedStructType} if
   * the name would collide with an existing name, but that means the name
   * of the returned {@link LLVMIdentifiedStructType} is not
   * straight-forward to predict. You can use
   * {@link LLVMIdentifiedStructType#getName()} to get the resulting name.
   * In case you need to look it up by that name, perhaps for debugging or
   * testing, you can use {@link LLVMModule#getTypeByName}, which looks it
   * up in the associated {@link LLVMContext}.
   * </p>
   * 
   * <p>
   * Keep in mind that LLVM will omit an {@link LLVMIdentifiedStructType}'s
   * definition in the output LLVM IR for an {@link LLVMModule} if there are
   * no uses of it in that {@link LLVMModule}.
   * </p>
   * 
   * @param isStruct
   *          true for a struct, false for a union
   * @param tag
   *          the struct or union tag
   * @param node
   *          the node that (1) contains the declaration, definition, or
   *          reference for the struct or union and, (2) along with its
   *          ancestor nodes, should be searched for a {@link SymbolTable}
   *          representing the scope of the struct or union
   * @param explicit
   *          true iff {@code node} contains an explicit definition or
   *          explicit declaration of the struct or union rather than an
   *          implicit declaration or a reference (see
   *          {@link StructOrUnionKey} for discussion of these kinds of
   *          uses)
   * @return the {@link SrcStructOrUnionType}
   */
  public SrcStructOrUnionType getSrcStructOrUnionType(
    boolean isStruct, String tag, Traversable node, boolean explicit,
    LLVMContext llvmContext)
  {
    SymbolTable symTableHere = lookupSymbolTable(node);
    // Return type if struct/union is already declared/defined here.
    StructOrUnionKey keyHere = new StructOrUnionKey(tag, symTableHere);
    SrcStructOrUnionType type = structOrUnionMap.get(keyHere);
    if (type != null)
      return type;
    // Unless this is an explicit declaration/definition, look for a
    // declaration/definition in an ancestor SymbolTable, and return the
    // type if found.
    if (!explicit) {
      for (SymbolTable parent = lookupSymbolTable(symTableHere.getParent());
           parent != null;
           parent = lookupSymbolTable(parent.getParent()))
      {
        StructOrUnionKey key = new StructOrUnionKey(tag, parent);
        type = structOrUnionMap.get(key);
        if (type != null)
          return type;
      }
    }
    // Create it here.
    if (isStruct)
      type = new SrcStructType(tag, llvmContext);
    else
      type = new SrcUnionType(tag, llvmContext);
    structOrUnionMap.put(keyHere, type);
    return type;
  }

  private void addFunctionBuiltin(SrcFunctionBuiltin b) {
    srcFunctionBuiltinMap.put(b.getName(), b);
  }
  public SrcFunctionBuiltin getFunctionBuiltin(String name) {
    return srcFunctionBuiltinMap.get(name);
  }

  public void addLocalFnDecl(Declarator declarator, SrcType srcType,
                             LLVMValue fn, JumpScopeTable jumpScopeTable)
  {
    final ValueAndType valueAndType = new ValueAndType(fn, srcType, true);
    localMap.put(declarator, valueAndType);
    jumpScopeTable.addSymbol(declarator, valueAndType, false);
  }
  public void addLocalExternVar(Declarator declarator,
                                SrcType srcType, LLVMValue value,
                                JumpScopeTable jumpScopeTable)
  {
    srcType.checkStaticOrAutoObject(
      "variable \""+declarator.getID().getName()+"\"", false);
    final ValueAndType valueAndType = new ValueAndType(value, srcType, true);
    localMap.put(declarator, valueAndType);
    jumpScopeTable.addSymbol(declarator, valueAndType, false);
  }
  public ValueAndType addStaticLocalVar(
    LLVMModule llvmModule, Declarator declarator, String llvmName,
    SrcType srcType, boolean storageAllocated, boolean explicitInit,
    boolean hasThread, JumpScopeTable jumpScopeTable,
    LLVMGlobalVariable llvmVar)
  {
    final LLVMType llvmType = srcType.getLLVMType(llvmModule.getContext());
    srcType.checkStaticOrAutoObject(
      "variable \""+declarator.getID().getName()+"\"", storageAllocated);
    if (!explicitInit)
      srcType.checkAllocWithoutExplicitInit(true);
    if (llvmVar == null)
      llvmVar = llvmModule.addGlobal(llvmType, llvmName);
    llvmVar.setLinkage(LLVMInternalLinkage);
    if (hasThread)
      llvmVar.setThreadLocal(true);
    if (!explicitInit)
      llvmVar.setInitializer(LLVMConstant.constNull(llvmType));
    final ValueAndType valueAndType
      = new ValueAndType(llvmVar, srcType, true);
    localMap.put(declarator, valueAndType);
    jumpScopeTable.addSymbol(declarator, valueAndType, false);
    return valueAndType;
  }
  public ValueAndType addNonStaticLocalVar(
    LLVMModule llvmModule, LLVMInstructionBuilder llvmAllocaBuilder,
    String what, Declarator declarator, String llvmName,
    SrcType srcType, boolean storageAllocated, boolean explicitInit,
    JumpScopeTable jumpScopeTable, LLVMStackAllocation llvmVar)
  {
    final LLVMType llvmType = srcType.getLLVMType(llvmModule.getContext());
    srcType.checkStaticOrAutoObject(
      what == null ? "variable \""+declarator.getID().getName()+"\"" : what,
      storageAllocated);
    if (!explicitInit)
      srcType.checkAllocWithoutExplicitInit(false);
    if (llvmVar == null)
      llvmVar = new LLVMStackAllocation(llvmAllocaBuilder, llvmName, llvmType,
                                        null);
    final ValueAndType valueAndType
      = new ValueAndType(llvmVar, srcType, true);
    localMap.put(declarator, valueAndType);
    jumpScopeTable.addSymbol(declarator, valueAndType, true);
    return valueAndType;
  }
  public ValueAndType getLocal(Symbol sym) {
    if (!(sym instanceof Declarator))
      return null;
    return getLocal((Declarator)sym);
  }
  public ValueAndType getLocal(Declarator declarator) {
    return localMap.get(declarator);
  }

  public ValueAndType addGlobalVar(
    LLVMModule llvmModule, int llvmModuleIndex, String srcName,
    String llvmVarName, String llvmVarTmpName, SrcType srcType,
    boolean storageAllocatedHere, boolean hasThread,
    LLVMGlobalVariable llvmVar)
  {
    srcType.checkStaticOrAutoObject("variable \""+srcName+"\"",
                                    storageAllocatedHere);
    if (llvmVar == null)
      llvmVar = llvmModule.addGlobal(
        srcType.getLLVMType(llvmModule.getContext()), llvmVarTmpName);
    if (hasThread)
      llvmVar.setThreadLocal(true);
    final ValueAndType var = new ValueAndType(llvmVar, srcType, true);
    globalVarMaps[llvmModuleIndex].put(llvmVarName, var);
    return var;
  }
  public ValueAndType getGlobalVar(int moduleIndex, String name) {
    return globalVarMaps[moduleIndex].get(name);
  }
  public void recordGlobalVarWithTentativeDef(int moduleIndex, String name) {
    globalVarTentativeDefSets[moduleIndex].add(name);
  }
  public void recordGlobalVarWithoutTentativeDef(int moduleIndex, String name)
  {
    globalVarTentativeDefSets[moduleIndex].remove(name);
  }
  public void replaceLLVMGlobalVariable(
    LLVMGlobalVariable oldVar, LLVMGlobalValue newVar, String name)
  {
    if (oldVar == null || oldVar == newVar)
      return;
    oldVar.replaceAllUsesWith(
      LLVMConstantExpression.bitCast(newVar, oldVar.typeOf()));
    oldVar.delete();
    // LLVM appended a suffix because the original name was taken, so
    // reset the name now that the original name is not taken.
    newVar.setValueName(name);
  }
  public void initTentativeDefinitions(int llvmModuleIndex,
                                       LLVMModule llvmModule)
  {
    final LLVMContext llvmContext = llvmModule.getContext();
    // ISO C99 sec. 6.9.2p2: adjust tentative definitions to have implicit
    // init 0.
    for (final String llvmName : globalVarTentativeDefSets[llvmModuleIndex]) {
      final ValueAndType oldVar = getGlobalVar(llvmModuleIndex, llvmName);
      final LLVMGlobalVariable oldLLVMVar
        = (LLVMGlobalVariable)oldVar.getLLVMValue();
      SrcType srcType = oldVar.getSrcType();
      // ISO C99 sec. 6.9.2p5 (based on 6.9.2p2). Also, see
      // http://compgroups.net/comp.std.c/tentative-array-object-definition/2616181
      {
        final SrcArrayType srcArrayType = srcType.toIso(SrcArrayType.class);
        if (srcArrayType != null && !srcArrayType.numElementsIsSpecified())
          srcType = SrcArrayType.get(srcArrayType.getElementType(), 1);
      }
      final ValueAndType newVar
        = addGlobalVar(llvmModule, llvmModuleIndex, llvmName, llvmName,
                       llvmName, srcType, true, oldLLVMVar.isThreadLocal(),
                       null);
      final LLVMGlobalVariable newLLVMVar
        = (LLVMGlobalVariable)newVar.getLLVMValue();
      newLLVMVar.setInitializer(
        LLVMConstant.constNull(srcType.getLLVMType(llvmContext)));
      srcType.checkAllocWithoutExplicitInit(true);
      newLLVMVar.setLinkage(oldLLVMVar.getLinkage());
      replaceLLVMGlobalVariable(oldLLVMVar, newLLVMVar, llvmName);
    }
  }

  public void addFunction(int moduleIndex, String name, ValueAndType func) {
    functionMaps[moduleIndex].put(name, func);
  }
  public ValueAndType getFunction(int moduleIndex, String name) {
    return functionMaps[moduleIndex].get(name);
  }
  public void setFunctionInlineDefState(int moduleIndex, String name,
                                        InlineDefState state)
  {
    functionInlineDefStateMaps[moduleIndex].put(name, state);
  }
  public InlineDefState getFunctionInlineDefState(int moduleIndex,
                                                  String name)
  {
    return functionInlineDefStateMaps[moduleIndex].get(name);
  }
  public void pruneUnusedFunctions(int moduleIndex) {
    // ISO C99 sec. 6.9p3 permits identifiers with internal linkage that are
    // not used in any expression (sizeof expressions don't count, but those
    // uses don't appear in the LLVM IR) to be left undefined, but LLVM
    // module verification will fail for such a function.
    for (Map.Entry<String,ValueAndType> entry
         : functionMaps[moduleIndex].entrySet())
    {
      final LLVMFunction fn = (LLVMFunction)entry.getValue().getLLVMValue();
      if (fn.getLinkage() != LLVMInternalLinkage || !fn.isDeclaration())
        continue;
      if (fn.getFirstUse().getInstance() != null)
        throw new SrcRuntimeException(
          "function has internal linkage and is used but is never defined: "
          + entry.getKey());
      functionMaps[moduleIndex].remove(entry);
      fn.delete();
    }
  }

  public void addLabel(Label label, LLVMBasicBlock bb) {
    labelMap.put(label, bb);
  }
  public void addCase(Case case_, LLVMBasicBlock bb) {
    caseMap.put(case_, bb);
  }
  public void addDefault(Default default_, LLVMBasicBlock bb) {
    defaultMap.put(default_, bb);
  }
  public LLVMBasicBlock getLabelBasicBlock(Label label) {
    return labelMap.get(label);
  }
  public LLVMBasicBlock getCaseBasicBlock(Case case_) {
    return caseMap.get(case_);
  }
  public LLVMBasicBlock getDefaultBasicBlock(Default default_) {
    return defaultMap.get(default_);
  }

  /**
   * Find the immediately enclosing symbol table of a reference to a typedef
   * name, struct, union, enum, or identifier.
   * 
   * <p>
   * This method uses {@link #parentFixupMap} where other parent-child
   * relationships are missing in the Cetus IR. For example, a
   * {@link ProcedureDeclarator} or {@link NestedDeclarator} that has no
   * parent is often the child of a {@link Procedure}, or it's part of some
   * type operand.
   * </p>
   * 
   * <p>
   * TODO: However, Cetus handles an explicit struct/union/enum definition
   * in a parameter list incorrectly: it inserts a {@link ClassDeclaration}
   * or {@link Enumeration} for it as a child of the {@link TranslationUnit}
   * (or whatever encloses it), so it resolves the scope as the enclosing
   * scope. ISO C99 says the scope should instead be the function definition
   * (or the function parameter list if there is no function definition).
   * For an implicit struct/union declaration (whether in a parameter list
   * or not), Cetus doesn't create a {@link ClassDeclaration} or insert
   * anything into a symbol table, so it doesn't choose a scope. For
   * consistency with Cetus's handling of the first case, we always skip the
   * {@link Procedure} symbol table when resolving the scope of a
   * struct/union/enum. We do the same when resolving the scope of a typedef
   * name, but that's fine as a typedef declaration should never appear in a
   * parameter list. Ultimately Cetus should be fixed and then skipping the
   * {@link Procedure} symbol table will never be the right implementation
   * here.
   * </p>
   * 
   * <p>
   * TODO: When resolving the scope of an identifier, one parameter type
   * might reference another parameter. For example, sizeof for another
   * parameter might be used in array dimensions, or the other parameter's
   * value might be used directly in the case of a variable-length array.
   * Thus, we should not skip the {@link Procedure} symbol table when
   * looking up identifiers. Actually, when any kind of declarator
   * represents a function prototype without a definition, we might need to
   * resolve identifiers in the same way, but Cetus does not create a symbol
   * table for parameters then. Alas, we do not yet have any workaround to
   * enable parameter types to reference other parameters regardless of
   * whether there's a function definition. In summary, here are the
   * problems: (1) {@link ProcedureDeclarator} not {@link Procedure} should
   * be a {@link SymbolTable} so that Cetus can place parameters in a
   * {@link SymbolTable} even when there's no function definition, (2)
   * {@link ProcedureDeclarator} should never be parent-less or Cetus will
   * skip its parameters when performing symbol linking, (3) I'm not sure
   * what {@link LLVMValue} to create for a parameter when there's no
   * definition as I normally would create a stack allocation. If it turns
   * out some code needs this ability, then we can try to fix all this then.
   * </p>
   * 
   * <p>
   * Also, this method skips any {@link ClassDeclaration}, which does not
   * represent a scope in C, unlike C++.
   * </p>
   * 
   * @param node
   *          the node that the symbol table should be or should enclose, or
   *          null if none
   * @return the symbol table
   */
  public SymbolTable lookupSymbolTable(Traversable node) {
    Traversable t = node;
    while (t != null) {
      if (t instanceof SymbolTable && !(t instanceof ClassDeclaration))
        return (SymbolTable)t;
      Traversable parent = t.getParent() != null
                           ? t.getParent() : parentFixupMap.get(t);
      if (parent instanceof Procedure)
        parent = parent.getParent();
      if (parent == null && !(t instanceof Program))
        throw new IllegalStateException();
      t = parent;
    }
    return null;
  }

  /**
   * Convert a list of type specifiers and type qualifiers to a
   * {@link SrcType}.
   * 
   * @param specifiers
   *          the list of type specifiers and type qualifiers. In this list,
   *          it is fine to build any {@link UserSpecifier} (that is,
   *          typedef name, enum, struct, or union) using
   *          {@link SymbolTools#getOrphanID} in order to look up a fixed
   *          name (like {@code MPI_Group}) because only the name is then
   *          checked here.
   * @param parent
   *          the parent node of the declaration or type expression
   *          containing the specifiers
   * @param llvmModule
   *          the LLVM module into which any required declarations should be
   *          generated
   * @param warningsAsErrors
   *          whether to report warnings as errors
   * @return the {@link SrcType} for {@code specifiers}
   * @throws SrcRuntimeException
   *           if {@code specifiers} is empty or type is invalid
   */
  public SrcType typeSpecifiersAndQualifiersToSrcType(
    List<Specifier> specifiers, Traversable parent,
    LLVMModule llvmModule, boolean warningsAsErrors)
  {
    // This could be an implicit int, which is not permitted in C99. It
    // could also be an old-style parameter list, which we do not support.
    if (specifiers.isEmpty())
      throw new UnsupportedOperationException(
        "declaration without type specifiers is not supported");

    final LLVMContext llvmContext = llvmModule.getContext();

    // Specifier.WCHAR_T supposedly represents wchar_t. However, ISO C99 says
    // wchar_t is not a keyword but is defined in stddef.h. So far, as
    // expected, Cetus always parses wchar_t as a typedef name
    // (UserSpecifier) when stddef.h is included, and it reports a parsing
    // error if stddef.h is not included. Thus, we don't handle
    // Specifier.WCHAR_T.
    final SrcSymbolTable.TypeSpec typeSpec = new SrcSymbolTable.TypeSpec();
    final List<Specifier> typeQualifierList = new ArrayList<>();
    for (Specifier s : specifiers) {
      if (s == Specifier.VOID)          typeSpec.addVoid();
      else if (s == Specifier.CBOOL)    typeSpec.addBool();
      else if (s == Specifier.CHAR)     typeSpec.addChar();
      else if (s == Specifier.INT)      typeSpec.addInt();
      else if (s == Specifier.FLOAT)    typeSpec.addFloat();
      else if (s == Specifier.DOUBLE)   typeSpec.addDouble();
      else if (s == Specifier.SHORT)    typeSpec.addShort();
      else if (s == Specifier.LONG)     typeSpec.addLong();
      else if (s == Specifier.SIGNED)   typeSpec.addSigned();
      else if (s == Specifier.UNSIGNED) typeSpec.addUnsigned();
      else if (s == Specifier.CCOMPLEX || s == Specifier.CIMAGINARY) {
        // TODO: Handle CCOMPLEX and CIMAGINARY.
        throw new UnsupportedOperationException(
          "complex types are not yet supported");
      }
      else if (s instanceof UserSpecifier) {
        final String name = ((UserSpecifier)s).getIDExpression().getName();
        final int space = name.indexOf(' ');
        final String keyword = space == -1 ? "" : name.substring(0, space);
        final String tag = name.substring(space + 1);
        switch (keyword) {
        case "": {
          // No keyword, so must be a typedef name.
          SrcType userType = builtinTypeTable.lookup(name, llvmModule,
                                                     warningsAsErrors);
          if (userType == null) {
            for (SymbolTable symTable = lookupSymbolTable(parent);
                 userType == null && symTable != null;
                 symTable = lookupSymbolTable(symTable.getParent()))
            {
              for (Symbol sym : symTable.getSymbols()) {
                if (sym.getSymbolName().equals(name)) {
                  userType = typedefMap.get(sym);
                  // If sym is not in typedefMap, sym must be a symbol defined
                  // later in this scope, so keep searching enclosing scopes.
                  // Otherwise, userType != null, so we'll stop searching.
                  break;
                }
              }
            }
            if (userType == null)
              throw new SrcRuntimeException("unknown typedef name: " + name);
          }
          typeSpec.addUserType(userType);
          break;
        }
        case "enum": {
          SrcType userType = null;
          for (SymbolTable symTable = lookupSymbolTable(parent);
               userType == null && symTable != null;
               symTable = lookupSymbolTable(symTable.getParent()))
          {
            for (Declaration decl : symTable.getDeclarations()) {
              if (decl instanceof Enumeration) {
                Enumeration enumDecl = (Enumeration)decl;
                if (enumDecl.getName().getName().equals(tag)) {
                  userType = enumMap.get(enumDecl);
                  // If sym is not in enumMap, sym must be a symbol defined
                  // later in this scope, so keep searching enclosing scopes.
                  // Otherwise, userType != null, so we'll stop searching.
                  break;
                }
              }
            }
          }
          if (userType == null)
            throw new SrcRuntimeException("unknown enum: " + tag);
          typeSpec.addUserType(userType);
          break;
        }
        case "struct":
          // In all cases, we look up the struct or union starting at the
          // parent of the declaration. A Procedure is a SymbolTable, but not
          // for the specifiers appearing in the Procedure's return type, so
          // we must start the lookup at the Procedure's parent not at the
          // Procedure. A VariableDeclaration is not a SymbolTable, so
          // starting at its parent is equivalent to starting at it.
          //
          // A UserSpecifier is an implicit declaration or a reference to a
          // struct or union, so the last argument here is false.
          typeSpec.addUserType(getSrcStructOrUnionType(true, tag, parent,
                                                       false, llvmContext));
          break;
        case "union":
          typeSpec.addUserType(getSrcStructOrUnionType(false, tag, parent,
                                                       false, llvmContext));
          break;
        default:
          throw new SrcRuntimeException("unknown keyword: " + keyword);
        }
      }
      else
        // SrcQualifiedType.get call below throws an exception if specifier is
        // not a type qualifier.
        typeQualifierList.add(s);
    }
    final Specifier[] typeQualifiers
      = typeQualifierList.toArray(new Specifier[0]);

    // This list of possible type specifier lists comes from ISO C99 sec.
    // 6.7.2p2.
    final SrcSymbolTable.TypeSpec cmp = new SrcSymbolTable.TypeSpec();
    final SrcType unqualifiedType;
    if      (   typeSpec.cmp(cmp.clear().addVoid()))
      unqualifiedType = SrcVoidType;
    else if (   typeSpec.cmp(cmp.clear().addChar()))
      unqualifiedType = SrcCharType;
    else if (   typeSpec.cmp(cmp.clear().addSigned().addChar()))
      unqualifiedType = SrcSignedCharType;
    else if (   typeSpec.cmp(cmp.clear().addUnsigned().addChar()))
      unqualifiedType = SrcUnsignedCharType;
    else if (   typeSpec.cmp(cmp.clear().addShort())
             || typeSpec.cmp(cmp.clear().addSigned().addShort())
             || typeSpec.cmp(cmp.clear().addShort().addInt())
             || typeSpec.cmp(cmp.clear().addSigned().addShort().addInt()))
      unqualifiedType = SrcShortType;
    else if (   typeSpec.cmp(cmp.clear().addUnsigned().addShort())
             || typeSpec.cmp(cmp.clear().addUnsigned().addShort().addInt()))
      unqualifiedType = SrcUnsignedShortType;
    else if (   typeSpec.cmp(cmp.clear().addInt())
             || typeSpec.cmp(cmp.clear().addSigned())
             || typeSpec.cmp(cmp.clear().addSigned().addInt()))
      unqualifiedType = SrcIntType;
    else if (   typeSpec.cmp(cmp.clear().addUnsigned())
             || typeSpec.cmp(cmp.clear().addUnsigned().addInt()))
      unqualifiedType = SrcUnsignedIntType;
    else if (   typeSpec.cmp(cmp.clear().addLong())
             || typeSpec.cmp(cmp.clear().addSigned().addLong())
             || typeSpec.cmp(cmp.clear().addLong().addInt())
             || typeSpec.cmp(cmp.clear().addSigned().addLong().addInt()))
      unqualifiedType = SrcLongType;
    else if (   typeSpec.cmp(cmp.clear().addUnsigned().addLong())
             || typeSpec.cmp(cmp.clear().addUnsigned().addLong().addInt()))
      unqualifiedType = SrcUnsignedLongType;
    else if (   typeSpec.cmp(cmp.clear().addLong().addLong())
             || typeSpec.cmp(cmp.clear().addSigned().addLong().addLong())
             || typeSpec.cmp(cmp.clear().addLong().addLong().addInt())
             || typeSpec.cmp(cmp.clear().addSigned().addLong().addLong().addInt()))
      unqualifiedType = SrcLongLongType;
    else if (   typeSpec.cmp(cmp.clear().addUnsigned().addLong().addLong())
             || typeSpec.cmp(cmp.clear().addUnsigned().addLong().addLong().addInt()))
      unqualifiedType = SrcUnsignedLongLongType;
    else if (   typeSpec.cmp(cmp.clear().addFloat()))
      unqualifiedType = SrcFloatType;
    else if (   typeSpec.cmp(cmp.clear().addDouble()))
      unqualifiedType = SrcDoubleType;
    else if (   typeSpec.cmp(cmp.clear().addLong().addDouble()))
      unqualifiedType = SrcLongDoubleType;
    else if (   typeSpec.cmp(cmp.clear().addBool()))
      unqualifiedType = SrcBoolType;
    else if (   typeSpec.cmpNonUser(cmp.clear())
             && typeSpec.getUserTypes().size() == 1)
      unqualifiedType = typeSpec.getUserTypes().get(0);
    else {
      // complain about unknown type qualifiers first because they might be
      // some sort of unknown type specifiers
      SrcQualifiedType.get(SrcIntType, typeQualifiers);
      throw new SrcRuntimeException(
        "invalid combination of type specifiers: " + specifiers);
    }
    final SrcType res = SrcQualifiedType.get(unqualifiedType, typeQualifiers);

    // If a use of a struct type S is not explicitly NVM-stored, complain if
    // the struct type contains any NVM storage.
    if (res.iso(SrcStructType.class)
        && !res.hasEffectiveQualifier(SrcTypeQualifier.NVL))
    {
      new SrcStorageHasNoNVMCheck(
        res, "non-NVM-stored struct type has NVM-stored member")
      .run();
    }

    return res;
  }

  /**
   * A key that is unique for each named struct or union in the source code.
   * (Cetus adds tags to tag-less structs and unions, so that case need not be
   * handled specially.) This is intended to be used only by
   * {@link #structOrUnionMap} and {@link #getSrcStructOrUnionType}.
   * 
   * <p>
   * Named structs or unions are distinct if their tags are distinct or if
   * their scopes are different. Thus, a {@link StructOrUnionKey} stores the
   * tag (compared by value) and the scope's {@link SymbolTable} (compared by
   * reference). It is used by {@link #structOrUnionMap} to map a struct or
   * union to its corresponding {@link LLVMIdentifiedStructType}.
   * </p>
   * 
   * <p>
   * The scope of a struct or union is the scope in which it was explicitly
   * defined (that is, with a member list), explicitly declared (that is, by
   * itself like "{@code struct foo;}"), or implicitly declared (referenced
   * before it's ever explicitly declared or defined). For an implicit
   * declaration of a struct or union, Cetus does not generate a
   * {@link ClassDeclaration} node (or even insert a symbol into a
   * {@link SymbolTable}), so {@link ClassDeclaration} cannot replace
   * {@link StructOrUnionKey} as the key. TODO: If that problem is one day
   * fixed in Cetus, scope analysis in {@link #getSrcStructOrUnionType}
   * could trivially depend on Cetus's symbol lookup, but keep in mind that
   * Cetus generates more than one {@link ClassDeclaration} when there are
   * multiple declarations or definitions for a struct or union, so care
   * would have to be taken to consistently use the one found by symbol
   * lookup.
   * </p>
   */
  private static final class StructOrUnionKey {
    /**
     * Construct a key for a struct or union.
     * 
     * @param tag
     *          the tag of the struct or union
     * @param scope
     *          the {@link SymbolTable} for the scope in which the struct or
     *          union was declared or defined
     */
    public StructOrUnionKey(String tag, SymbolTable scope) {
      this.tag = tag;
      this.scope = scope;
    }
    @Override
    public boolean equals(Object obj) {
      if (this == obj)
        return true;
      if (obj == null || getClass() != obj.getClass())
        return false;
      StructOrUnionKey other = (StructOrUnionKey)obj;
      if (scope != other.scope)
        return false;
      if (tag == null)
        return other.tag == null;
      return tag.equals(other.tag);
    }
    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime*result + ((tag == null) ? 0 : tag.hashCode());
      result = prime*result + ((tag == null) ? 0 : tag.hashCode());
      return result;
    }
    final private String tag;
    final private SymbolTable scope;
  }

  /** Helper class for {@link #typeSpecifiersToSrcType}. */
  private static final class TypeSpec {
    public TypeSpec() {
      clear();
    }
    public TypeSpec clear() {
      voids = 0;
      bools = 0;
      chars = 0;
      ints = 0;
      floats = 0;
      doubles = 0;
      shorts = 0;
      longs = 0;
      signeds = 0;
      unsigneds = 0;
      userTypes = new ArrayList<>(1);
      return this;
    }

    private int voids;
    private int bools;
    private int chars;
    private int ints;
    private int floats;
    private int doubles;
    private int shorts;
    private int longs;
    private int signeds;
    private int unsigneds;
    public TypeSpec addVoid()     { ++voids;     return this; }
    public TypeSpec addBool()     { ++bools;     return this; }
    public TypeSpec addChar()     { ++chars;     return this; }
    public TypeSpec addInt()      { ++ints;      return this; }
    public TypeSpec addFloat()    { ++floats;    return this; }
    public TypeSpec addDouble()   { ++doubles;   return this; }
    public TypeSpec addShort()    { ++shorts;    return this; }
    public TypeSpec addLong()     { ++longs;     return this; }
    public TypeSpec addSigned()   { ++signeds;   return this; }
    public TypeSpec addUnsigned() { ++unsigneds; return this; }

    private List<SrcType> userTypes;
    public void addUserType(SrcType type) { userTypes.add(type); }
    List<SrcType> getUserTypes() { return userTypes; }

    /**
     * Returns true iff the non-user portion of two type specifications are
     * exactly the same.
     */
    public boolean cmpNonUser(TypeSpec other) {
      if (other == null) return false;
      if (voids != other.voids) return false;
      if (bools != other.bools) return false;
      if (chars != other.chars) return false;
      if (doubles != other.doubles) return false;
      if (floats != other.floats) return false;
      if (ints != other.ints) return false;
      if (longs != other.longs) return false;
      if (shorts != other.shorts) return false;
      if (signeds != other.signeds) return false;
      if (unsigneds != other.unsigneds) return false;
      return true;
    }

    /**
     * Returns true iff the non-user portion of two type specifications are
     * exactly the same and neither contains user type specifications.
     */
    public boolean cmp(TypeSpec other) {
      assert(other.userTypes.isEmpty());
      return cmpNonUser(other) && userTypes.isEmpty();
    }
  }

  public void checkNVMStoredStructs() {
    // TODO: Ideally, we wouldn't waste time checking struct types for
    // previous translation units, but currently we index all struct types
    // together, so it's easiest just to recheck them all.
    for (SrcStructOrUnionType structOrUnion : structOrUnionMap.values()) {
      if (structOrUnion.hasNVMStoredUses() && structOrUnion.isIncompleteType())
        throw new SrcRuntimeException(
          "NVM-stored struct type is incomplete in translation unit");
    }
  }

  /**
   * Get a pointer to the global variable that will hold the checksum for
   * the specified type.
   * 
   * @param srcType
   *          the type for which the checksum will be computed
   * @param llvmModule
   *          the module into which to insert the global variable. If this
   *          method has previously been called for the same type and
   *          module, then the global variable previously inserted will be
   *          used instead.
   * @param llvmModuleIndex
   *          index of {@code llvmModule} within the array of modules being
   *          constructed by {@link BuildLLVM}
   * @param llvmBuilder
   *          the IR builder with which any LLVM instructions should be
   *          generated
   * @return a pointer, of type {@code i8*}, to the first element of the
   *         global variable. The global variable's initializer is zero
   *         until until the end of the translation unit (when
   *         {@link #initializeTypeChecksumVariables} is called) after all
   *         struct definitions have been seen. Otherwise, the checksum
   *         might not encode the full type.
   */
  public LLVMValue computeTypeChecksumPointer(
    SrcType srcType, LLVMModule llvmModule, int llvmModuleIndex,
    LLVMInstructionBuilder llvmBuilder)
  {
    final LLVMContext ctxt = llvmModule.getContext();
    final int checksumLength = getMessageDigest().getDigestLength();
    final LLVMArrayType arrayType
      = SrcArrayType.get(SrcCharType, checksumLength).getLLVMType(ctxt);
    final LLVMGlobalVariable varOld = typeChecksumMap[llvmModuleIndex]
                                      .get(srcType);
    final LLVMGlobalVariable var;
    if (varOld != null)
      var = varOld;
    else {
      var = llvmModule.addGlobal(arrayType, ".typeChecksum");
      // These three properties mimic clang (clang-600.0.56):
      // private unnamed_addr constant
      var.setLinkage(LLVMLinkage.LLVMPrivateLinkage);
      var.setConstant(true);
      // TODO: When we have an LLVM whose C API supports it, extend jllvm
      // accordingly, uncomment this, and add check for it in the test
      // suite. It's available by at least LLVM 3.5.0 as
      // LLVMSetUnnamedAddr.
      // See: http://lists.cs.uiuc.edu/pipermail/llvm-commits/Week-of-Mon-20140310/208071.html
      // See similar todo for __FUNCTION__ in BuildLLVM.
      //var.setUnnamedAddr(true);
      var.setInitializer(LLVMConstant.constNull(arrayType));
      typeChecksumMap[llvmModuleIndex].put(srcType, var);
    }
    final LLVMConstant zero = LLVMConstant.constNull(LLVMIntegerType
                                                     .get(ctxt, 32));
    return LLVMGetElementPointerInstruction.create(llvmBuilder, "", var,
                                                   zero, zero);
  }

  /**
   * Set initializers for all type checksum global variables created by
   * {@link #computeTypeChecksumPointer} for the specified module.
   * 
   * <p>
   * This method must be called at the end of the specified module's source
   * translation unit after all struct type definitions have been
   * translated.
   * </p>
   * 
   * @param llvmModule
   *          the module containing the type checksum global variables to be
   *          initialized
   * @param llvmModuleIndex
   *          index of {@code llvmModule} within the array of modules being
   *          constructed by {@link BuildLLVM}
   */
  public void initializeTypeChecksumVariables(LLVMModule llvmModule,
                                              int llvmModuleIndex)
  {
    final LLVMContext ctxt = llvmModule.getContext();
    for (Iterator<Entry<SrcType, LLVMGlobalVariable>> itr
           = typeChecksumMap[llvmModuleIndex].entrySet().iterator();
         itr.hasNext();)
    {
      Entry<SrcType, LLVMGlobalVariable> entry = itr.next();
      final SrcType srcType = entry.getKey();
      final LLVMGlobalVariable var = entry.getValue();
      final String compatStr = srcType.toCompatibilityString(ctxt);
      final MessageDigest messageDigest = getMessageDigest();
      final byte[] compatStrBytes;
      try {
        compatStrBytes = compatStr.getBytes(TYPE_COMPATSTR_ENCODING);
      }
      catch (UnsupportedEncodingException e) {
        throw new IllegalStateException("failure computing type checksum",
                                        e);
      }
      final byte[] checksum = messageDigest.digest(compatStrBytes);
      final LLVMIntegerType elementType = LLVMIntegerType.get(ctxt, 8);
      final LLVMConstant[] elements = new LLVMConstant[checksum.length];
      if (debugTypeChecksums)
        System.err.print("type checksum: var=<"+var.getValueName()
                         +">, string=<"+compatStr+">, checksum=<");
      for (int i = 0; i < checksum.length; ++i) {
        elements[i] = LLVMConstantInteger.get(elementType, checksum[i],
                                              false);
        if (debugTypeChecksums)
          System.err.print(String.format("%02x", checksum[i]));
      }
      if (debugTypeChecksums)
        System.err.println(">");
      final LLVMConstant init = LLVMConstantArray.get(elementType,
                                                      elements);
      var.setInitializer(init);
    }
  }
}
