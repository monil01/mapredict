package openacc.codegen.llvmBackend;

import org.jllvm.LLVMModule;
import org.jllvm.LLVMType;

import cetus.hir.Traversable;


/**
 * The LLVM backend's base type for source types each of whose set of
 * possible roles must include at least one of parameter type or return
 * type. In other words, the set of source types included is the union of
 * the sets included by {@link SrcParamType} and {@link SrcReturnType}.
 * 
 * <p>
 * WARNING: Be careful when adding new types to this hierarchy. To permit
 * multiple inheritance of both interface and implementation despite Java's
 * limitations on the latter, several types that are logically abstract
 * classes (because they provide some implementation to be inherited) are
 * defined as interfaces instead, and each has an abstract static nested
 * class called {@code Impl} containing the implementation. Here's how to
 * use the type hierarchy:
 * </p>
 * <ul>
 *   <li>
 *     The only diamond, which contains the only case of multiple
 *     inheritance, is at the top of the logical type hierarchy:
 *     <ul>
 *       <li> {@link SrcParamOrReturnType} = top of the diamond</li>
 *       <li> {@link SrcParamType} and {@link SrcReturnType}</li>
 *       <li> {@link SrcParamAndReturnType} = bottom of the diamond</li>
 *       <li> {@link SrcType}</li>
 *       <li> many more types</li>
 *     </ul>
 *   </li>
 *   <li>
 *     Each type within the diamond is defined as an interface and has an
 *     {@code Impl} abstract static nested class. For example,
 *     {@link SrcParamOrReturnType} has {@link SrcParamOrReturnType.Impl}.
 *   </li>
 *   <li>
 *     To add a new method to a type within the diamond, declare it in the
 *     interface, and implement it in the {@code Impl} class. If you're
 *     adding a method to {@link SrcParamType} or {@link SrcReturnType},
 *     then you'll also have to add some implementation to
 *     {@link SrcParamAndReturnType.Impl}, which inherits implementation
 *     only from {@link SrcParamOrReturnType.Impl}.
 *   </li>
 *   <li>
 *     To add a new type below the diamond, extend from either
 *     {@link SrcParamType.Impl}, {@link SrcReturnType.Impl}, or
 *     {@link SrcParamAndReturnType.Impl}.
 *   </li>
 *   <li>
 *     Hopefully we'll never need to extend the diamond or add additional
 *     cases of multiple inheritance.
 *   </li>
 * </ul>
 * 
 * <p>
 * Currently, this type is used solely to support function builtins (see
 * {@link SrcFunctionBuiltin}).
 * </p>
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public interface SrcParamOrReturnType {
  /**
   * For a call where this type is a parameter/return type, get the
   * parameter/return type as a {@link SrcType}.
   * 
   * @param callNode
   *          the AST node for the call. Used as the context for any special
   *          symbol table lookups.
   * @param calleeName
   *          name to use for the callee in error messages. Usually it's
   *          something like "foo" or "foo pragma".
   * @param paramName
   *          name to use for the actual argument in error messages, or null
   *          if the type is being used as a return type. Usually it's
   *          something like "argument 2" or "foo clause".
   * @param args
   *          the actual args passed to the call
   * @param typeArg
   *          any type argument passed to the call, or null if none. Only
   *          one type argument is currently supported.
   * @param srcSymbolTable
   *          the {@link SrcSymbolTable} for the C source
   * @param llvmModule
   *          the LLVM module into which any required declarations should be
   *          generated
   * @param warningsAsErrors
   *          whether to report warnings as errors
   * @return the parameter/return type as a {@link SrcType}, or null if it
   *         cannot be represented as a {@link SrcType} (must not be the
   *         case for a return type) or cannot be computed because of
   *         invalid arguments, which must be guaranteed to be caught during
   *         argument validation against the formal parameter specification.
   * @throws IllegalStateException
   *           if the {@link SrcType} is irrelevant because this type is
   *           being used as a parameter type and
   *           {@link SrcParamType#discardsArg} returns true. The overriding
   *           method must implement this throw.
   */
  public SrcType getCallSrcType(
    Traversable callNode, String calleeName, String paramName,
    ValueAndType args[], SrcType typeArg, SrcSymbolTable srcSymbolTable,
    LLVMModule llvmModule, boolean warningsAsErrors);

  /**
   * For a call where this type is a parameter/return type, get the
   * parameter/return type as an {@link LLVMType} for the LLVM call to which
   * this call is translated.
   * 
   * <p>
   * The default implementation returns the result of
   * {@link SrcType#getLLVMType} for {@code callSrcType}.
   * </p>
   * 
   * @param callSrcType
   *          the type previously returned by {@link #getCallSrcType} for
   *          this call
   * @param typeArg
   *          any type argument passed to the call, or null if none. Only
   *          one type argument is currently supported.
   * @param llvmModule
   *          the LLVM module into which any required declarations should be
   *          generated
   * @return the parameter/return type as an {@link LLVMType} for the target
   *         call. It's never null.
   * @throws IllegalStateException
   *           if the {@link SrcType} is irrelevant because this type is a
   *           parameter type and {@link SrcParamType#discardsArg} returns
   *           true. The overriding method must implement this throw.
   */
  public LLVMType getTargetCallLLVMType(
    SrcType callSrcType, SrcType typeArg, LLVMModule llvmModule);

  /**
   * For a call where this type is a parameter/return type, if the target
   * LLVM callee is overloaded on this parameter/return type with a special
   * (non-LLVM) name suffix, what is it?
   * 
   * <p>
   * See documentation for {@link SrcFunctionBuiltin} for how the appending
   * is performed. Note how all special suffixes are appended first.
   * </p>
   * 
   * <p>
   * The default implementation returns null.
   * </p>
   * 
   * @param targetCallLLVMType
   *          the type previously returned by {@link #getTargetCallLLVMType}
   *          for this call
   * @return a special (non-LLVM) suffix for the target LLVM callee name, or
   *         null iff the target LLVM callee is not overloaded on this
   *         parameter/return type. Any "." starting the suffix must be
   *         included here.
   */
  public String getSpecialOverloadSuffix(LLVMType targetCallLLVMType);

  /**
   * For a call where this type is a parameter/return type, is the target
   * LLVM callee overloaded on this parameter/return type with the usual
   * LLVM overload suffix?
   * 
   * <p>
   * If so, the target LLVM function name requires the parameter/return LLVM
   * type to be appended. See documentation for {@link SrcFunctionBuiltin}
   * for how the appending is performed. Note how, in the case of pointer
   * types, the target types are appended instead. Also note how all special
   * suffixes are appended first.
   * </p>
   * 
   * <p>
   * The default implementation returns false.
   * </p>
   * 
   * @param targetCallLLVMType
   *          the type previously returned by {@link #getTargetCallLLVMType}
   *          for this call
   * @param overloadsCall
   *          whether the target LLVM call is separately specified as
   *          overloaded on this parameter/return type by the definition of
   *          the {@link SrcFunctionBuiltin} being called
   * @return true iff the target LLVM callee is overloaded on this
   *         parameter/return type with the usual LLVM overload suffix
   * @throws IllegalStateException
   *           if {@code overloadsCall} and the overriding method can
   *           sometimes returns true. That is, {@link SrcFunctionBuiltin}
   *           constructor calls must not decide overloading for this
   *           parameter/return type if it's decided here. The overriding
   *           method must implement this throw.
   */
  public boolean overloadsWithLLVMSuffix(LLVMType targetCallLLVMType,
                                         boolean overloadsCall);

  /**
   * The implementation for {@link SrcParamOrReturnType}. See header
   * comments there for details.
   */
  public static abstract class Impl implements SrcParamOrReturnType {
    @Override
    public LLVMType getTargetCallLLVMType(
      SrcType callSrcType, SrcType typeArg, LLVMModule llvmModule)
    {
      return callSrcType.getLLVMType(llvmModule.getContext());
    }
    @Override
    public boolean overloadsWithLLVMSuffix(LLVMType targetCallLLVMType,
                                           boolean overloadsCall)
    {
      return false;
    }
    @Override
    public String getSpecialOverloadSuffix(LLVMType targetCallLLVMType) {
      return null;
    }
  }
}
