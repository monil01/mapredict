package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcParamAndReturnType.SRC_PARAMRET_NVL_HEAP_PTR;
import static openacc.codegen.llvmBackend.SrcParamType.SRC_PARAM_COND;
import static openacc.codegen.llvmBackend.SrcParamType.SRC_PARAM_METADATA;
import static openacc.codegen.llvmBackend.SrcParamType.SRC_PARAM_MPI_GROUP;
import static openacc.codegen.llvmBackend.SrcParamType.SRC_PARAM_PTR_TO_NVL;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SRC_SIZE_T_TYPE;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcBoolType;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SrcCharType;
import static openacc.codegen.llvmBackend.SrcPrimitiveNonNumericType.SrcVoidType;
import static openacc.codegen.llvmBackend.SrcQualifiedType.SrcTypeQualifier.NVL;

import java.util.ArrayList;

import openacc.hir.NVLAnnotation;

import org.jllvm.LLVMConstant;
import org.jllvm.LLVMConstantInteger;
import org.jllvm.LLVMIntegerType;
import org.jllvm.LLVMMDNode;
import org.jllvm.LLVMMDString;
import org.jllvm.LLVMValue;

import cetus.hir.Annotation;
import cetus.hir.AnnotationStatement;
import cetus.hir.Expression;

/**
 * The LLVM backend's class for translating NVL-C pragmas.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public class NVL extends PragmaTranslator {
  private final StatementPragmaTranslator atomicTranslator
    = new StatementPragmaTranslator(NVLAnnotation.class, "atomic") {
      @Override
      public void start(Annotation annot) {
        // Start llvm.nvl.explicitTx basic block set and entry portal.
        final LLVMMDNode explicitTxMetadata = v.basicBlockSetStack.push(
          "llvm.nvl.explicitTx", "llvm.nvl.explicitTxList", v.llvmBuilder);
        v.basicBlockSetStack.push(
          "llvm.nvl.explicitTx.entryPortal",
          "llvm.nvl.explicitTx.entryPortalList",
          ".nvl.explicitTx.entryFirstBB", v.llvmBuilder);

        // Get and validate heap clause.
        final Expression heapExpr = (Expression)annot.get("heap");
        if (heapExpr == null)
          throw new SrcRuntimeException(
            "nvl "+getName()+" pragma has no heap clause");
        v.srcSymbolTable.addParentFixup(heapExpr, annot.getAnnotatable());
        v.visitTree(heapExpr);
        final ValueAndType heap = v.postOrderValuesAndTypes.remove(
          v.postOrderValuesAndTypes.size()-1);
        final SrcPointerType heapPtrTy = heap.getSrcType().prepareForOp()
                                         .toIso(SrcPointerType.class);
        if (heap.isNullPointerConstant())
          throw new SrcRuntimeException(
            "nvl "+getName()+" pragma's heap clause is a null pointer"
            +" constant");
        if (heapPtrTy == null)
          throw new SrcRuntimeException(
            "nvl "+getName()+" pragma's heap clause has non-pointer type");
        final SrcType nvlHeapT = v.srcSymbolTable.getBuiltinTypeTable()
                                 .getNVLHeapT(v.llvmModule,
                                              v.warningsAsErrors);
        if (!heapPtrTy.getTargetType().eqv(nvlHeapT))
          throw new SrcRuntimeException(
            "nvl "+getName()+" pragma's heap clause is not a pointer to an"
            +" unqualified nvl_heap_t");

        // Complain about readonly clause.
        if (annot.get("readonly") != null)
          throw new UnsupportedOperationException(
            "nvl "+getName()+" pragma does not yet support readonly"
            +" clause");

        // Get and validate default clause.
        final String defaultSpec = annot.get("default");
        final String default_ = defaultSpec == null ? "backup"
                                                    : defaultSpec;
        if (default_.equals("backup")) {
          if (annot.get("backup_writeFirst") != null)
            throw new UnsupportedOperationException(
              "nvl "+getName()+" pragma does not yet support"
              +" backup_writeFirst clause without default(readonly)"
              +" clause");
          if (annot.get("clobber") != null)
            throw new UnsupportedOperationException(
              "nvl "+getName()+" pragma does not yet support clobber clause"
              +" without default(readonly) clause");
        }
        else if (default_.equals("backup_writeFirst"))
          throw new UnsupportedOperationException(
            "nvl "+getName()+" pragma does not yet support"
            +" default(backup_writeFirst) clause");
        else if (default_.equals("clobber"))
          throw new UnsupportedOperationException(
            "nvl "+getName()+" pragma does not yet support default(clobber)"
            +" clause");
        else if (default_.equals("readonly"))
          ;
        else
          throw new UnsupportedOperationException(
            "nvl "+getName()+" pragma's default clause has unknown"
            +" argument: "+default_);

        // TODO: Complain if the same memory is specified in multiple
        // clauses (backup, clobber, readonly).

        // Get and validate mpiGroup clause.
        final Expression mpiGroupExpr = (Expression)annot.get("mpiGroup");
        final ValueAndType mpiGroup;
        if (mpiGroupExpr == null)
          mpiGroup = null;
        else {
          v.srcSymbolTable.addParentFixup(mpiGroupExpr,
                                          annot.getAnnotatable());
          v.visitTree(mpiGroupExpr);
          mpiGroup = v.postOrderValuesAndTypes.remove(
                       v.postOrderValuesAndTypes.size()-1);
          assert(mpiGroup != null);
        }

        // Insert tx.persist calls for clobber clauses.
        // TODO: Complain if memory contains a pointer because then it must
        // be backed up. Currently, it will be backed up anyway, and the
        // clobber clause will merely produce wasteful persist calls.
        for (ValueAndType[] arr : evalPragmaDataClause(annot, getName(),
                                                       "clobber"))
        {
          final ValueAndType addr = ValueAndType.add(
            "nvl "+getName()+" pragma clobber clause",
            arr[0], arr[1], v.llvmModule, v.llvmBuilder);
          final ValueAndType numElements = arr[2];
          final ValueAndType elementSize = arr[3];
          // TODO: Support weak pointer as arg.
          new SrcFunctionBuiltin(
            "nvl "+getName()+" pragma clobber clause",
            "llvm.nvl.tx.persist.v2nv", SrcVoidType, false,
            new SrcParamType[]{SRC_PARAM_PTR_TO_NVL, SRC_SIZE_T_TYPE,
                               SRC_SIZE_T_TYPE},
            false, new boolean[]{false, true, true}, null)
          .call(annot.getAnnotatable(),
                new ValueAndType[]{addr, numElements, elementSize},
                v.srcSymbolTable, v.llvmModule, v.llvmModuleIndex,
                v.llvmTargetData, v.llvmBuilder, v.warningsAsErrors);
        }

        // Call entry intrinsic and end entry portal.
        final ArrayList<SrcParamType> paramTypes = new ArrayList<>();
        final ArrayList<String> paramNames = new ArrayList<>();
        final ArrayList<ValueAndType> args = new ArrayList<>();
        paramTypes.add(SRC_PARAM_METADATA);
        paramTypes.add(SRC_PARAM_METADATA);
        paramTypes.add(SRC_PARAMRET_NVL_HEAP_PTR);
        paramNames.add("atomic basic block set ID");
        paramNames.add("default clause");
        paramNames.add("heap clause");
        args.add(heap);
        if (mpiGroup != null) {
          paramTypes.add(SRC_PARAM_MPI_GROUP);
          paramNames.add("mpiGroup clause");
          args.add(mpiGroup);
        }
        new SrcFunctionBuiltin(
          "nvl "+getName()+" pragma",
          "llvm.nvl.tx.begin.heap"+(mpiGroup==null ? ".local" : ""),
          SrcVoidType, false,
          paramTypes.toArray(new SrcParamType[paramTypes.size()]),
          false, null, paramNames.toArray(new String[paramNames.size()]))
        .call(annot.getAnnotatable(),
              args.toArray(new ValueAndType[args.size()]),
              new LLVMValue[]{explicitTxMetadata,
                              LLVMMDString.get(v.llvmContext, default_)},
              v.srcSymbolTable, v.llvmModule, v.llvmModuleIndex,
              v.llvmTargetData, v.llvmBuilder, v.warningsAsErrors);
        v.basicBlockSetStack.pop(".nvl.explicitTx.bodyFirstBB",
                                 v.llvmBuilder);

        // Insert tx.add calls for backup clauses.
        for (ValueAndType[] arr : evalPragmaDataClause(annot, getName(),
                                                       "backup"))
          insertTxAdd(annot, arr, "backup", false);
        // Insert tx.add calls for backup_writeFirst clauses.
        for (ValueAndType[] arr : evalPragmaDataClause(annot, getName(),
                                                       "backup_writeFirst"))
          insertTxAdd(annot, arr, "backup_writeFirst", true);
      }

      private void insertTxAdd(Annotation annot, ValueAndType[] arr,
                               String clauseName, boolean writeFirst)
      {
        final ValueAndType addr = ValueAndType.add(
          "nvl "+getName()+" pragma "+clauseName+" clause",
          arr[0], arr[1], v.llvmModule, v.llvmBuilder);
        final ValueAndType numElements = arr[2];
        final ValueAndType elementSize = arr[3];
        final LLVMIntegerType i1 = SrcBoolType.getLLVMType(v.llvmContext);
        final ValueAndType false_ = new ValueAndType(
          LLVMConstant.constNull(i1), SrcBoolType, false);
        final ValueAndType writeFirstArg = new ValueAndType(
          LLVMConstantInteger.get(i1, writeFirst?1:0, false),
          SrcBoolType, false);
        // TODO: Support weak pointer as arg and return type.
        new SrcFunctionBuiltin(
          "nvl "+getName()+" pragma "+clauseName+" clause",
          "llvm.nvl.tx.add.v2nv",
          // TODO: This is ugly. We should create a new target-only
          // SrcReturnType for
          // {@code i8 addrspace({@link #LLVM_ADDRSPACE_NVL})*}
          SrcPointerType.get(SrcQualifiedType.get(SrcCharType, NVL)),
          false,
          new SrcParamType[]{SRC_PARAM_PTR_TO_NVL, SRC_SIZE_T_TYPE,
                             SRC_SIZE_T_TYPE, SRC_PARAM_COND,
                             SRC_PARAM_COND},
          false, new boolean[]{false, true, true, false, false}, null)
        .call(annot.getAnnotatable(),
              new ValueAndType[]{addr, numElements, elementSize, false_,
                                 writeFirstArg},
              v.srcSymbolTable, v.llvmModule, v.llvmModuleIndex,
              v.llvmTargetData, v.llvmBuilder, v.warningsAsErrors);
      }

      @Override
      public void end(Annotation annot) {
        v.basicBlockSetStack.pop(".nvl.explicitTx.nextBB", v.llvmBuilder);
      }
    };
  private final StatementPragmaTranslator[] statementPragmaTranslators = {
    atomicTranslator
  };

  public NVL(BuildLLVM.Visitor v) {
    super(v);
  }

  @Override
  protected StatementPragmaTranslator[] getStatementPragmaTranslators() {
    return statementPragmaTranslators;
  }

  @Override
  public void translateStandalonePragmas(AnnotationStatement node) {
  }
}
