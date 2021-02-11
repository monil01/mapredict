package openacc.codegen.llvmBackend;

import static openacc.codegen.llvmBackend.SrcParamType.SRC_PARAM_COND;
import static openacc.codegen.llvmBackend.SrcParamType.SRC_PARAM_METADATA;
import static openacc.codegen.llvmBackend.SrcPrimitiveIntegerType.SRC_SIZE_T_TYPE;
import static openacc.codegen.llvmBackend.SrcPrimitiveNonNumericType.SrcVoidType;

import java.util.HashSet;
import java.util.Set;

import openacc.analysis.SubArray;
import openacc.hir.ARCAnnotation;

import org.jllvm.LLVMMDNode;
import org.jllvm.LLVMMDString;
import org.jllvm.LLVMValue;

import cetus.hir.Annotation;
import cetus.hir.AnnotationStatement;

/**
 * The LLVM backend's class for translating OpenARC fault-injection pragmas
 * to FITL.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public class FITL extends PragmaTranslator {
  private final StatementPragmaTranslator resilienceTranslator
    = new StatementPragmaTranslator(ARCAnnotation.class, "resilience") {
      @Override
      public void start(Annotation annot) {
        final LLVMMDNode domainMetadata = v.basicBlockSetStack.push(
          "llvm.fitl.domain", "llvm.fitl.domainList", v.llvmBuilder);
        v.basicBlockSetStack.push(
          "llvm.fitl.desert", "llvm.fitl.desertList",
          ".fitl.domain.entryFirstBB", v.llvmBuilder);

        final ValueAndType ftcond
          = evalPragmaIntegerClause(annot, getName(), "ftcond", 1);
        final ValueAndType numFaults
          = evalPragmaIntegerClause(annot, getName(), "num_faults", -1);
        final ValueAndType numFtbits
          = evalPragmaIntegerClause(annot, getName(), "num_ftbits", 1);
        final ValueAndType repeat
          = evalPragmaIntegerClause(annot, getName(), "repeat", 1);
        final ValueAndType ftprofile
          = evalPragmaIntegerLvalueClause(annot, getName(), "ftprofile");
        final ValueAndType ftpredict
          = evalPragmaIntegerClause(annot, getName(), "ftpredict", -1);

        new SrcFunctionBuiltin(
          getName()+" pragma", "llvm.fitl.startDomain", SrcVoidType, false,
          new SrcParamType[]{SRC_PARAM_METADATA, SRC_PARAM_COND,
                             numFaults.getSrcType().prepareForOp(),
                             numFtbits.getSrcType().prepareForOp(),
                             repeat.getSrcType().prepareForOp(),
                             ftprofile.getSrcType().prepareForOp(),
                             ftpredict.getSrcType().prepareForOp()},
          false, new boolean[]{false, false, true, true, true, true, true},
          null)
        .call(annot.getAnnotatable(),
              new ValueAndType[]{ftcond, numFaults, numFtbits, repeat,
                                 ftprofile, ftpredict},
              new LLVMValue[]{domainMetadata},
              v.srcSymbolTable, v.llvmModule, v.llvmModuleIndex,
              v.llvmTargetData, v.llvmBuilder, v.warningsAsErrors);

        v.basicBlockSetStack.pop(".fitl.domain.bodyFirstBB", v.llvmBuilder);

        // If this pragma does not contain the ftregion clause, then it won't
        // be picked up by the ftregionTranslator, so process it now if it
        // contains ftregion subclauses.
        if (annot.get("ftregion") == null
            && (annot.get("ftthread") != null
                || annot.get("ftdata") != null
                || annot.get("ftkind") != null))
          ftregionTranslator.start(annot);
      }

      @Override
      public void end(Annotation annot) {
        if (annot.get("ftregion") == null
            && (annot.get("ftthread") != null
                || annot.get("ftdata") != null
                || annot.get("ftkind") != null))
          ftregionTranslator.end(annot);
        v.basicBlockSetStack.pop(".fitl.domain.nextBB", v.llvmBuilder);
      }
    };
  private final StatementPragmaTranslator ftregionTranslator
    = new StatementPragmaTranslator(ARCAnnotation.class, "ftregion") {
      @Override
      public void start(Annotation annot) {
        // Be sure not to generate any instructions before starting the
        // first FITL desert below.
        final Set<SubArray> ftdata = annot.get("ftdata");
        final Set<String> ftkind = annot.get("ftkind");
        if (ftdata == null && ftkind == null)
          throw new SrcRuntimeException(
            getName()+" pragma has no ftdata or ftkind clause");
        final Set<SubArray> ftdataArrs;
        final Set<String> ftkindStrings;
        if (ftdata == null) {
          ftdataArrs = new HashSet<>();
          ftdataArrs.add(null);
        }
        else
          ftdataArrs = ftdata;
        if (ftkind == null) {
          ftkindStrings = new HashSet<>();
          ftkindStrings.add("");
        }
        else
          ftkindStrings = ftkind;

        for (String kind : ftkindStrings) {
          for (final SubArray arr : ftdataArrs) {
            final LLVMMDNode regionMetadata = v.basicBlockSetStack.push(
              "llvm.fitl.region", "llvm.fitl.regionList", v.llvmBuilder);
            v.basicBlockSetStack.push(
              "llvm.fitl.desert", "llvm.fitl.desertList",
              ".fitl.region.entryFirstBB", v.llvmBuilder);
            final ValueAndType ftthread
              = evalPragmaIntegerClause(annot, getName(), "ftthread", 0);
            final ValueAndType[] arrValues = evalPragmaDataClauseSubArray(
              annot.getAnnotatable(), getName(), "ftdata", arr);
            new SrcFunctionBuiltin(
              getName()+" pragma", "llvm.fitl.startRegion", SrcVoidType,
              false,
              new SrcParamType[]{
                SRC_PARAM_METADATA, SRC_PARAM_METADATA,
                ftthread.getSrcType().prepareForOp(),
                SrcPointerType.get(SrcVoidType),
                SRC_SIZE_T_TYPE, SRC_SIZE_T_TYPE, SRC_SIZE_T_TYPE},
              false,
              new boolean[]{false, false, true, false, true, false, false},
              null)
            .call(annot.getAnnotatable(),
                  new ValueAndType[]{ftthread,
                                     arrValues[0], arrValues[1],
                                     arrValues[2], arrValues[3]},
                  new LLVMValue[]{regionMetadata,
                                  LLVMMDString.get(v.llvmContext, kind)},
                  v.srcSymbolTable, v.llvmModule, v.llvmModuleIndex,
                  v.llvmTargetData, v.llvmBuilder, v.warningsAsErrors);
            v.basicBlockSetStack.pop(".fitl.region.bodyFirstBB",
                                     v.llvmBuilder);
          }
        }
      }

      @Override
      public void end(Annotation annot) {
        final Set<SubArray> ftdata = annot.get("ftdata");
        final Set<String> ftkind = annot.get("ftkind");
        int ftdataSize = ftdata == null ? 1 : ftdata.size();
        int ftkindSize = ftkind == null ? 1 : ftkind.size();
        for (int i = 0; i < ftdataSize; ++i)
          for (int j = 0; j < ftkindSize; ++j)
            v.basicBlockSetStack.pop(".fitl.region.nextBB", v.llvmBuilder);
      }
    };
  private final StatementPragmaTranslator[] statementPragmaTranslators = {
    resilienceTranslator, ftregionTranslator
  };

  public FITL(BuildLLVM.Visitor v) {
    super(v);
  }

  @Override
  protected StatementPragmaTranslator[] getStatementPragmaTranslators() {
    return statementPragmaTranslators;
  }

  @Override
  public void translateStandalonePragmas(AnnotationStatement node) {
    final String pragmaName = "ftinject";
    final ARCAnnotation annot = node.getAnnotation(ARCAnnotation.class,
                                                   pragmaName);
    if (annot == null)
      return;

    v.basicBlockSetStack.push(
      "llvm.fitl.desert", "llvm.fitl.desertList",
      ".fitl.inject.firstBB", v.llvmBuilder);

    final ValueAndType ftthread
      = evalPragmaIntegerClause(annot, pragmaName, "ftthread", 0);
    final Set<SubArray> ftdata = annot.get("ftdata");
    if (ftdata == null)
      throw new SrcRuntimeException(
        pragmaName+" pragma has no ftdata clause");
    for (final SubArray arr : ftdata) {
      final ValueAndType[] arrValues = evalPragmaDataClauseSubArray(
        annot.getAnnotatable(), pragmaName, "ftdata", arr);
      new SrcFunctionBuiltin(
        pragmaName+" pragma", "llvm.fitl.inject", SrcVoidType, false,
        new SrcType[]{ftthread.getSrcType().prepareForOp(),
                      SrcPointerType.get(SrcVoidType),
                      SRC_SIZE_T_TYPE, SRC_SIZE_T_TYPE, SRC_SIZE_T_TYPE},
        false, new boolean[]{true, false, true, false, false}, null)
      .call(annot.getAnnotatable(),
            new ValueAndType[]{ftthread, arrValues[0], arrValues[1],
                               arrValues[2], arrValues[3]},
            v.srcSymbolTable, v.llvmModule, v.llvmModuleIndex,
            v.llvmTargetData, v.llvmBuilder, v.warningsAsErrors);
    }

    v.basicBlockSetStack.pop(".fitl.inject.nextBB", v.llvmBuilder);
  }
}
