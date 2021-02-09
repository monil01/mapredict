package openacc.codegen.llvmBackend;

@org.junit.runner.RunWith(org.junit.runners.Suite.class)
@org.junit.runners.Suite.SuiteClasses({
  BuildLLVMTest_Preprocessor.class,
  BuildLLVMTest_GlobalVariableDeclarations.class,
  BuildLLVMTest_ProcedureDeclarations.class,
  BuildLLVMTest_Types.class,
  BuildLLVMTest_TypeConversions.class,
  BuildLLVMTest_TypeQualifiers.class,
  BuildLLVMTest_PrimaryExpressions.class,
  BuildLLVMTest_OtherExpressions.class,
  BuildLLVMTest_Statements.class,
  BuildLLVMTest_MultipleFiles.class,
  BuildLLVMTest_NVL.class,
  BuildLLVMTest_LULESH.class,
  BuildLLVMTest_SpecCPU2006.class,
})

public class JUnitSuite {}
