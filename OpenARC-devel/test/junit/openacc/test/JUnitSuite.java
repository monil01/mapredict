package openacc.test;

@org.junit.runner.RunWith(org.junit.runners.Suite.class)
@org.junit.runners.Suite.SuiteClasses({
  org.jllvm.JUnitSuite.class,
  openacc.codegen.llvmBackend.JUnitSuite.class,
  JUnitTest.XFailSummary.class,
})

public class JUnitSuite {}
