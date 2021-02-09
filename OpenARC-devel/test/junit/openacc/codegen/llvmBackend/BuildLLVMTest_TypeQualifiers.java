package openacc.codegen.llvmBackend;

import java.io.IOException;

import org.hamcrest.CoreMatchers;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Checks the ability to handle C's type qualifiers. For now, we focus mostly
 * on negative tests because existing apps provide positive tests. Moreover,
 * we focus on {@code const} because we have not attempted implement the
 * semantics of {@code restrict} or {@code volatile} beyond C's general
 * contraints on type qualifiers.
 * 
 * @author Joel E. Denny <dennyje@ornl.gov> - Future Technologies Group, Oak
 *         Ridge National Laboratory
 */
public class BuildLLVMTest_TypeQualifiers extends BuildLLVMTest {
  @BeforeClass public static void setup() {
    System.loadLibrary("jllvm");
  }

  @Test public void constFn() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "type qualifiers specified on function type"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "typedef void T();",
        "T const fn;",
      });
  }

  @Test public void assignToConst() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "first operand of simple assignment operator is not a modifiable"
      +" lvalue"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "int const i;",
        "void fn() {",
        "  i = 5;",
        "}",
      });
  }

  @Test public void assignToConstByPtr() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "first operand of simple assignment operator is not a modifiable"
      +" lvalue"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "int const *p;",
        "void fn() {",
        "  *p = 5;",
        "}",
      });
  }

  @Test public void assignToConstStruct() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "first operand of simple assignment operator is not a modifiable"
      +" lvalue"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "struct T { int i; } const t1, t2;",
        "void fn() {",
        "  t1 = t2;",
        "}",
      });
  }

  @Test public void assignToConstStructDot() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "first operand of simple assignment operator is not a modifiable"
      +" lvalue"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "struct T { int i; };",
        "void fn() {",
        "  struct T const t;",
        "  t.i = 5;",
        "}",
      });
  }

  @Test public void assignToConstStructArrow() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "first operand of simple assignment operator is not a modifiable"
      +" lvalue"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "struct T { int i; };",
        "void fn() {",
        "  struct T const *p;",
        "  p->i = 5;",
        "}",
      });
  }

  @Test public void initDiscardsQualifiers() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "warning treated as error:"
      +" initialization discards type qualifiers from pointer target type:"
      +" const"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "void foo() {",
        "  int const *p1;",
        "  int *p2 = p1;",
        "}",
      });
  }

  @Test public void argDiscardsQualifiers() throws IOException {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "warning treated as error:"
      +" argument 1 to \"foo\" discards type qualifiers from pointer target"
      +" type: const"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "void foo(int *p) {}",
        "void bar() {",
        "  int const *p;",
        "  foo(p);",
        "}",
      });
  }

  @Test public void assignFromMemberDiscardsStructQualifiers()
    throws IOException
  {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "warning treated as error:"
      +" assignment operator discards type qualifiers from pointer target type:"
      +" const"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "void foo() {",
        "  struct T { int i; } const t;",
        "  int *p;",
        "  p = &t.i;",
        "}",
      });
  }

  /**
   * Example from ISO C99 sec. 6.7.3p11. Type qualifiers on array (via
   * typedef) applies to element type.
   */
  @Test public void assignFromElementDiscardsArrayQualifiers()
    throws IOException
  {
    exception.expect(SrcRuntimeException.class);
    exception.expectMessage(CoreMatchers.equalTo(
      "warning treated as error:"
      +" assignment operator discards type qualifiers from pointer target type:"
      +" const"));
    buildLLVMSimple(
      "", "",
      new String[]{
        "void foo() {",
        "  typedef int A[2][3];",
        "  const A a = {{4, 5, 6}, {7, 8, 9}};",
        "  int *pi;",
        "  pi = a[0];",
        "}",
      });
  }
}
