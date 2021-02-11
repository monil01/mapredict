/**
 * 
 */
package openacc.transforms;

import cetus.analysis.Reduction;
import cetus.hir.*;
import cetus.transforms.TransformPass;
import openacc.analysis.ACCAnalysis;
import openacc.analysis.ACCParser;
import openacc.analysis.AnalysisTools;
import openacc.analysis.ParserTools;
import openacc.analysis.SubArray;
import openacc.hir.ACCAnnotation;
import openacc.hir.ARCAnnotation;
import openacc.hir.ASPENAnnotation;
import openacc.hir.NVLAnnotation;
import openacc.hir.ReductionOperator;

import java.util.*;

/**
 * @author Jacob Lambert <jlambert@cs.uoregon.edu>
 *         Seyong Lee <lees2@ornl.gov>
 *
 */
public class ACCtoOMP3Translator extends TransformPass {
  private enum OmpRegion
  {
    Target,
    Target_Data,
    Target_Enter_Data,
    Target_Exit_Data,
    Target_Update,
    Teams,
    Distribute,
    Parallel_For,
    Parallel,
    SIMD
  }

  private enum AccRegion
  {
    Data,
    Loop,
    Parallel,
    Update,
    Enter_Data,
    Exit_Data,
    Parallel_Loop
  }

  private enum taskType
  {
    innerTask,
    outerTask
  }

  private boolean DEBUG = true;

  protected String pass_name = "[ACCtoOMP3Translator]";
  protected Program program;

  private List<ACCAnnotation> removeList = new LinkedList<ACCAnnotation>();
  private List<FunctionCall> funcCallList = null;
  private int defaultNumAsyncQueues = 4;

  public ACCtoOMP3Translator(Program prog, int numAsyncQueues) {
    super(prog);
    program = prog;
    defaultNumAsyncQueues = numAsyncQueues;
  }

  @Override
    public String getPassName() {
      return pass_name;
    }

  @Override
    public void start() 
    {

      ACCtoOMP3Trans(program);

    }

    public void ACCtoOMP3Trans(Traversable program) {

      if (DEBUG) {
        //System.out.println("--------");
        //System.out.println(program.toString());
        //System.out.println("--------");
      }


      // Iterate over each Annotatable in Program
      DFIterator<Annotatable> iter = new DFIterator<Annotatable>(program, Annotatable.class);
      while (iter.hasNext()) 
      {
        Annotatable at = iter.next();

        // Iterate over OpenACC directives
        List<ACCAnnotation> acc_pragmas = at.getAnnotations(ACCAnnotation.class);
        for (ACCAnnotation pragma : acc_pragmas) { 


          // #pragma acc enter data [clauses] (OpenACC 2.0)
          if (pragma.containsKey("data") && pragma.containsKey("enter")) {
            //parse_data_pragma(pragma, AccRegion.Enter_Data);
          }

          // #pragma acc enter data [clauses] (OpenACC 2.0)
          else if (pragma.containsKey("data") && pragma.containsKey("exit")) {
            //parse_data_pragma(pragma, AccRegion.Exit_Data);
          }

          // #pragma acc data [clauses]
          else if (pragma.containsKey("data")) {
            //parse_data_pragma(pragma, AccRegion.Data);
          }

          // #pragma acc parallel loop [clauses]
          else if (pragma.containsKey("loop") && pragma.containsKey("parallel")) {
            parse_parallel_loop_pragma(pragma, AccRegion.Parallel_Loop);
          }

          // #pragma acc kernels loop [clauses]
          else if (pragma.containsKey("loop") && pragma.containsKey("kernels")) {
            parse_parallel_loop_pragma(pragma, AccRegion.Parallel_Loop);
          }

          // #pragma acc loop [clauses]
          else if (pragma.containsKey("loop")) {
            parse_loop_pragma(pragma, AccRegion.Loop);
          }

          // #pragma acc parallel [clauses]
          else if (pragma.containsKey("parallel")) {
            parse_parallel_pragma(pragma, AccRegion.Parallel);
          }

          // #pragma acc kernels [clauses]
          else if (pragma.containsKey("kernels")) {
            parse_parallel_pragma(pragma, AccRegion.Parallel);
          }

          // #pragma acc update [clauses]
          else if (pragma.containsKey("update")) {
            //parse_update_pragma(pragma, AccRegion.Update);
          }

        }
      }

      //Migrate simdlen clauses
      migrate_simdlen_clauses(program);

      // Migrate num_threads clauses
      migrate_nthreads_clauses(program);
      
      // Remove private clauses from omp simd directives
      List<OmpAnnotation> simd_pragma_list = IRTools.collectPragmas(program, OmpAnnotation.class, "simd");
      for (OmpAnnotation simd_pragma : simd_pragma_list) {
        simd_pragma.remove("private");
      }

/*      // Hide Cetus annotations
      List<CetusAnnotation> cetus_pragma_list = IRTools.collectPragmas(program, CetusAnnotation.class, null); 
      for (CetusAnnotation cetus_pragma : cetus_pragma_list) {
        cetus_pragma.setSkipPrint(true);
      }

      // Hide ARC annotations
      List<ARCAnnotation> arc_pragma_list = IRTools.collectPragmas(program, ARCAnnotation.class, null); 
      for (ARCAnnotation arc_pragma : arc_pragma_list) {
        arc_pragma.setSkipPrint(true);
      }*/

      // Hide "#pragma openarc #define" annotations
      List<PragmaAnnotation> def_pragma_list = IRTools.collectPragmas(program, PragmaAnnotation.class, null);
      for (PragmaAnnotation def_pragma : def_pragma_list) {
        if (def_pragma.containsKey("pragma") &&
            ( (String) def_pragma.get("pragma")).contains("define")) {
          def_pragma.setSkipPrint(true);
        }
      }

      // Replace OpenACC API calls
      funcCallList = IRTools.getFunctionCalls(program);
      for( FunctionCall fCall : funcCallList ) {

        String fName = fCall.getName().toString();
        FunctionCall newFCall = fCall.clone();

        if (fName.equals("acc_get_num_devices")) {} // omp_get_num_devices

        if (fName.equals("acc_set_device_type")) {} 

        if (fName.equals("acc_get_device_type")) {} 

        if (fName.equals("acc_set_device_num")) {
          //newFCall = new FunctionCall(new NameID("omp_set_default_device"));
          //newFCall.addArgument(fCall.getArgument(0).clone());
          //fCall.swapWith(newFCall);
        } // omp_set_default_device

        if (fName.equals("acc_get_device_num")) {
          //newFCall = new FunctionCall(new NameID("omp_get_default_device"));
          //fCall.swapWith(newFCall);
        } // omp_get_default_device

        if (fName.equals("acc_async_test")) {}

        if (fName.equals("acc_async_test_all")) {}

        if (fName.equals("acc_async_wait")) {}

        if (fName.equals("acc_async_wait_all")) {}

        if (fName.equals("acc_init")) {}

        if (fName.equals("acc_shutdown")) {}

        if (fName.equals("acc_on_device")) {
          //newFCall = new FunctionCall(new NameID("omp_is_initial_device"));
          // Functions are actually opposites
          //UnaryExpression negFCall = new UnaryExpression(UnaryOperator.LOGICAL_NEGATION, newFCall);
          //fCall.swapWith(negFCall);
        } // omp_is_initial_device 

        if (fName.equals("acc_malloc")) {
          //newFCall = new FunctionCall(new NameID("omp_target_alloc"));
          //newFCall.addArgument(fCall.getArgument(0).clone());
          //fCall.swapWith(newFCall);
        } // omp_target_alloc

        if (fName.equals("acc_free")) {
          //newFCall = new FunctionCall(new NameID("omp_target_free"));
          //newFCall.addArgument(fCall.getArgument(0).clone());
          //`fCall.swapWith(newFCall);
        } // omp_target_free

        if (fCall.getName().toString().contains("acc_")) {

          if (DEBUG) {
            System.out.println("\n-------- ACC Function Call --------");
            System.out.println(fCall.toString());
            System.out.println("-------- OMP Function Call --------");
            System.out.println(newFCall.toString());
          }
        } 

      }

      //Tools.exit(program.toString());
    }

  // ----------------
  // Parse Pragmas 

  private void parse_loop_pragma(ACCAnnotation pragma, AccRegion regionType)
  {
    boolean removePragma = true;
    Annotatable at = pragma.getAnnotatable();

    if (DEBUG) {
      System.out.println("\n---- Old ACC Annotation ----");
      System.out.println(pragma.toString());
    }

    OmpAnnotation newAnnot = new OmpAnnotation();

    // Parallel Clauses
    parse_parallel_clauses(pragma, newAnnot, regionType);

    // Other Clauses
    if ( pragma.containsKey("collapse") ) {
      Expression condExp = pragma.get("collapse");
      newAnnot.put("collapse", condExp);
    }

    if (pragma.containsKey("private")) {
      Set<String> privateSet = pragma.get("private");
      newAnnot.put("private", privateSet);
    }

    if (!newAnnot.isEmpty())
      pragma.getAnnotatable().annotate(newAnnot);

    if (removePragma) {
      pragma.setSkipPrint(true);
      removeList.add(pragma);
    }

    if (DEBUG) { 
      System.out.println("---- New OMP Annotation ----");
      System.out.println(newAnnot.toString());
    }
  }

  private void parse_parallel_pragma(ACCAnnotation pragma, AccRegion regionType)
  {
    boolean removePragma = true;
    Annotatable at = pragma.getAnnotatable();

    if (DEBUG) {
      System.out.println("\n---- Old ACC Annotation ----");
      System.out.println(pragma.toString());
    }

    OmpAnnotation newAnnot = new OmpAnnotation();
    newAnnot.put("parallel", "_directive");
    pragma.getAnnotatable().annotate(newAnnot);

    // Parallel Size Clauses
    parse_parallel_size_clauses(pragma, newAnnot, regionType);

    // Data Clauses 
    parse_data_clauses(pragma, newAnnot, regionType);

    // Other Clauses
    if( pragma.containsKey("if") ) {
      Expression condExp = pragma.get("if");
      newAnnot.put("if", condExp);
    }

    if (pragma.containsKey("private")) {
      Set<String> privateSet = pragma.get("private");
      newAnnot.put("private", privateSet);
    }

    if (removePragma) {
      pragma.setSkipPrint(true);
      removeList.add(pragma);
    }

    if (DEBUG) {
      System.out.println("---- New OMP Annotation ----");
      System.out.println(newAnnot.toString());
    }
  }

  private void parse_parallel_loop_pragma(ACCAnnotation pragma, AccRegion regionType)
  {
    boolean removePragma = true;

    if (DEBUG) {
      System.out.println("\n---- Old ACC Annotation ----");
      System.out.println(pragma.toString());
    }

    OmpAnnotation newAnnot = new OmpAnnotation();
    newAnnot.put("parallel", "_directive");
    newAnnot.put("for", "_directive");

    // Parallel Clauses
    parse_parallel_clauses(pragma, newAnnot, regionType);

    // Parallel Size Clauses
    parse_parallel_size_clauses(pragma, newAnnot, regionType);

    // Data Clauses 
    parse_data_clauses(pragma, newAnnot, regionType);

    // Other Clauses 
    if ( pragma.containsKey("collapse") ) {
      Expression condExp = pragma.get("collapse");
      newAnnot.put("collapse", condExp);
    }

    if( pragma.containsKey("if") ) {
      Expression condExp = pragma.get("if");
      newAnnot.put("if", condExp);
    }

    if (pragma.containsKey("private")) {
      Set<String> privateSet = pragma.get("private");
      newAnnot.put("private", privateSet);
    }

    if (pragma.containsKey("reduction")) {
      // Convert from OpenACC data structure to OpenMP data structure
      Map<ReductionOperator, Set<SubArray>> acc_red_map = pragma.get("reduction");
      HashMap<String, Set> omp_red_map = new HashMap<String, Set>();

      for (ReductionOperator op : acc_red_map.keySet() ) {
        omp_red_map.put(op.toString(), acc_red_map.get(op));
      }
      newAnnot.put("reduction", omp_red_map);
    }

    pragma.getAnnotatable().annotate(newAnnot);

    if (removePragma) {
      pragma.setSkipPrint(true);
      removeList.add(pragma);
    }

    if (DEBUG) {
      System.out.println("---- New OMP Annotation ----");
      System.out.println(newAnnot.toString());
    }
  }

  // ----------------
  // Parse Clauses

  private void parse_parallel_size_clauses(ACCAnnotation pragma, OmpAnnotation newAnnot, AccRegion regionType)
  {
    // This clause needs to be migrated to loops with the simd clause
    if (pragma.containsKey("vector_length")) {
      Expression simdlen = pragma.get("vector_length");

      // OpenMP requires constant argument
      if ( !(simdlen instanceof Literal) ) {
        Tools.exit("[ERROR in ACC2OMPTranslation] vector_length() argument must be constant");
      }
      newAnnot.put("simdlen", simdlen);
    }
  }

  private void parse_parallel_clauses(ACCAnnotation pragma, OmpAnnotation newAnnot, AccRegion regionType) 
  {
    if (pragma.containsKey("seq")) return;

    boolean is_gang =   pragma.containsKey("gang");
    boolean is_worker = pragma.containsKey("worker");
    boolean is_vector = pragma.containsKey("vector");
    boolean is_none = !(is_gang || is_worker || is_vector);

    // We can try to automatically apply an appropriate directive if none is provided
    boolean parent_gang = false;
    boolean parent_worker = false;
    boolean parent_vector = false;

    // Traverse parents looking for gang, worker or vector clauses
    Annotatable at = pragma.getAnnotatable();
    Annotatable parent_vector_at = null;
    while (true) {

      if (at.containsAnnotation(ACCAnnotation.class, "gang") ||
          at.containsAnnotation(OmpAnnotation.class, "for")) parent_gang = true;

      if (at.containsAnnotation(ACCAnnotation.class, "worker") ||
          at.containsAnnotation(OmpAnnotation.class, "for")) parent_worker = true;

      if (at.containsAnnotation(ACCAnnotation.class, "vector") ||
          at.containsAnnotation(OmpAnnotation.class, "simd")) {
        parent_vector = true;
        parent_vector_at = at;
      }

      if (at.containsAnnotation(ACCAnnotation.class, "parallel") ||
          at.containsAnnotation(ACCAnnotation.class, "kernels"))  break;

      at = (Annotatable) at.getParent();
    }


    // Handle cases where there are parallel clauses present
    // gang (parallel for)
    if (is_gang) {
      newAnnot.put("for", "_directive");
      System.out.println("adding gang for directive");

      return;
    }

    // worker (simd)
    // We should check if there is a child vector (simd) before adding here
    if (is_worker) {
      newAnnot.put("simd", "_directive");
      System.out.println("adding worker simd directive");

      return;
    }

    if (is_vector && !parent_worker) {
      newAnnot.put("simd", "_directive");
      System.out.println("adding vector simd directive");
    }

    // Handle cases without parallel clauses present
    // Should not happen
    if (!parent_gang) {
      Tools.exit("Orphan directive: " + at.toString());
      return;
    }

    // Add vector clause
    if ( (parent_gang || parent_worker) && !parent_vector) {
      newAnnot.put("simd", "_directive");
      return;
    }

    // Move vector clause to inner loop
    if ( (parent_gang || parent_worker) && parent_vector) {
      parent_vector_at.removeAnnotations(OmpAnnotation.class);
      newAnnot.put("simd", "directive");
      return;
    }

    Tools.exit("No Action Taken");
    return;

  }

  private void parse_data_clauses(ACCAnnotation pragma, OmpAnnotation newAnnot, AccRegion regionType) 
  {

    // private
    // firstprivate
    // shared

  }

  // Migrate simdlen clauses
  //   acc:vector_length clauses are attached to acc:parallel pragmas, thus
  //   omp:simdlen clauses are attached to omp:target teams pragmas
  //
  //   However, we need
  //   omp:simdlen clauses to be attached to omp:simd pragmas
  private void migrate_simdlen_clauses(Traversable T)
  {
    List<OmpAnnotation> simdlen_pragma_list = IRTools.collectPragmas(T, OmpAnnotation.class, "simdlen"); 
    for (OmpAnnotation simdlen_pragma : simdlen_pragma_list) {

      // remove simdlen clause
      Object length = simdlen_pragma.remove("simdlen");
      Annotatable simdlen_at = simdlen_pragma.getAnnotatable();

      // get child simd clauses
      List<OmpAnnotation> simd_pragma_list = IRTools.collectPragmas(simdlen_at, OmpAnnotation.class, "simd"); 

      // annotate child simd clauses with simdlen clause
      for (OmpAnnotation simd_pragma : simd_pragma_list) {
        simd_pragma.put("simdlen", length);
      }
    }

  }

  // Migrate num_threads clauses
  //   acc:num_workers clauses are attached to acc:parallel pragmas, thus
  //   omp:num_threads clauses are attached to omp:target teams pragmas
  //
  //   However, we need
  //   omp:num_threads clauses to be attached to omp:parallel for pragmas
  private void migrate_nthreads_clauses(Traversable T)
  {
    List<OmpAnnotation> nthread_pragma_list = IRTools.collectPragmas(T, OmpAnnotation.class, "num_threads"); 
    for (OmpAnnotation nthread_pragma : nthread_pragma_list) {

      // remove num_threads clause
      Object length = nthread_pragma.remove("num_threads");
      Annotatable nthread_at = nthread_pragma.getAnnotatable();

      // get child parallel for clauses
      Set<String> searchKeys = new HashSet<String>();
      searchKeys.add("parallel");
      searchKeys.add("for");
      List<OmpAnnotation> pfor_pragma_list = AnalysisTools.collectPragmas(nthread_at, OmpAnnotation.class, searchKeys, false); 

      // annotate child simd clauses with simdlen clause
      for (OmpAnnotation pfor_pragma: pfor_pragma_list) {
        pfor_pragma.put("num_threads", length);
      }
    }

  }

  // We can get the reduction from the cetus directive, and then 
  //   migrate it up to the OpenMP Teams directive. This is easier than 
  //   using the OpenACC directives, because there is no guarantee about
  //   where the reduciton clause may be
  private void migrate_reduction_clause(ACCAnnotation pragma) 
  {
    Annotatable at = pragma.getAnnotatable();

    // Get the reduction from the Cetus pragma
    List<CetusAnnotation> cetus_pragma_list = at.getAnnotations(CetusAnnotation.class);
    for (CetusAnnotation cetus_pragma : cetus_pragma_list) {
      if (cetus_pragma.containsKey("reduction")) {

        HashMap<String, Set> cetus_red_map = cetus_pragma.get("reduction");
        // Traverse up to teams directive
        while (!at.containsAnnotation(OmpAnnotation.class, "teams")) {
          at = (Annotatable) at.getParent();
        }

        OmpAnnotation teamsAnnot = at.getAnnotation(OmpAnnotation.class, "teams"); 
        teamsAnnot.put("reduction", cetus_red_map);
      }
    }

  }

}
