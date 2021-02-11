/**
 * 
 */
package openacc.transforms;

import cetus.analysis.LoopTools;
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
public class ACCtoOMP4Translator extends TransformPass {
  private enum OmpRegion
  {
    Declare,
    Declare_End,
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
    Routine,
    Declare,
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

  protected String pass_name = "[ACCtoOMP4Translator]";
  protected Program program;

  private boolean DEBUG = true;
  private boolean ASSERT_FLAG = false; 

  private List<ACCAnnotation> removeList = new LinkedList<ACCAnnotation>();
  private List<FunctionCall> funcCallList = null;
  private int defaultNumAsyncQueues = 4;
  int enableAdvancedMapping = 0;
  int vectorizationMode = 0;

  public ACCtoOMP4Translator(Program prog, int numAsyncQueues, int enableAdvMapping, int vectorMode) {
    super(prog);
    program = prog;
    defaultNumAsyncQueues = numAsyncQueues;
    enableAdvancedMapping = enableAdvMapping;
    vectorizationMode = vectorMode;
    
  }

  @Override
    public String getPassName() {
      return pass_name;
    }

  @Override
    public void start() 
    {

      ACCtoOMP4Trans(program);

    }

    public void ACCtoOMP4Trans(Traversable program) {

      if (DEBUG) {
        //System.out.println("--------");
        //System.out.println(program.toString());
        //System.out.println("--------");
      }

      // Iterate over each Annotatable in Program
      DFIterator<Annotatable> iter = new DFIterator<Annotatable>(program, Annotatable.class);
      while (iter.hasNext()) 
      {
        ASSERT_FLAG = false;

        Annotatable at = iter.next();

        // Iterate over OpenACC directives
        List<ACCAnnotation> acc_pragmas = at.getAnnotations(ACCAnnotation.class);
        for (ACCAnnotation pragma : acc_pragmas) { 

          // #pragma acc routine [clauses]
          if (pragma.containsKey("routine")) {
            parse_routine_pragma(pragma, AccRegion.Routine);
          }

          // #pragma acc declare [clauses]
          if (pragma.containsKey("declare")) {
            parse_declare_pragma(pragma, AccRegion.Declare);
          }

          // #pragma acc enter data [clauses] (OpenACC 2.0)
          if (pragma.containsKey("data") && pragma.containsKey("enter")) {
            parse_data_pragma(pragma, AccRegion.Enter_Data);
          }
          
          // #pragma acc enter data [clauses] (OpenACC 2.0)
          else if (pragma.containsKey("data") && pragma.containsKey("exit")) {
            parse_data_pragma(pragma, AccRegion.Exit_Data);
          }
          
          // #pragma acc data [clauses]
          else if (pragma.containsKey("data")) {
            parse_data_pragma(pragma, AccRegion.Data);
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
            parse_update_pragma(pragma, AccRegion.Update);
          }

        }
      }

      //Migrate simdlen clauses
      migrate_simdlen_clauses(program);

      // Migrate num_threads clauses
      migrate_nthreads_clauses(program);

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
        //System.out.println("DEF: " + def_pragma.toString());
        //System.out.println("CLASS: " + def_pragma.getClass());
        //System.out.println("KEY: " + def_pragma.keySet());
        //System.out.println("VALS: " + def_pragma.values());
        if (def_pragma.containsKey("pragma") &&
            ( (String) def_pragma.get("pragma")).contains("define")) {
          def_pragma.setSkipPrint(true);
        }
      }
      
      // Hide Loop Name annotations
      List<PragmaAnnotation> pragma_list = IRTools.collectPragmas(program, PragmaAnnotation.class, null); 
      for (PragmaAnnotation gen_pragma : pragma_list) {
        if (gen_pragma.containsKey("name"))
          gen_pragma.setSkipPrint(true);
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
          newFCall = new FunctionCall(new NameID("omp_set_default_device"));
          newFCall.addArgument(fCall.getArgument(0).clone());
          fCall.swapWith(newFCall);
        } // omp_set_default_device

        if (fName.equals("acc_get_device_num")) {
          newFCall = new FunctionCall(new NameID("omp_get_default_device"));
          fCall.swapWith(newFCall);
        } // omp_get_default_device

        if (fName.equals("acc_async_test")) {}

        if (fName.equals("acc_async_test_all")) {}

        if (fName.equals("acc_async_wait")) {}

        if (fName.equals("acc_async_wait_all")) {}

        if (fName.equals("acc_init")) {}

        if (fName.equals("acc_shutdown")) {}

        if (fName.equals("acc_on_device")) {
          newFCall = new FunctionCall(new NameID("omp_is_initial_device"));
          // Functions are actually opposites
          UnaryExpression negFCall = new UnaryExpression(UnaryOperator.LOGICAL_NEGATION, newFCall);
          fCall.swapWith(negFCall);
        } // omp_is_initial_device 

        if (fName.equals("acc_malloc")) {
          newFCall = new FunctionCall(new NameID("omp_target_alloc"));
          newFCall.addArgument(fCall.getArgument(0).clone());
          fCall.swapWith(newFCall);
        } // omp_target_alloc

        if (fName.equals("acc_free")) {
          newFCall = new FunctionCall(new NameID("omp_target_free"));
          newFCall.addArgument(fCall.getArgument(0).clone());
          fCall.swapWith(newFCall);
        } // omp_target_free

      }

    }

  // ----------------
  // Parse Pragmas 

  private void parse_routine_pragma(ACCAnnotation pragma, AccRegion regionType)
  {
    boolean removePragma = true;
    Annotatable at = pragma.getAnnotatable();

    // If there are no clauses
    // #pragma omp declare target
    // #pragma omp end declare target
    {
      OmpAnnotation newAnnot1 = new OmpAnnotation();
      //newAnnot1.put("target", "_directive");
      newAnnot1.put("declare", "target");

      OmpAnnotation newAnnot2 = new OmpAnnotation();
      //newAnnot2.put("target", "_directive");
      newAnnot2.put("end", "true");
      newAnnot2.put("declare", "target");

      pragma.getAnnotatable().annotate(newAnnot1);
      pragma.getAnnotatable().annotateAfter(newAnnot2);

      if (DEBUG) {
    	String tProcName = "Unknown";
    	Procedure tProc = IRTools.getParentProcedure(pragma.getAnnotatable());
    	if( tProc != null ) {
    		tProcName = tProc.getSymbolName();
    	}
        System.out.println("\n---- Old ACC Annotation ----");
        System.out.println("Enclosing procedure: " + tProcName);
        System.out.println(pragma.toString());
      }

      if (removePragma) {
        pragma.setSkipPrint(true);
        removeList.add(pragma);
      }

      if (DEBUG) {
        System.out.println("---- New OMP Annotation ----");
        System.out.println(newAnnot1.toString());
        System.out.println(newAnnot2.toString());
      }
    }

  }

  private void parse_declare_pragma(ACCAnnotation pragma, AccRegion regionType)
  {
    boolean removePragma = true;
    Annotatable at = pragma.getAnnotatable();

    {
      Set<SubArray> declaredSet = new HashSet<SubArray>();
      Set<SubArray> linkSet = new HashSet<SubArray>();
      if( pragma.containsKey("copyin") ) {
    	  declaredSet.addAll(pragma.get("copyin"));
      }
      if( pragma.containsKey("copy") ) {
    	  declaredSet.addAll(pragma.get("copy"));
      }
      if( pragma.containsKey("copyout") ) {
    	  declaredSet.addAll(pragma.get("copyout"));
      }
      if( pragma.containsKey("create") ) {
    	  declaredSet.addAll(pragma.get("create"));
      }
      if( pragma.containsKey("present") ) {
    	  //declaredSet.addAll(pragma.get("present"));
    	  Tools.exit("[ERROR in ACC2OMP4Translator.parse_declare_pragma()] "
    	  		+ "the current implementation cannot translate the OpenACC present clause to OpenMP; exit!\n" + 
    			  "Current annotation: " + pragma  + 
    			  AnalysisTools.getEnclosingAnnotationContext(pragma));
      }
      if( pragma.containsKey("deviceptr") ) {
    	  //declaredSet.addAll(pragma.get("deviceptr"));
    	  Tools.exit("[ERROR in ACC2OMP4Translator.parse_declare_pragma()] "
    	  		+ "the current implementation cannot translate the OpenACC deviceptr clause to OpenMP; exit!\n" + 
    			  "Current annotation: " + pragma  + 
    			  AnalysisTools.getEnclosingAnnotationContext(pragma));
      }
      if( pragma.containsKey("device_resident") ) {
    	  declaredSet.addAll(pragma.get("device_resident"));
      }
      if( pragma.containsKey("link") ) {
    	  linkSet.addAll(pragma.get("link"));
      }

      OmpAnnotation newAnnot1 = new OmpAnnotation();
      newAnnot1.put("declare", "target");
      if( !declaredSet.isEmpty() ) {
    	 newAnnot1.put("to", declaredSet); 
      }
      if( !linkSet.isEmpty() ) {
    	 newAnnot1.put("link", linkSet); 
      }
      pragma.getAnnotatable().annotate(newAnnot1);

      if (DEBUG) {
    	String tProcName = "Unknown";
    	Procedure tProc = IRTools.getParentProcedure(pragma.getAnnotatable());
    	if( tProc != null ) {
    		tProcName = tProc.getSymbolName();
    	}
        System.out.println("\n---- Old ACC Annotation ----");
        System.out.println("Enclosing procedure: " + tProcName);
        System.out.println(pragma.toString());
      }

      if (removePragma) {
        pragma.setSkipPrint(true);
        removeList.add(pragma);
      }

      if (DEBUG) {
        System.out.println("---- New OMP Annotation ----");
        System.out.println(newAnnot1.toString());
      }
    }

  }
  
  private void parse_data_pragma(ACCAnnotation pragma, AccRegion regionType)
  {
    boolean removePragma = true;
    Annotatable at = pragma.getAnnotatable();

    OmpAnnotation newAnnot = new OmpAnnotation();
    newAnnot.put("target", "_directive");
   
    if (regionType == AccRegion.Enter_Data) newAnnot.put("enter", "_directive");
    if (regionType == AccRegion.Exit_Data) newAnnot.put("exit", "_directive");

    newAnnot.put("data", "_directive");
  
    // Data Clauses
    parse_data_clauses(pragma, newAnnot, regionType);

    // Only annotate if we have at least one data clause 
    if (newAnnot.containsKey("to") ||
        newAnnot.containsKey("from") ||
        newAnnot.containsKey("tofrom") ||
        newAnnot.containsKey("alloc")) 
      pragma.getAnnotatable().annotate(newAnnot);


    if (DEBUG) {    	
    	String tProcName = "Unknown";
    	Procedure tProc = IRTools.getParentProcedure(pragma.getAnnotatable());
    	if( tProc != null ) {
    		tProcName = tProc.getSymbolName();
    	}
    	System.out.println("\n---- Old ACC Annotation ----");
        System.out.println("Enclosing procedure: " + tProcName);
    	System.out.println(pragma.toString());
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

  private void parse_loop_pragma(ACCAnnotation pragma, AccRegion regionType)
  {
    boolean removePragma = true;
    Annotatable at = pragma.getAnnotatable();

    OmpAnnotation newAnnot = new OmpAnnotation();

    // Parallel Clauses
    boolean rv = parse_parallel_clauses(pragma, newAnnot, regionType);
    if (!rv) {
      if (removePragma) {
        pragma.setSkipPrint(true);
        removeList.add(pragma);
      }
      return;
    }

    // Other Clauses
    if ( pragma.containsKey("collapse") ) {
      Expression condExp = pragma.get("collapse");
      newAnnot.put("collapse", condExp);
    }
    
/*    if (pragma.containsKey("private") && !newAnnot.containsKey("simd")) {
      Set privateSet = pragma.get("private");
      newAnnot.put("private", privateSet);
    }*/
    if (pragma.containsKey("private")) {
      Set<SubArray> privateSet = pragma.get("private");
      //[DEBUG] Semantics of OpenMP linear clause is different from that of OpenACC private clause.
      //OpenMP Specification Version 5.0, Section 2.19.4.6 linear Clause says that the value corresponding to 
      //the sequentially last iteration of the associated loop(s) is assigned to the original list item.
      //Therefore, converting private clause to linear clause may break the original program semantics.
      //[DEBUG on May 28, 2020] In OpenMP Specification Version 5.0 Section 2.19.1.1:
      //	- The loop iteration variable in the associated for-loop of a simd construct with just one associated for-loop 
      //	is linear with a linear-step that is the increment of the associated for-loop.
      //	- The loop iteration variables in the associated for-loops of a simd construct with multiple 
      //	associated for-loops are lastprivate.
      //	- The collapse clause may be used to specify how many loops are associated with the construct.
      //	- If no collapse clause is present, the only loop that is associated with the simd construct is the one
      //	that immediately follows the directive.
      if( newAnnot.containsKey("simd") ) {
    	  if( at instanceof ForLoop ) {
    		  Set<Symbol> loopIndexSymbols = AnalysisTools.getWorkSharingLoopIndexVarSet(at);
    		  if( loopIndexSymbols != null ) {
    			  String indexVariableType = "linear";
    			  if( (loopIndexSymbols.size() > 1) && newAnnot.containsKey("collapse") ) {
    				  if( ((IntegerLiteral)newAnnot.get("collapse")).getValue() > 1 ) {
    					  indexVariableType = "lastprivate";
    				  }
    			  }
    			  for( Symbol indexSymbol : loopIndexSymbols ) {
    				  SubArray indexVarSArray = null;
    				  for( SubArray tSubA : privateSet ) {
    					  if( tSubA.getArrayName().toString().equals(indexSymbol.getSymbolName()) ) {
    						  indexVarSArray = tSubA;
    						  break;
    					  }
    				  }
    				  if( indexVarSArray != null ) {
    					  Set<SubArray> linearSet = newAnnot.get(indexVariableType);
    					  if( linearSet == null ) {
    						  linearSet = new HashSet<SubArray>();
    						  newAnnot.put(indexVariableType, linearSet);
    					  }
    					  linearSet.add(indexVarSArray);
    					  privateSet.remove(indexVarSArray);
    				  }
    			  }
    		  }
    	  }
      }
      if( !privateSet.isEmpty() ) {
    	  newAnnot.put("private", privateSet);
      }
    }

    if( newAnnot.containsKey("distribute") || newAnnot.containsKey("parallel") || newAnnot.containsKey("for") ) {
    	if (pragma.containsKey("firstprivate") ) {
    		Set<SubArray> firstprivateSet = pragma.get("firstprivate");
    		newAnnot.put("firstprivate", firstprivateSet);
    	}
    	
    }
    if( newAnnot.containsKey("simd") || newAnnot.containsKey("parallel") || newAnnot.containsKey("for") ) {
    	if (pragma.containsKey("reduction")) {
    		// Convert from OpenACC data structure to OpenMP data structure
    		Map<ReductionOperator, Set<SubArray>> acc_red_map = pragma.get("reduction");
    		HashMap<String, Set> omp_red_map = new HashMap<String, Set>();

    		for (ReductionOperator op : acc_red_map.keySet() ) {
    			omp_red_map.put(op.toString(), acc_red_map.get(op));
    		}
    		newAnnot.put("reduction", omp_red_map);
    	}
    }

    boolean addNewAnnotation = false;
    if (!newAnnot.isEmpty()) {
    	if( newAnnot.containsKey("distribute") || newAnnot.containsKey("parallel") || newAnnot.containsKey("for")
    			|| newAnnot.containsKey("simd") ) {
    		pragma.getAnnotatable().annotate(newAnnot);
    		addNewAnnotation = true;
    	}
    }

    if (DEBUG) {
    	String tProcName = "Unknown";
    	Procedure tProc = IRTools.getParentProcedure(pragma.getAnnotatable());
    	if( tProc != null ) {
    		tProcName = tProc.getSymbolName();
    	}
    	System.out.println("\n---- Old ACC Annotation ----");
        System.out.println("Enclosing procedure: " + tProcName);
    	System.out.println(pragma.toString());
    }

    if (removePragma) {
      pragma.setSkipPrint(true);
      removeList.add(pragma);
    }

    if (DEBUG) {
      System.out.println("---- New OMP Annotation ----");
      if( addNewAnnotation ) {
    	  System.out.println(newAnnot.toString());
      } else {
    	  System.out.println("");
      }
    }
  }

  private void parse_parallel_pragma(ACCAnnotation pragma, AccRegion regionType)
  {
    boolean removePragma = true;
    Annotatable at = pragma.getAnnotatable();

    OmpAnnotation newAnnot = new OmpAnnotation();
    newAnnot.put("target", "_directive");
    newAnnot.put("teams", "_directive");
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
      Set<SubArray> privateSet = pragma.get("private");
      newAnnot.put("private", privateSet);
    }

    if (pragma.containsKey("firstprivate")) {
      Set<SubArray> firstprivateSet = pragma.get("firstprivate");
      newAnnot.put("firstprivate", firstprivateSet);
    }


    if (DEBUG) {
    	String tProcName = "Unknown";
    	Procedure tProc = IRTools.getParentProcedure(pragma.getAnnotatable());
    	if( tProc != null ) {
    		tProcName = tProc.getSymbolName();
    	}
    	System.out.println("\n---- Old ACC Annotation ----");
        System.out.println("Enclosing procedure: " + tProcName);
    	System.out.println(pragma.toString());
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

    OmpAnnotation newAnnot = new OmpAnnotation();
    newAnnot.put("target", "_directive");
    newAnnot.put("teams", "_directive");

    // Parallel Clauses
    boolean rv = parse_parallel_clauses(pragma, newAnnot, regionType);
    if (!rv) {
      if (removePragma) {
        pragma.setSkipPrint(true);
        removeList.add(pragma);
      }
      return;
    }

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

/*    if (pragma.containsKey("private")) {
      Set<SubArray> privateSet = pragma.get("private");
      newAnnot.put("private", privateSet);
    }*/
    if (pragma.containsKey("private")) {
      Set<SubArray> privateSet = pragma.get("private");
      //[DEBUG] Semantics of OpenMP linear clause is different from that of OpenACC private clause.
      //OpenMP Specification Version 5.0, Section 2.19.4.6 linear Clause says that the value corresponding to 
      //the sequentially last iteration of the associated loop(s) is assigned to the original list item.
      //Therefore, converting private clause to linear clause may break the original program semantics.
      //[DEBUG on May 28, 2020] In OpenMP Specification Version 5.0 Section 2.19.1.1:
      //	- The loop iteration variable in the associated for-loop of a simd construct with just one associated for-loop 
      //	is linear with a linear-step that is the increment of the associated for-loop.
      //	- The loop iteration variables in the associated for-loops of a simd construct with multiple 
      //	associated for-loops are lastprivate.
      //	- The collapse clause may be used to specify how many loops are associated with the construct.
      //	- If no collapse clause is present, the only loop that is associated with the simd construct is the one
      //	that immediately follows the directive.
      if( newAnnot.containsKey("simd") ) {
    	  Annotatable at = pragma.getAnnotatable();
    	  if( at instanceof ForLoop ) {
    		  Set<Symbol> loopIndexSymbols = AnalysisTools.getWorkSharingLoopIndexVarSet(at);
    		  if( loopIndexSymbols != null ) {
    			  String indexVariableType = "linear";
    			  if( (loopIndexSymbols.size() > 1) && newAnnot.containsKey("collapse") ) {
    				  if( ((IntegerLiteral)newAnnot.get("collapse")).getValue() > 1 ) {
    					  indexVariableType = "lastprivate";
    				  }
    			  }
    			  for( Symbol indexSymbol : loopIndexSymbols ) {
    				  SubArray indexVarSArray = null;
    				  for( SubArray tSubA : privateSet ) {
    					  if( tSubA.getArrayName().toString().equals(indexSymbol.getSymbolName()) ) {
    						  indexVarSArray = tSubA;
    						  break;
    					  }
    				  }
    				  if( indexVarSArray != null ) {
    					  Set<SubArray> linearSet = newAnnot.get(indexVariableType);
    					  if( linearSet == null ) {
    						  linearSet = new HashSet<SubArray>();
    						  newAnnot.put(indexVariableType, linearSet);
    					  }
    					  linearSet.add(indexVarSArray);
    					  privateSet.remove(indexVarSArray);
    				  }
    			  }
    		  }
    	  }
      }
      if( !privateSet.isEmpty() ) {
    	  newAnnot.put("private", privateSet);
      }
    }

    if (pragma.containsKey("firstprivate")) {
      Set<SubArray> firstprivateSet = pragma.get("firstprivate");
      newAnnot.put("firstprivate", firstprivateSet);
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

    if (DEBUG) {
    	String tProcName = "Unknown";
    	Procedure tProc = IRTools.getParentProcedure(pragma.getAnnotatable());
    	if( tProc != null ) {
    		tProcName = tProc.getSymbolName();
    	}
    	System.out.println("\n---- Old ACC Annotation ----");
        System.out.println("Enclosing procedure: " + tProcName);
    	System.out.println(pragma.toString());
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

  private void parse_update_pragma(ACCAnnotation pragma, AccRegion regionType)
  {
    boolean removePragma = true;
    Annotatable at = pragma.getAnnotatable();

    OmpAnnotation newAnnot = new OmpAnnotation();
    newAnnot.put("target", "_directive");
    newAnnot.put("update", "_directive");
    pragma.getAnnotatable().annotate(newAnnot);

    // Data Clauses
    parse_data_clauses(pragma, newAnnot, regionType);

    // Other Clauses
    if( pragma.containsKey("if") ) {
      Expression condExp = pragma.get("if");
      newAnnot.put("if", condExp);
    }

    if (DEBUG) {
    	String tProcName = "Unknown";
    	Procedure tProc = IRTools.getParentProcedure(pragma.getAnnotatable());
    	if( tProc != null ) {
    		tProcName = tProc.getSymbolName();
    	}
    	System.out.println("\n---- Old ACC Annotation ----");
        System.out.println("Enclosing procedure: " + tProcName);
    	System.out.println(pragma.toString());
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

  // ----------------
  // Parse Clauses

  private void parse_parallel_size_clauses(ACCAnnotation pragma, OmpAnnotation newAnnot, AccRegion regionType)
  {
    if (pragma.containsKey("num_gangs")) {
      Expression num_teams = pragma.get("num_gangs");
      //newAnnot.put("num_teams", num_teams);
    }

    if (pragma.containsKey("num_workers")) {
      Expression num_threads = pragma.get("num_workers");

      // Temporarily require constant argument.
      //   Because we may need to move the clause to the appropriate
      //   "parallel for" directive. This can be fixed by creating a 
      //   temporary variable and assignment statement before this directive
      if ( !(num_threads instanceof Literal) ) {
    	  Tools.exit("[ERROR in ACC2OMP4Translator.parse_parallel_size_clause()] "
    			  + "vector_length() argument must be constant; exit!\n" + 
    			  "Current annotation: " + pragma  + 
    			  AnalysisTools.getEnclosingAnnotationContext(pragma));
      }

      newAnnot.put("num_threads", num_threads);
    }

    // This clause needs to be migrated to loops with the simd clause
    if (pragma.containsKey("vector_length")) {
      Expression simdlen = pragma.get("vector_length");

      // OpenMP requires constant argument
      if ( !(simdlen instanceof Literal) ) {
    	  Tools.exit("[ERROR in ACC2OMP4Translator.parse_parallel_size_clause()] "
    			  + "vector_length() argument must be constant; exit!\n" + 
    			  "Current annotation: " + pragma  + 
    			  AnalysisTools.getEnclosingAnnotationContext(pragma));
      }
      newAnnot.put("simdlen", simdlen);
    }
  }

  private boolean parse_parallel_clauses(ACCAnnotation pragma, OmpAnnotation newAnnot, AccRegion regionType) 
  {
    boolean is_gang =   pragma.containsKey("gang");
    boolean is_worker = pragma.containsKey("worker");
    boolean is_vector = pragma.containsKey("vector");
    boolean is_none = !(is_gang || is_worker || is_vector);

    // gang loop
    if (is_gang) {
      newAnnot.put("distribute", "_directive");

      // migrate reduction clause up to teams directive
      if (regionType == AccRegion.Loop)
        migrate_reduction_clause(pragma);
    }

    // worker loop
    if (is_worker) {
      newAnnot.put("parallel", "_directive");
      newAnnot.put("for", "_directive");
    }

    // vector loop
    // We need to verify that either this directive or a parent directive contains 
    //   a gang or worker clause. Otherwise we need to add "parallel for num_threads(1)"
    //   to satisfy OpenMP requirments
    if (is_vector) {

      // Traverse parents looking for gang or worker clauses
      Annotatable at = pragma.getAnnotatable();
      boolean lone_vector = true;
      while (lone_vector) {

        if (at.containsAnnotation(ACCAnnotation.class, "gang") || 
            at.containsAnnotation(ACCAnnotation.class, "worker")) {
          lone_vector = false;
          break;
        }

        if (at.containsAnnotation(ACCAnnotation.class, "parallel")) {
          lone_vector = true;
          break;
        }

        at = (Annotatable) at.getParent();
      }

      if (lone_vector) {
        newAnnot.put("parallel", "_directive");
        newAnnot.put("for", "_directive");
        newAnnot.put("num_threads", 1);
        newAnnot.put("simd", "_directive");
      }
      
      if (!lone_vector) {
        newAnnot.put("simd", "_directive");
      }
    }

    // We can try to automatically apply an appropriate directive if none is provided
    if (is_none) {
      boolean parent_gang = false;
      boolean parent_worker = false;
      boolean parent_vector = false;

      // Traverse parents looking for gang, worker or vector clauses
      Annotatable at = pragma.getAnnotatable();
      while (true) {

        if (at.containsAnnotation(ACCAnnotation.class, "gang") ||
            at.containsAnnotation(OmpAnnotation.class, "distribute")) parent_gang = true;

        if (at.containsAnnotation(ACCAnnotation.class, "worker") ||
            at.containsAnnotation(OmpAnnotation.class, "for")) parent_worker = true;

        if (at.containsAnnotation(ACCAnnotation.class, "vector") ||
            at.containsAnnotation(OmpAnnotation.class, "simd")) parent_vector = true;

        if (at.containsAnnotation(ACCAnnotation.class, "parallel") ||
            at.containsAnnotation(ACCAnnotation.class, "kernels") ||
            at.containsAnnotation(OmpAnnotation.class, "teams")) break;
        	//[DEBUG on June 2, 2020] the last condition was changed from "parallel" to "teams".
         

        at = (Annotatable) at.getParent();
      }

      // invalid loop
      if (parent_gang && parent_worker && parent_vector) {
        //Tools.exit("[ERROR in ACC2OMPTranslation] ACC loop directive nested inside vector loop: not allowed: " +  at.toString());
        // We should omit the directive in this case
        return false;
      }
      // vector loop
      else if (parent_gang && parent_worker) {
    	  if( enableAdvancedMapping == 0 ) 
        newAnnot.put("simd", "_directive");
      }

      // worker loop
      else if (parent_gang) {
        newAnnot.put("parallel", "_directive");
        newAnnot.put("for", "_directive");
      }

      // gang loop
      else {
        newAnnot.put("distribute", "_directive");
      }

    }

    return true;
  }

  private void parse_data_clauses(ACCAnnotation pragma, OmpAnnotation newAnnot, AccRegion regionType) 
  {

    if (regionType == AccRegion.Data || regionType == AccRegion.Parallel || regionType == AccRegion.Parallel_Loop) {
      if (pragma.containsKey("create") && !((HashSet) pragma.get("create")).isEmpty())
        newAnnot.put("alloc", pragma.get("create"));
      if (pragma.containsKey("copyin") && !((HashSet) pragma.get("copyin")).isEmpty())
        newAnnot.put("to", pragma.get("copyin"));
      if (pragma.containsKey("copyout") && !((HashSet) pragma.get("copyout")).isEmpty())
        newAnnot.put("from", pragma.get("copyout"));
      if (pragma.containsKey("copy") && !((HashSet) pragma.get("copy")).isEmpty())
        newAnnot.put("tofrom", pragma.get("copy"));

      // pcopy
      if (pragma.containsKey("pcreate") && !((HashSet) pragma.get("pcreate")).isEmpty())
        newAnnot.put("alloc", pragma.get("pcreate"));
      if (pragma.containsKey("pcopyin") && !((HashSet) pragma.get("pcopyin")).isEmpty())
        newAnnot.put("to", pragma.get("pcopyin"));
      if (pragma.containsKey("pcopyout") && !((HashSet) pragma.get("pcopyout")).isEmpty())
        newAnnot.put("from", pragma.get("pcopyout"));
      if (pragma.containsKey("pcopy") && !((HashSet) pragma.get("pcopy")).isEmpty())
        newAnnot.put("tofrom", pragma.get("pcopy"));

      // present
      if (pragma.containsKey("present") && !((HashSet) pragma.get("present")).isEmpty()) {

        // Get enclosing compound statement
        Annotatable at = pragma.getAnnotatable();
        Traversable parent = at.getParent();

        while (parent.getClass() != CompoundStatement.class) {
          parent = parent.getParent();
        }

        CompoundStatement compStmt = (CompoundStatement) parent;
        Statement atStmt = (Statement) at;

        // For each variable in present clause
        HashSet<SubArray> varSet = (HashSet<SubArray>) pragma.get("present");
        for (SubArray sArray : varSet) {

          // Generate function call
          FunctionCall presFcall = new FunctionCall(new NameID("omp_target_is_present"));
          
          // Dereference non-pointer variables
          if (sArray.getArrayDimension() != 0) 
            presFcall.addArgument(sArray.getArrayName().clone());
          else 
            presFcall.addArgument(new UnaryExpression(UnaryOperator.ADDRESS_OF, 
              sArray.getArrayName().clone() ));

          presFcall.addArgument(new IntegerLiteral(0));

          FunctionCall assertFcall = new FunctionCall(new NameID("assert"));
          assertFcall.addArgument(presFcall);

          Statement assertStmt = new ExpressionStatement(assertFcall);

          // Insert function call into compound statement 
          //compStmt.addStatementBefore(atStmt, assertStmt);

          // Insert <assert.h> into program
          CodeAnnotation assertHeader = new CodeAnnotation("#include <assert.h>");
          if (ASSERT_FLAG == false) {
            //assertStmt.annotate(assertHeader);
            ASSERT_FLAG = true;
          }

        }
      }

    }

    if (regionType == AccRegion.Enter_Data) {
      if (pragma.containsKey("create") && !((HashSet) pragma.get("create")).isEmpty())
        newAnnot.put("alloc", pragma.get("create"));
      if (pragma.containsKey("copyin") && !((HashSet) pragma.get("copyin")).isEmpty())
        newAnnot.put("to", pragma.get("copyin"));

      // pcopy
      if (pragma.containsKey("pcreate") && !((HashSet) pragma.get("pcreate")).isEmpty())
        newAnnot.put("alloc", pragma.get("pcreate"));
      if (pragma.containsKey("pcopyin") && !((HashSet) pragma.get("pcopyin")).isEmpty())
        newAnnot.put("to", pragma.get("pcopyin"));
      
      // present
      if (pragma.containsKey("present") && !((HashSet) pragma.get("present")).isEmpty())
        newAnnot.put("tofrom", pragma.get("present"));
    }

    if (regionType == AccRegion.Exit_Data) {
      if (pragma.containsKey("copyout") && !((HashSet) pragma.get("copyout")).isEmpty())
        newAnnot.put("from", pragma.get("copyout"));

      if (pragma.containsKey("pcopyout") && !((HashSet) pragma.get("pcopyout")).isEmpty())
        newAnnot.put("from", pragma.get("pcopyout"));
    }

    if (regionType == AccRegion.Update) {
      if (pragma.containsKey("host"))     newAnnot.put("from", pragma.get("host")); 
      if (pragma.containsKey("device"))   newAnnot.put("to", pragma.get("device")); 
    }

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
