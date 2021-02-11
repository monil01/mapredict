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

import java.util.*;

/**
 * @author Putt Sakdhnagool <psakdhna@purdue.edu>
 *         Seyong Lee <lees2@ornl.gov>
 *
 */
public class OMP4toACCTranslator extends TransformPass {
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
    
    private enum taskType
    {
    	innerTask,
    	outerTask
    }

    protected String pass_name = "[OMP4toACCTranslator]";
    protected Program program;
    //Main refers either a procedure containing acc_init() call or main() procedure if no explicit acc_init() call exists.
    protected Procedure main;

    //Target information
    private Expression deviceExpr = null;

    private Stack<Traversable> parentStack = new Stack<Traversable>();
    private Stack<OmpRegion> regionStack = new Stack<OmpRegion>();
    private List<OmpAnnotation> removeList = new LinkedList<OmpAnnotation>();
    private HashMap<String, String> macroMap = null;
    private List<Declaration> procDeclList = null;
	private	List<FunctionCall> funcCallList = null;
	private int defaultNumAsyncQueues = 4;
	private boolean createDeviceTask = false;

    public OMP4toACCTranslator(Program prog, int numAsyncQueues) {
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
    	HashMap<String, String> gMacroMap = ACCAnnotationParser.parseCmdLineMacros();
    	List<OmpAnnotation> TargetRegionAnnots = new LinkedList<OmpAnnotation>();
		Annotation new_annot = null;
		boolean attach_to_next_annotatable = false;
		LinkedList<Annotation> annots_to_be_attached = new LinkedList<Annotation>();
		HashMap<String, Object> new_map = null;

    	convertCritical2Reduction();

		funcCallList = IRTools.getFunctionCalls(program);

    	for( Traversable trUnt : program.getChildren() ) {
    		macroMap = new HashMap<String, String>();
    		macroMap.putAll(gMacroMap); //Add global commandline macros.
        	List<Statement> insertRefList = new LinkedList<Statement>();
        	List<Statement> insertTargetList = new LinkedList<Statement>();
    		DFIterator<Annotatable> iter = new DFIterator<Annotatable>(trUnt, Annotatable.class);
    		while (iter.hasNext()) 
    		{
    			Annotatable at = iter.next();

				////////////////////////////////////////////////////////////////////////
				// AnnotationParser store OpenACC annotations as PragmaAnnotations in //
				// either AnnotationStatements or AnnotationDeclaration, depending on //
				// their locations.                                                   //
				////////////////////////////////////////////////////////////////////////
				if ((at instanceof AnnotationStatement) || (at instanceof AnnotationDeclaration) )
				{
					List<PragmaAnnotation> annot_list = 
						at.getAnnotations(PragmaAnnotation.class);
					if( (annot_list == null) || (annot_list.size() == 0) ) {
						continue;
					}
					////////////////////////////////////////////////////////////////////////////
					// AnnotationParser creates one AnnotationStatement/AnnotationDeclaration //
					// for each PragmaAnnotation.                                             //
					////////////////////////////////////////////////////////////////////////////
					PragmaAnnotation pAnnot = annot_list.get(0);
					//////////////////////////////////////////////////////////////////////////
					// The above pAnnot may be a derived annotation.                        //
					// If so, skip it, except for OmpAnnotation.                            //
					// (The below annotations are child classes of PragmaAnnotation.)       //
					// DEBUG: If new child class of the PragmaAnnotation is added, below    //
					// should be updated too.                                               //
					// (ex: If CudaAnnotation is added, it should be checked here.)         //
					//////////////////////////////////////////////////////////////////////////
					if( pAnnot instanceof CetusAnnotation || pAnnot instanceof ASPENAnnotation ||
							pAnnot instanceof InlineAnnotation || pAnnot instanceof PragmaAnnotation.Event ||
							pAnnot instanceof PragmaAnnotation.Range || pAnnot instanceof ACCAnnotation ||
							pAnnot instanceof ARCAnnotation || pAnnot instanceof NVLAnnotation ) {
						continue;
					}
					String old_annot = pAnnot.getName();
					if( old_annot == null ) {
						if( !(pAnnot instanceof OmpAnnotation) ) {
							PrintTools.println("\n[WARNING in OMP4toACCTranslator] Pragma annotation, "
								+ pAnnot +", does not have name.\n", 0);
							continue;
						}
					} else {
						old_annot = ACCAnnotationParser.modifyAnnotationString(old_annot);
						String[] token_array = old_annot.split("\\s+");
						// If old_annot string has a leading space, the 2nd token should be checked.
						//String old_annot_key = token_array[1];
						String old_annot_key = token_array[0];
						if ((old_annot_key.compareTo("openarc")==0)) {
							String old_annot_key2 = token_array[1];
							if( old_annot_key2.equals("#") ) { //Preprocess macros on OpenACC/OpenARC directives.
								ACCParser.preprocess_acc_pragma(token_array, macroMap);
								continue;
							}
						}
					}
				}

    			List<OmpAnnotation> pragmas = at.getAnnotations(OmpAnnotation.class);
    			for (int i = 0; i < pragmas.size(); i++) 
    			{
    				OmpAnnotation pragma = pragmas.get(i);
    				if(pragma.containsKey("barrier"))
    				{
    					if( !at.containsAnnotation(ACCAnnotation.class, "wait") ) {
    						ACCAnnotation newAnnot = new ACCAnnotation();
    						newAnnot.put("wait", "_directive");
    						at.annotate(newAnnot);
    						//pragma.setSkipPrint(true);
    						//pragma.remove("barrier");
    						//removeList.add(pragma);
    					}
    					continue;
    				}
    				if(pragma.containsKey("taskwait"))
    				{
    					if( !at.containsAnnotation(ACCAnnotation.class, "wait") ) {
    						ACCAnnotation newAnnot = new ACCAnnotation();
    						newAnnot.put("wait", "_directive");
    						at.annotate(newAnnot);
    						//pragma.setSkipPrint(true);
    						//pragma.remove("taskwait");
    						//removeList.add(pragma);
    					}
    					continue;
    				}
    				if(pragma.containsKey("taskgroup"))
    				{
    					//[DEBUG] current implementation checks lexically-included target regions only; fixed.
    					//List<OmpAnnotation> tList = IRTools.collectPragmas(at, OmpAnnotation.class, "target"); 
    					List<OmpAnnotation> tList = AnalysisTools.ipCollectPragmas(at, OmpAnnotation.class, "target", null); 
    					if( (tList != null) && !tList.isEmpty() ) {
    						ACCAnnotation newAnnot = new ACCAnnotation();
    						newAnnot.put("wait", "_directive");
    						AnnotationStatement annotStmt = new AnnotationStatement(newAnnot);
    						insertRefList.add((Statement)at);
    						insertTargetList.add(annotStmt);
    						//pragma.setSkipPrint(true);
    						//pragma.remove("taskgroup");
    						//removeList.add(pragma);
    					}
    					continue;
    				}
    				//Handle implicit barriers at the end of loop construct (omp for), sections construct, 
    				//and single construct if there is no nowait clause.
    				if( (pragma.containsKey("for") || pragma.containsKey("single") || pragma.containsKey("sections")) && !pragma.containsKey("nowait"))
    				{
    					if(!at.containsAnnotation(OmpAnnotation.class, "parallel") && !at.containsAnnotation(OmpAnnotation.class, "target")) {
    						//[DEBUG] current implementation checks lexically-included target regions only; fixed.
    						//List<OmpAnnotation> tList = IRTools.collectPragmas(at, OmpAnnotation.class, "target"); 
    						List<OmpAnnotation> tList = AnalysisTools.ipCollectPragmas(at, OmpAnnotation.class, "target", null); 
    						if( (tList != null) && !tList.isEmpty() ) {
    							ACCAnnotation newAnnot = new ACCAnnotation();
    							newAnnot.put("wait", "_directive");
    							AnnotationStatement annotStmt = new AnnotationStatement(newAnnot);
    							insertRefList.add((Statement)at);
    							insertTargetList.add(annotStmt);
    						}
    					}
    					if( !pragma.containsKey("target") ) {
    						continue;
    					}
    				}
    				if(pragma.containsKey("parallel") && !pragma.containsKey("for") && !pragma.containsKey("sections"))
    				{
    					if(!at.containsAnnotation(OmpAnnotation.class, "target")) {
    						//[DEBUG] current implementation checks lexically-included target regions only; fixed.
    						//List<OmpAnnotation> tList = IRTools.collectPragmas(at, OmpAnnotation.class, "target"); 
    						List<OmpAnnotation> tList = AnalysisTools.ipCollectPragmas(at, OmpAnnotation.class, "target", null); 
    						if( (tList != null) && !tList.isEmpty() ) {
    							ACCAnnotation newAnnot = new ACCAnnotation();
    							newAnnot.put("wait", "_directive");
    							AnnotationStatement annotStmt = new AnnotationStatement(newAnnot);
    							insertRefList.add((Statement)at);
    							insertTargetList.add(annotStmt);
    						}
    					}
    					if( !pragma.containsKey("target") ) {
    						continue;
    					}
    				}

    				if(pragma.containsKey("declare"))
    				{
    					String declareStr = pragma.get("declare");
    					if( declareStr.equals("target") ) {
    						if( !pragma.containsKey("end") ) {
    							//#pragma omp declare target
    							parse_declare_target(pragma);
    						}
    					} else if( declareStr.equals("simd") ) {
    						//#pragma omp declare simd clause, clause ...
    					}
    					continue;
    				}
    				if(pragma.containsKey("target") && pragma.containsKey("update"))
    				{
    					parse_target_pragma(pragma, OmpRegion.Target_Update);
    					continue;
    				}
    				if(pragma.containsKey("target") && pragma.containsKey("enter") && pragma.containsKey("data"))
    				{
    					parse_target_pragma(pragma, OmpRegion.Target_Enter_Data);
    					continue;
    				}
    				if(pragma.containsKey("target") && pragma.containsKey("exit") && pragma.containsKey("data"))
    				{
    					parse_target_pragma(pragma, OmpRegion.Target_Exit_Data);
    					continue;
    				}
    				if(pragma.containsKey("target") && pragma.containsKey("data"))
    				{
    					parse_target_pragma(pragma, OmpRegion.Target_Data);
    					continue;
    				}

    				if(pragma.containsKey("target"))
    				{
    					if( !(at instanceof ForLoop) ) {
    						String directive = null;
    						if (pragma.containsKey("distribute")) {
    							directive = "distribute";
    						} else if (pragma.containsKey("for")) {
    							directive = "for";
    						} else if (pragma.containsKey("simd")) {
    							directive = "simd";
    						} 
    						if (directive != null) {
    							Tools.exit("[ERROR in OMP4toACCTranslator.start()] An OpenMP "+ directive + " directive is allowed only to for-loops, "
    									+ "but the following construct illegally uses the " + directive + " directive:\n"
    									+ "OpenMP construct: " + at + "\n" + AnalysisTools.getEnclosingContext(at));
    						}
    					}
    					//Enter Target region
    					TargetRegionAnnots.add(pragma);
    					Object condExp = null;
    					if( pragma.containsKey("if") ) {
    						condExp = pragma.get("if");
    					}
    					if( at instanceof CompoundStatement ) {
    						FlatIterator<Statement> flatIterator = new FlatIterator<Statement>(at);
    						while (flatIterator.hasNext())
    						{
    							Statement stmt = flatIterator.next();
    							OmpAnnotation stmtAnnot = stmt.getAnnotation(OmpAnnotation.class, "teams");
    							if(stmtAnnot != null) {
    								if( condExp != null ) {
    									//If target region has if clause, teams region should be executed conditionally too.
    									stmtAnnot.put("if", condExp);
    								}
    							} else if( stmt.containsAnnotation(OmpAnnotation.class, "distribute") ) {
    								stmtAnnot = stmt.getAnnotation(OmpAnnotation.class, "distribute");
    								if( condExp != null ) {
    									stmtAnnot.put("if", condExp);
    								}
    							} else if( stmt.containsAnnotation(OmpAnnotation.class, "parallel") ) {
    								stmtAnnot = stmt.getAnnotation(OmpAnnotation.class, "parallel");
    								if( condExp != null ) {
    									stmtAnnot.put("if", condExp);
    								}
    							} else if( stmt.containsAnnotation(OmpAnnotation.class, "simd") ) {
    								stmtAnnot = stmt.getAnnotation(OmpAnnotation.class, "simd");
    								if( condExp != null ) {
    									stmtAnnot.put("if", condExp);
    								}
    							} else {
    								List<OmpAnnotation> ompList = stmt.getAnnotations(OmpAnnotation.class);
    								if( (ompList != null) && (ompList.size() > 0) ) {
    									Tools.exit("[ERROR in OMP4toACCTranslator] target construct allows only teams, parallel, and simd constructs in its region, but "
    											+ "the following target region contains other OpenMP constructs:\n"
    											+ "OmpAnnotation: " + pragma + "\n"
    											+ AnalysisTools.getEnclosingAnnotationContext(pragma));
    								}
    							}

    							//TODO: if clause should be passed to the enclosed distribute/parallel/simd directives too.
    						}
    					}

    					parse_target_pragma(pragma, OmpRegion.Target);
    				}
    			}
    		}
    		if( !insertRefList.isEmpty() ) {
    			for( int k=0; k<insertRefList.size(); k++ ) {
    				Statement refStmt = insertRefList.get(k);
    				Statement taskCStmt = insertTargetList.get(k);
    				((CompoundStatement)refStmt.getParent()).addStatementAfter(
    						refStmt, taskCStmt);
    			}
    		}
    		
    		for( OmpAnnotation oAnnot : TargetRegionAnnots ) {
    			Annotatable targetRegion = oAnnot.getAnnotatable();
    			ARCAnnotation devTaskAnnot = targetRegion.getAnnotation(ARCAnnotation.class, "devicetask");
    			String taskMapping = null;
    			int taskMappingType = 0;
    			if( devTaskAnnot != null ) {
    				taskMapping = devTaskAnnot.get("map");
    				if( taskMapping != null ) { 
    					if( taskMapping.equals("coarse_grained") ) {
    						taskMappingType = 1;
    						continue;
    					} else if( taskMapping.equals("fine_grained") ) {
    						taskMappingType = 2;
    						if( !createDeviceTask ) {
    							TranslationUnit newTrUnt = new TranslationUnit("juggler_app.h");
    							program.addTranslationUnit(newTrUnt);
    							createDeviceTask = true;
    						}
    					}
    				}
    			}
    			PrintTools.println("Enter target region", 2);
    			DFIterator<Annotatable> titer = new DFIterator<Annotatable>(targetRegion, Annotatable.class);
    			while (titer.hasNext()) 
    			{
    				Annotatable at = titer.next();
    				List<OmpAnnotation> pragmas = at.getAnnotations(OmpAnnotation.class);
    				if( pragmas != null ) {
    					for (int i = 0; i < pragmas.size(); i++) 
    					{
    						OmpAnnotation pragma = pragmas.get(i);
    						if( !(at instanceof ForLoop) ) {
    							String directive = null;
    							if (pragma.containsKey("distribute")) {
    								directive = "distribute";
    							} else if (pragma.containsKey("for")) {
    								directive = "for";
    							} else if (pragma.containsKey("simd")) {
    								directive = "simd";
    							} 
    							if (directive != null) {
    								Tools.exit("[ERROR in OMP4toACCTranslator.start()] An OpenMP "+ directive + " directive is allowed only for for-loops, "
    										+ "but the following construct illegally uses the " + directive + " directive:\n"
    										+ "OpenMP construct: " + at + "\n" + AnalysisTools.getEnclosingContext(at));
    							}
    						}

    						if (pragma.containsKey("teams")) {
    							if( taskMappingType == 0 ) {
    								//Enter Teams region
    								PrintTools.println("Enter teams region", 2);
    								parse_teams_pragma(pragma);
    							} else {
    								pragma.setSkipPrint(true);
    								removeList.add(pragma);
    							}
    						}

    						if (pragma.containsKey("distribute")) {
    							if( taskMappingType == 0 ) {
    								//Enter Distribute region
    								PrintTools.println("Enter distribute region", 2);
    								parse_distribute_pragma(pragma);
    							} else {
    								pragma.setSkipPrint(true);
    								removeList.add(pragma);
    							}
    						} else if (pragma.containsKey("parallel")) {
    							if( taskMappingType == 0 ) {
    								//Enter Parallel region
    								PrintTools.println("Enter parallel region", 2);
    								parse_parallel_pragma(pragma);
    							} else {
    								pragma.setSkipPrint(true);
    								removeList.add(pragma);
    							}
    						} else if (pragma.containsKey("for")) {
    							if( taskMappingType == 0 ) {
    								//Enter for region
    								PrintTools.println("Enter for region", 2);
    								parse_for_pragma(pragma);
    							} else {
    								pragma.setSkipPrint(true);
    								removeList.add(pragma);
    							}
    						} else if (pragma.containsKey("simd")) {
    							if( taskMappingType == 0 ) {
    								//Enter SIMD region
    								PrintTools.println("Enter simd region", 2);
    								parse_simd_pragma(pragma);
    							} else {
    								pragma.setSkipPrint(true);
    								removeList.add(pragma);
    							}
    						}

    						if (pragma.containsKey("atomic")) {
    							if( taskMappingType == 0 ) {
    								parse_atomic(pragma);
    							} else {
    								pragma.setSkipPrint(true);
    								removeList.add(pragma);
    							}
    						}
    					}
    				}
    			}
    		}

    		if( removeList.size() > 0 ) {
    			for( OmpAnnotation oAnnot : removeList ) {
    				//oAnnot.setSkipPrint(false);
    				//System.out.println("annotataion to remove: " + oAnnot);
    				Annotatable at = oAnnot.getAnnotatable();
    				if( (at != null) && (at.getParent() != null) ) {
    					at.removeAnnotations(OmpAnnotation.class);
/*    					List<OmpAnnotation> ompAnnotList = at.getAnnotations(OmpAnnotation.class);
    					at.removeAnnotations(OmpAnnotation.class);
    					for( OmpAnnotation ompAnnot : ompAnnotList ) {
    						if( !oAnnot.equals(ompAnnot) ) {
    							at.annotate(ompAnnot);
    						}
    					}*/
    					List<ACCAnnotation> accAnnotList = at.getAnnotations(ACCAnnotation.class);
    					if( (at instanceof AnnotationStatement) && ((accAnnotList ==null) || (accAnnotList.isEmpty())) ) {
    						CompoundStatement cStmt = (CompoundStatement)at.getParent();
    						cStmt.removeChild(at);
    					}
    				}
    			}
    			removeList.clear();
    		}
    	}
    	
    	for( FunctionCall fCall : funcCallList ) {
    		String fName = fCall.getName().toString();
    		if( fName.equals("omp_target_alloc") ) {
    			//Replace "void *omp_target_alloc(size_t size, int device_num);" with 
    			//"void acc_set_device_num(int, acc_device_t);"
    			//"d_void *acc_malloc(size_t);"
    			FunctionCall newFCall = null;
    			Symbol newFCallSym = SymbolTools.getSymbolOfName("acc_malloc", fCall);
    			if( newFCallSym == null ) {
    				newFCall = new FunctionCall(new NameID("acc_malloc"));
    			} else {
    				newFCall = new FunctionCall(new Identifier(newFCallSym));
    			}
    			newFCall.addArgument(fCall.getArgument(0).clone());
    			fCall.swapWith(newFCall);
    			Traversable tt = newFCall.getParent();
    			while( (tt != null) && !(tt instanceof Statement) ) {
    				tt = tt.getParent();
    			}
    			if( tt instanceof Statement ) {
    				Statement tStmt = (Statement)tt;
    				CompoundStatement ctStmt = (CompoundStatement)tStmt.getParent();
    				FunctionCall newFCall2 = null;
    				Symbol newFCallSym2 = SymbolTools.getSymbolOfName("acc_set_device_num", ctStmt);
    				if( newFCallSym2 == null ) {
    					newFCall2 = new FunctionCall(new NameID("acc_set_device_num"));
    				} else {
    					newFCall2 = new FunctionCall(new Identifier(newFCallSym2));
    				}
    				newFCall2.addArgument(fCall.getArgument(1).clone());
    				newFCall2.addArgument(new NameID("acc_device_current"));
    				Statement tStmt2 = new ExpressionStatement(newFCall2);
    				ctStmt.addStatementBefore(tStmt, tStmt2);
    			} else {
    				Tools.exit("[ERROR in OMP4toACCTranslator.start()] fail to convert omp_target_alloc() funcion; exit!");
    			}
    		} else if( fName.equals("omp_target_free") ) {
    			//Replace "void omp_target_free(void * device_ptr, int device_num);" with 
    			//"void acc_set_device_num(int, acc_device_t);"
    			//"void acc_free(d_void*);"
    			FunctionCall newFCall = null;
    			Symbol newFCallSym = SymbolTools.getSymbolOfName("acc_free", fCall);
    			if( newFCallSym == null ) {
    				newFCall = new FunctionCall(new NameID("acc_free"));
    			} else {
    				newFCall = new FunctionCall(new Identifier(newFCallSym));
    			}
    			newFCall.addArgument(fCall.getArgument(0).clone());
    			fCall.swapWith(newFCall);
    			Traversable tt = newFCall.getParent();
    			while( (tt != null) && !(tt instanceof Statement) ) {
    				tt = tt.getParent();
    			}
    			if( tt instanceof Statement ) {
    				Statement tStmt = (Statement)tt;
    				CompoundStatement ctStmt = (CompoundStatement)tStmt.getParent();
    				FunctionCall newFCall2 = null;
    				Symbol newFCallSym2 = SymbolTools.getSymbolOfName("acc_set_device_num", ctStmt);
    				if( newFCallSym2 == null ) {
    					newFCall2 = new FunctionCall(new NameID("acc_set_device_num"));
    				} else {
    					newFCall2 = new FunctionCall(new Identifier(newFCallSym2));
    				}
    				newFCall2.addArgument(fCall.getArgument(1).clone());
    				newFCall2.addArgument(new NameID("acc_device_current"));
    				Statement tStmt2 = new ExpressionStatement(newFCall2);
    				ctStmt.addStatementBefore(tStmt, tStmt2);
    			} else {
    				Tools.exit("[ERROR in OMP4toACCTranslator.start()] fail to convert omp_target_free() funcion; exit!");
    			}
    		} else if( fName.equals("omp_target_is_present") ) {
    			//Replace "void omp_target_is_present(void * ptr, int device_num);" with 
    			//"void acc_set_device_num(int, acc_device_t);"
    			//"void acc_is_present(h_void*, size_t);"
    			FunctionCall newFCall = null;
    			Symbol newFCallSym = SymbolTools.getSymbolOfName("acc_is_present", fCall);
    			if( newFCallSym == null ) {
    				newFCall = new FunctionCall(new NameID("acc_is_present"));
    			} else {
    				newFCall = new FunctionCall(new Identifier(newFCallSym));
    			}
    			newFCall.addArgument(fCall.getArgument(0).clone());
    			//[FIXME] Valid data size should be passed!
    			newFCall.addArgument(new IntegerLiteral(0));
    			PrintTools.println("[WARNING in OMP4toACCTranslator.start()] converting omp_target_is_present() function is incomplete; "
    					+ "it may not work with other OpenACC implementation!", 0);
    			fCall.swapWith(newFCall);
    			Traversable tt = newFCall.getParent();
    			while( (tt != null) && !(tt instanceof Statement) ) {
    				tt = tt.getParent();
    			}
    			if( tt instanceof Statement ) {
    				Statement tStmt = (Statement)tt;
    				CompoundStatement ctStmt = (CompoundStatement)tStmt.getParent();
    				FunctionCall newFCall2 = null;
    				Symbol newFCallSym2 = SymbolTools.getSymbolOfName("acc_set_device_num", ctStmt);
    				if( newFCallSym2 == null ) {
    					newFCall2 = new FunctionCall(new NameID("acc_set_device_num"));
    				} else {
    					newFCall2 = new FunctionCall(new Identifier(newFCallSym2));
    				}
    				newFCall2.addArgument(fCall.getArgument(1).clone());
    				newFCall2.addArgument(new NameID("acc_device_current"));
    				Statement tStmt2 = new ExpressionStatement(newFCall2);
    				ctStmt.addStatementBefore(tStmt, tStmt2);
    			} else {
    				Tools.exit("[ERROR in OMP4toACCTranslator.start()] fail to convert omp_target_is_present() funcion; exit!");
    			}
    		} else if( fName.equals("omp_target_associate_ptr") ) {
    			//Replace "void omp_target_associate_ptr(void * host_ptr, void * device_ptr, size_t size, size_t device_offset, int device_num);" with 
    			//"void acc_set_device_num(int, acc_device_t);"
    			//"void acc_map_data(h_void*, d_void*, size_t);"
    			FunctionCall newFCall = null;
    			Symbol newFCallSym = SymbolTools.getSymbolOfName("acc_map_data", fCall);
    			if( newFCallSym == null ) {
    				newFCall = new FunctionCall(new NameID("acc_map_data"));
    			} else {
    				newFCall = new FunctionCall(new Identifier(newFCallSym));
    			}
    			newFCall.addArgument(fCall.getArgument(0).clone());
    			newFCall.addArgument(new BinaryExpression(fCall.getArgument(1).clone(), BinaryOperator.ADD, fCall.getArgument(3)));
    			newFCall.addArgument(fCall.getArgument(2).clone());
    			fCall.swapWith(newFCall);
    			Traversable tt = newFCall.getParent();
    			while( (tt != null) && !(tt instanceof Statement) ) {
    				tt = tt.getParent();
    			}
    			if( tt instanceof Statement ) {
    				Statement tStmt = (Statement)tt;
    				CompoundStatement ctStmt = (CompoundStatement)tStmt.getParent();
    				FunctionCall newFCall2 = null;
    				Symbol newFCallSym2 = SymbolTools.getSymbolOfName("acc_set_device_num", ctStmt);
    				if( newFCallSym2 == null ) {
    					newFCall2 = new FunctionCall(new NameID("acc_set_device_num"));
    				} else {
    					newFCall2 = new FunctionCall(new Identifier(newFCallSym2));
    				}
    				newFCall2.addArgument(fCall.getArgument(4).clone());
    				newFCall2.addArgument(new NameID("acc_device_current"));
    				Statement tStmt2 = new ExpressionStatement(newFCall2);
    				ctStmt.addStatementBefore(tStmt, tStmt2);
    			} else {
    				Tools.exit("[ERROR in OMP4toACCTranslator.start()] fail to convert omp_target_associate_ptr() funcion; exit!");
    			}
    		} else if( fName.equals("omp_target_disassociate_ptr") ) {
    			//Replace "void omp_target_disassociate_ptr(void * ptr, int device_num);" with 
    			//"void acc_set_device_num(int, acc_device_t);"
    			//"void acc_unmap(h_void*);"
    			FunctionCall newFCall = null;
    			Symbol newFCallSym = SymbolTools.getSymbolOfName("acc_unmap_data", fCall);
    			if( newFCallSym == null ) {
    				newFCall = new FunctionCall(new NameID("acc_unmap_data"));
    			} else {
    				newFCall = new FunctionCall(new Identifier(newFCallSym));
    			}
    			newFCall.addArgument(fCall.getArgument(0).clone());
    			fCall.swapWith(newFCall);
    			Traversable tt = newFCall.getParent();
    			while( (tt != null) && !(tt instanceof Statement) ) {
    				tt = tt.getParent();
    			}
    			if( tt instanceof Statement ) {
    				Statement tStmt = (Statement)tt;
    				CompoundStatement ctStmt = (CompoundStatement)tStmt.getParent();
    				FunctionCall newFCall2 = null;
    				Symbol newFCallSym2 = SymbolTools.getSymbolOfName("acc_set_device_num", ctStmt);
    				if( newFCallSym2 == null ) {
    					newFCall2 = new FunctionCall(new NameID("acc_set_device_num"));
    				} else {
    					newFCall2 = new FunctionCall(new Identifier(newFCallSym2));
    				}
    				newFCall2.addArgument(fCall.getArgument(1).clone());
    				newFCall2.addArgument(new NameID("acc_device_current"));
    				Statement tStmt2 = new ExpressionStatement(newFCall2);
    				ctStmt.addStatementBefore(tStmt, tStmt2);
    			} else {
    				Tools.exit("[ERROR in OMP4toACCTranslator.start()] fail to convert omp_target_disassociate_ptr() funcion; exit!");
    			}
    		} else if( fName.equals("omp_target_memcpy") ) {
    			//Replace "void omp_target_memcpy(void * dst, void * src, size_t length, size_t dst_offset, size_t src_offset, int dst_device_num, int src_dev_num);" with 
    			//??
    			Tools.exit("[ERROR in OMP4toACCTranslator.start()] fail to convert omp_target_memcpy() funcion; exit!");
    		} else if( fName.equals("omp_target_memcpy_rect") ) {
    			Tools.exit("[ERROR in OMP4toACCTranslator.start()] fail to convert omp_target_memcpy_rect() funcion; exit!");
    			//int omp_target_memcpy_rect(
    			//void * dst, void * src,
    			//size_t element_size,
    			//int num_dims,
    			//const size_t* volume,
    			//const size_t* dst_offsets,
    			//const size_t* src_offsets,
    			//const size_t* dst_dimensions,
    			//const size_t* src_dimensions,
    			//int dst_device_num, int src_device_num);
    		}
    	}
    }

    public void start_old() 
    {
    	HashMap<String, String> gMacroMap = ACCAnnotationParser.parseCmdLineMacros();

    	convertCritical2Reduction();

    	for( Traversable trUnt : program.getChildren() ) {
    		macroMap = new HashMap<String, String>();
    		macroMap.putAll(gMacroMap); //Add global commandline macros.
    		DFIterator<Annotatable> iter = new DFIterator<Annotatable>(trUnt, Annotatable.class);
    		while (iter.hasNext()) 
    		{
    			Annotatable at = iter.next();

    			if(parentStack.size() > 0)
    			{
    				/**
    				 * Check if new annotatable has same parent as the current region.
    				 * If it's true then the iterator exit the region.
    				 */
    				if(parentStack.contains(at.getParent()))
    				{
    					//PrintTools.println(at.toString(), 0);

    					while(parentStack.size() > 0 && parentStack.peek() != at.getParent())
    					{
    						PrintTools.println("Exit " + regionStack.peek() + " region", 2);
    						parentStack.pop();
    						regionStack.pop();
    					}

    					while(parentStack.size() > 0 && parentStack.peek() == at.getParent())
    					{
    						PrintTools.println("Exit " + regionStack.peek() + " region", 2);
    						parentStack.pop();
    						regionStack.pop();
    					}

    				}
    			}

				////////////////////////////////////////////////////////////////////////
				// AnnotationParser store OpenACC annotations as PragmaAnnotations in //
				// either AnnotationStatements or AnnotationDeclaration, depending on //
				// their locations.                                                   //
				////////////////////////////////////////////////////////////////////////
				if ((at instanceof AnnotationStatement) || (at instanceof AnnotationDeclaration) )
				{
					List<PragmaAnnotation> annot_list = 
						at.getAnnotations(PragmaAnnotation.class);
					if( (annot_list == null) || (annot_list.size() == 0) ) {
						continue;
					}
					////////////////////////////////////////////////////////////////////////////
					// AnnotationParser creates one AnnotationStatement/AnnotationDeclaration //
					// for each PragmaAnnotation.                                             //
					////////////////////////////////////////////////////////////////////////////
					PragmaAnnotation pAnnot = annot_list.get(0);
					//////////////////////////////////////////////////////////////////////////
					// The above pAnnot may be a derived annotation.                        //
					// If so, skip it, except for OmpAnnotation.                            //
					// (The below annotations are child classes of PragmaAnnotation.)       //
					// DEBUG: If new child class of the PragmaAnnotation is added, below    //
					// should be updated too.                                               //
					// (ex: If CudaAnnotation is added, it should be checked here.)         //
					//////////////////////////////////////////////////////////////////////////
					if( pAnnot instanceof CetusAnnotation || pAnnot instanceof ASPENAnnotation ||
							pAnnot instanceof InlineAnnotation || pAnnot instanceof PragmaAnnotation.Event ||
							pAnnot instanceof PragmaAnnotation.Range || pAnnot instanceof ACCAnnotation ||
							pAnnot instanceof ARCAnnotation || pAnnot instanceof NVLAnnotation ) {
						continue;
					}
					String old_annot = pAnnot.getName();
					if( old_annot == null ) {
						if( !(pAnnot instanceof OmpAnnotation) ) {
							PrintTools.println("\n[WARNING in OMP4toACCTranslator] Pragma annotation, "
								+ pAnnot +", does not have name.\n", 0);
							continue;
						}
					} else {
						old_annot = ACCAnnotationParser.modifyAnnotationString(old_annot);
						String[] token_array = old_annot.split("\\s+");
						// If old_annot string has a leading space, the 2nd token should be checked.
						//String old_annot_key = token_array[1];
						String old_annot_key = token_array[0];
						if ((old_annot_key.compareTo("openarc")==0)) {
							String old_annot_key2 = token_array[1];
							if( old_annot_key2.equals("#") ) { //Preprocess macros on OpenACC/OpenARC directives.
								ACCParser.preprocess_acc_pragma(token_array, macroMap);
								continue;
							}
						}
					}
				}

    			List<OmpAnnotation> pragmas = at.getAnnotations(OmpAnnotation.class);
    			for (int i = 0; i < pragmas.size(); i++) 
    			{
    				OmpAnnotation pragma = pragmas.get(i);
    				if(pragma.containsKey("declare"))
    				{
    					String declareStr = pragma.get("declare");
    					if( declareStr.equals("target") ) {
    						if( !pragma.containsKey("end") ) {
    							//#pragma omp declare target
    							parse_declare_target(pragma);
    						}
    					} else if( declareStr.equals("simd") ) {
    						//#pragma omp declare simd clause, clause ...
    					}
    					continue;
    				}
    				if(pragma.containsKey("target") && pragma.containsKey("update"))
    				{
    					parse_target_pragma(pragma, OmpRegion.Target_Update);
    					continue;
    				}
    				if(pragma.containsKey("target") && pragma.containsKey("enter") && pragma.containsKey("data"))
    				{
    					parse_target_pragma(pragma, OmpRegion.Target_Enter_Data);
    					continue;
    				}
    				if(pragma.containsKey("target") && pragma.containsKey("exit") && pragma.containsKey("data"))
    				{
    					parse_target_pragma(pragma, OmpRegion.Target_Exit_Data);
    					continue;
    				}
    				if(pragma.containsKey("target") && pragma.containsKey("data"))
    				{
    					parse_target_pragma(pragma, OmpRegion.Target_Data);
    					continue;
    				}

    				if(pragma.containsKey("target"))
    				{
    					//Enter Target region
    					parentStack.push(at.getParent());
    					regionStack.push(OmpRegion.Target);
    					PrintTools.println("Enter " + regionStack.peek() + " region", 2);
    					Object condExp = null;
    					if( pragma.containsKey("if") ) {
    						condExp = pragma.get("if");
    					}
    					if( at instanceof CompoundStatement ) {
    						FlatIterator<Statement> flatIterator = new FlatIterator<Statement>(at);
    						while (flatIterator.hasNext())
    						{
    							Statement stmt = flatIterator.next();
    							OmpAnnotation stmtAnnot = stmt.getAnnotation(OmpAnnotation.class, "teams");
    							if(stmtAnnot != null) {
    								if( condExp != null ) {
    									//If target region has if clause, teams region should be executed conditionally too.
    									stmtAnnot.put("if", condExp);
    								}
    							} else if( stmt.containsAnnotation(OmpAnnotation.class, "distribute") ) {
    								stmtAnnot = stmt.getAnnotation(OmpAnnotation.class, "distribute");
    								if( condExp != null ) {
    									stmtAnnot.put("if", condExp);
    								}
    							} else if( stmt.containsAnnotation(OmpAnnotation.class, "parallel") ) {
    								stmtAnnot = stmt.getAnnotation(OmpAnnotation.class, "parallel");
    								if( condExp != null ) {
    									stmtAnnot.put("if", condExp);
    								}
    							} else if( stmt.containsAnnotation(OmpAnnotation.class, "simd") ) {
    								stmtAnnot = stmt.getAnnotation(OmpAnnotation.class, "simd");
    								if( condExp != null ) {
    									stmtAnnot.put("if", condExp);
    								}
    							} else {
    								List<OmpAnnotation> ompList = stmt.getAnnotations(OmpAnnotation.class);
    								if( (ompList != null) && (ompList.size() > 0) ) {
    									Tools.exit("[ERROR in OMP4toACCTranslator] target construct allows only teams, parallel, and simd constructs in its region, but "
    											+ "the following target region contains other OpenMP constructs:\n"
    											+ "OmpAnnotation: " + pragma + "\n"
    											+ AnalysisTools.getEnclosingAnnotationContext(pragma));
    								}
    							}

    							//TODO: if clause should be passed to the enclosed distribute/parallel/simd directives too.
    						}
    					}

    					parse_target_pragma(pragma, OmpRegion.Target);
    				}

    				if(regionStack.contains(OmpRegion.Target)) {
    					if (pragma.containsKey("teams")) {
    						//Enter Teams region
    						parentStack.push(at.getParent());
    						regionStack.push(OmpRegion.Teams);
    						PrintTools.println("Enter " + regionStack.peek() + " region", 2);
    						parse_teams_pragma(pragma);
    					}

    					if (pragma.containsKey("distribute")) {
    						//Enter Distribute region
    						parentStack.push(at.getParent());
    						regionStack.push(OmpRegion.Distribute);
    						PrintTools.println("Enter " + regionStack.peek() + " region",2);
    						parse_distribute_pragma(pragma);
    					} else if (pragma.containsKey("parallel") && pragma.containsKey("for")) {
    						//Enter Parallel For region
    						parentStack.push(at.getParent());
    						regionStack.push(OmpRegion.Parallel_For);
    						PrintTools.println("Enter " + regionStack.peek() + " region", 2);
    						parse_parallel_pragma(pragma);
    					} else if (pragma.containsKey("simd")) {
    						//Enter SIMD region
    						parentStack.push(at.getParent());
    						regionStack.push(OmpRegion.SIMD);
    						PrintTools.println("Enter " + regionStack.peek() + " region", 2);
    						parse_simd_pragma(pragma);
    					}

    					if (pragma.containsKey("atomic")) {
    						parse_atomic(pragma);
    					}
    				}
    			}
    		}

    		if( removeList.size() > 0 ) {
    			for( OmpAnnotation oAnnot : removeList ) {
    				Annotatable at = oAnnot.getAnnotatable();
    				if( (at != null) && (at.getParent() != null) ) {
    					at.removeAnnotations(OmpAnnotation.class);
    					if( at instanceof AnnotationStatement ) {
    						CompoundStatement cStmt = (CompoundStatement)at.getParent();
    						cStmt.removeChild(at);
    					}
    				}
    			}
    			removeList.clear();
    		}
    	}
    }
   

    private void parse_atomic(OmpAnnotation pragma)
    {        
        ACCAnnotation accAnnot = new ACCAnnotation();
        accAnnot.put("atomic", "_directive");

        if(pragma.containsKey("type"))
        {
            accAnnot.put((String)pragma.get("type"), "_clause");
        }
        else
        {
            accAnnot.put("update", "_directive");           
        }
        pragma.getAnnotatable().annotate(accAnnot);
        pragma.setSkipPrint(true);
        removeList.add(pragma);
    }

    private void parse_for_pragma(OmpAnnotation pragma)
    {
    	Set<SubArray> privateSet = new HashSet<SubArray>();
    	Set<SubArray> linearSet = new HashSet<SubArray>();
    	Set<SubArray> firstprivateSet = new HashSet<SubArray>();
    	ACCAnnotation accAnnot = pragma.getAnnotatable().getAnnotation(ACCAnnotation.class, "loop");
    	if( accAnnot == null ) {
    		accAnnot = new ACCAnnotation();
    		pragma.getAnnotatable().annotate(accAnnot);
    		accAnnot.put("loop", "_directive");
    	}
    	if( !accAnnot.containsKey("worker") ) {
    		accAnnot.put("worker", "_clause");
    	}
    	if(pragma.containsKey("private")) {
    		parseSet((Set<String>)pragma.get("private"), privateSet);
    		pragma.remove("private");
    	}
    	if(pragma.containsKey("linear")) {
    		parseSet((Set<String>)pragma.get("linear"), linearSet);
    		pragma.remove("linear");
    	}
    	if(pragma.containsKey("firstprivate")) {
    		parseSet((Set<String>)pragma.get("firstprivate"), firstprivateSet);
    		pragma.remove("firstprivate");
    	}

    	if(privateSet.size() > 0) {
    		Set<SubArray> oldPrivateSet = accAnnot.get("private");
    		if( oldPrivateSet != null ) {
    			oldPrivateSet.addAll(privateSet);
    		} else {
    			accAnnot.put("private", privateSet);
    		}
    	}
    	if(linearSet.size() > 0) {
        	//[FIXME] Semantics of OpenMP linear clause is different from that of OpenACC private clause.
        	//OpenMP Specification Version 5.0, Section 2.19.4.6 linear Clause says that the value corresponding to 
        	//the sequentially last iteration of the associated loop(s) is assigned to the original list item.
            PrintTools.println("[WARNING in OMP4toACCTranslator.parse_for_pragma()] the current translator converts "
            		+ "OpenMP linear clause to OpenACC private clause, which may break the original OpenMP semantics." +
                                        AnalysisTools.getEnclosingAnnotationContext(pragma), 0);
    		Set<SubArray> oldPrivateSet = accAnnot.get("private");
    		if( oldPrivateSet != null ) {
    			oldPrivateSet.addAll(linearSet);
    		} else {
    			accAnnot.put("private", linearSet);
    		}
    	}
    	if(firstprivateSet.size() > 0) {
    		Set<SubArray> oldFirstPrivateSet = accAnnot.get("firstprivate");
    		if( oldFirstPrivateSet != null ) {
    			oldFirstPrivateSet.addAll(firstprivateSet);
    		} else {
    			accAnnot.put("firstprivate", firstprivateSet);
    		}
    	}
    	if(pragma.containsKey("collapse")) {
    		accAnnot.put("collapse", ParserTools.strToExpression((String) pragma.get("collapse")));
    		pragma.remove("collapse");
    	}

        pragma.setSkipPrint(true);
        removeList.add(pragma);
    }

    private void parse_simd_pragma(OmpAnnotation pragma)
    {
    	ACCAnnotation accAnnot = pragma.getAnnotatable().getAnnotation(ACCAnnotation.class, "loop");
    	if( accAnnot == null ) {
    		accAnnot = new ACCAnnotation();
    		pragma.getAnnotatable().annotate(accAnnot);
    		accAnnot.put("loop", "_directive");
    	}
    	if( !accAnnot.containsKey("vector") ) {
    		accAnnot.put("vector", "_clause");
    	}
    	if(pragma.containsKey("collapse")) {
    		if(!accAnnot.containsKey("collapse")) {
    			accAnnot.put("collapse", ParserTools.strToExpression((String) pragma.get("collapse")));
    		}
    		pragma.remove("collapse");
    	}

        pragma.setSkipPrint(true);
        removeList.add(pragma);
    }

    private void parse_parallel_pragma(OmpAnnotation pragma)
    {
    	Set<SubArray> privateSet = new HashSet<SubArray>();
    	Set<SubArray> linearSet = new HashSet<SubArray>();
    	Set<SubArray> firstprivateSet = new HashSet<SubArray>();
    	Expression condition = null;
    	Expression num_threads = null;
        //Handle condition passed from enclosing target region.
        if(pragma.containsKey("if"))
        {
        	String condStr = pragma.get("if");
            condition =  ParserTools.strToExpression(condStr);
            pragma.remove("if");
        }

        if(pragma.containsKey("num_threads"))
        {
        	String wStr = pragma.get("num_threads");
            num_threads =  ParserTools.strToExpression(wStr);
            //pragma.remove("num_threads");
        }

    	Annotatable at = pragma.getAnnotatable();
        ACCAnnotation accAnnot = at.getAnnotation(ACCAnnotation.class, "parallel");
        ACCAnnotation loopAnnot = at.getAnnotation(ACCAnnotation.class, "loop");
        ACCAnnotation currentAnnot = accAnnot;
        OmpAnnotation ompTargetAnnot = at.getAnnotation(OmpAnnotation.class, "target");
        
        if( (ompTargetAnnot != null) && (accAnnot == null) ) {
        	//Outline each task in the parallel region as separate kernels, instead of the current target region,
        	//which has already been handled by parse_target_pragma().
        	ompTargetAnnot.remove("target");
        	ompTargetAnnot.remove("device");
        	ompTargetAnnot.remove("nowait");
        	ompTargetAnnot.remove("in");
        	ompTargetAnnot.remove("out");
        	ompTargetAnnot.remove("inout");
        	ompTargetAnnot.remove("to");
        	ompTargetAnnot.remove("from");
        	ompTargetAnnot.remove("tofrom");
        	ompTargetAnnot.remove("alloc");
        	ompTargetAnnot.remove("release");
        	ompTargetAnnot.remove("delete");
        } else {
        	if( pragma.containsKey("for") || pragma.containsKey("simd") ) {
        		if( loopAnnot == null ) {
        			if( accAnnot == null ) {
        				loopAnnot = new ACCAnnotation();
        				at.annotate(loopAnnot);
        			} else {
        				loopAnnot = accAnnot;
        			}
        			loopAnnot.put("loop", "_directive");
        		}
        		if(pragma.containsKey("for")) {
        			if( !loopAnnot.containsKey("worker") ) {
        				loopAnnot.put("worker", "_clause");
        			}
        			//pragma.remove("for");
        		}
        		if(pragma.containsKey("simd")) {
        			if( !loopAnnot.containsKey("vector") ) {
        				loopAnnot.put("vector", "_clause");
        			}
        			//pragma.remove("simd");
        		}
        	}
        	if( accAnnot == null ) {
        		currentAnnot = loopAnnot;
        	}
        	if( currentAnnot == null ) {
        		//[FIXME] If a target region contains a parallel region that is not a loop, we can ignore most of clauses, but private and firstprivate
        		//clauses should be handled correctly in this pass, but implemented yet.
        		Tools.exit("[ERROR in OMP4toACCTranslator.parse_parallel_pragma()]  the following OpenMP parallel region cannot be handled; exit!\n"
        				+ "OpenMP construct: " + at + "\n" + AnalysisTools.getEnclosingContext(at));
        	}

        	//Add implicit teams construct
        	//Deprecated; this will work with start_old();
        	/*    	if(!regionStack.contains(OmpRegion.Teams))
    		accAnnot.put("parallel", "_directive");*/
        	/*    	if( !at.containsAnnotation(OmpAnnotation.class, "teams") ) {
    		OmpAnnotation tAnnot = AnalysisTools.ipFindFirstPragmaInParent(at, OmpAnnotation.class, "teams", funcCallList, null);
    		if( tAnnot == null || tAnnot.isEmpty() ) {
    			accAnnot.put("parallel", "_directive");
    		}
    	}*/

        	if(pragma.containsKey("private")) {
        		parseSet((Set<String>)pragma.get("private"), privateSet);
        		pragma.remove("private");
        	}
        	if(pragma.containsKey("linear")) {
        		parseSet((Set<String>)pragma.get("linear"), linearSet);
        		pragma.remove("linear");
        	}
        	if(pragma.containsKey("firstprivate")) {
        		parseSet((Set<String>)pragma.get("firstprivate"), firstprivateSet);
        		pragma.remove("firstprivate");
        	}

        	if(privateSet.size() > 0) {
        		Set<SubArray> oldPrivateSet = currentAnnot.get("private");
        		if( oldPrivateSet != null ) {
        			oldPrivateSet.addAll(privateSet);
        		} else {
        			currentAnnot.put("private", privateSet);
        		}
        	}
        	if(linearSet.size() > 0) {
        		//[FIXME] Semantics of OpenMP linear clause is different from that of OpenACC private clause.
        		//OpenMP Specification Version 5.0, Section 2.19.4.6 linear Clause says that the value corresponding to 
        		//the sequentially last iteration of the associated loop(s) is assigned to the original list item.
        		PrintTools.println("[WARNING in OMP4toACCTranslator.parse_parallel_pragma()] the current translator converts "
        				+ "OpenMP linear clause to OpenACC private clause, which may break the original OpenMP semantics." +
        				AnalysisTools.getEnclosingAnnotationContext(pragma), 0);
        		Set<SubArray> oldPrivateSet = currentAnnot.get("private");
        		if( oldPrivateSet != null ) {
        			oldPrivateSet.addAll(linearSet);
        		} else {
        			currentAnnot.put("private", linearSet);
        		}
        	}
        	if(firstprivateSet.size() > 0) {
        		Set<SubArray> oldFirstPrivateSet = currentAnnot.get("firstprivate");
        		if( oldFirstPrivateSet != null ) {
        			oldFirstPrivateSet.addAll(firstprivateSet);
        		} else {
        			currentAnnot.put("firstprivate", firstprivateSet);
        		}
        	}
        	if(pragma.containsKey("collapse")) {
        		currentAnnot.put("collapse", ParserTools.strToExpression((String) pragma.get("collapse")));
        		pragma.remove("collapse");
        	}
        	if(condition != null) {
        		currentAnnot.put("if", condition);
        	}
        	if(num_threads != null) {
        		if( currentAnnot.containsKey("num_workers") ) {
        			Expression num_workers = currentAnnot.get("num_workers");
        			if( (num_workers instanceof IntegerLiteral) && (num_threads instanceof IntegerLiteral) ) {
        				long t1 = ((IntegerLiteral)num_workers).getValue();
        				long t2 = ((IntegerLiteral)num_threads).getValue();
        				if( t2 < t1 ) {
        					currentAnnot.put("num_workers", num_threads);
        				}
        			} else {
        				//[DEBUG] conservatively use thread_limit value.
        			}
        		} else {
        			currentAnnot.put("num_workers", num_threads);
        		}
        	}

        	if(pragma.containsKey("reduction")) {
        		Map<String, Set> reduction_map = pragma.get("reduction");
        		parseReduction(currentAnnot, reduction_map);
        		pragma.remove("reduction");
        	}

        	pragma.setSkipPrint(true);
        	removeList.add(pragma);

        }

    }

    private void parse_distribute_pragma(OmpAnnotation pragma)
    {
        Set<SubArray> privateSet = new HashSet<SubArray>();
        Set<SubArray> linearSet = new HashSet<SubArray>();
        Set<SubArray> firstprivateSet = new HashSet<SubArray>();

        ACCAnnotation accAnnot = pragma.getAnnotatable().getAnnotation(ACCAnnotation.class, "loop");
        if( accAnnot == null ) {
        	accAnnot = pragma.getAnnotatable().getAnnotation(ACCAnnotation.class, "parallel");
        	if( accAnnot == null ) {
        		accAnnot = new ACCAnnotation();
        		pragma.getAnnotatable().annotate(accAnnot);
        	}
        	accAnnot.put("loop", "_directive");
        }
        accAnnot.put("gang", "_clause");

        if (pragma.containsKey("parallel") && pragma.containsKey("for")) {
            //accAnnot.put("worker", "_clause");
        	parse_parallel_pragma(pragma);
        } else if(pragma.containsKey("simd")) {
            //accAnnot.put("vector", "_clause");
        	parse_simd_pragma(pragma);
    	}

        if(pragma.containsKey("private")) {
            parseSet((Set<String>)pragma.get("private"), privateSet);
            pragma.remove("private");
        }
        if(pragma.containsKey("linear")) {
            parseSet((Set<String>)pragma.get("linear"), linearSet);
            pragma.remove("linear");
        }
        if(pragma.containsKey("firstprivate")) {
            parseSet((Set<String>)pragma.get("firstprivate"), firstprivateSet);
            pragma.remove("firstprivate");
        }

        if(privateSet.size() > 0)
            accAnnot.put("private", privateSet);
        if( linearSet.size() > 0) {
        	//[FIXME] Semantics of OpenMP linear clause is different from that of OpenACC private clause.
        	//OpenMP Specification Version 5.0, Section 2.19.4.6 linear Clause says that the value corresponding to 
        	//the sequentially last iteration of the associated loop(s) is assigned to the original list item.
            PrintTools.println("[WARNING in OMP4toACCTranslator.parse_distribute_pragma()] the current translator converts "
            		+ "OpenMP linear clause to OpenACC private clause, which may break the original OpenMP semantics." +
                                        AnalysisTools.getEnclosingAnnotationContext(pragma), 0);
        	Set<SubArray> oldPrivateSet = accAnnot.get("private");
        	if( oldPrivateSet == null ) {
        		accAnnot.put("private", linearSet);
        	} else {
        		oldPrivateSet.addAll(linearSet);
        	}
        }
        if(firstprivateSet.size() > 0)
            accAnnot.put("firstprivate", firstprivateSet);
        if(pragma.containsKey("collapse")) {
            accAnnot.put("collapse", ParserTools.strToExpression((String) pragma.get("collapse")));
            pragma.remove("collapse");
        }

        //TODO: dist_schedule clause

        pragma.setSkipPrint(true);
        removeList.add(pragma);
    }

    private void parse_teams_pragma(OmpAnnotation pragma)
    {
        Set<SubArray> privateSet = new HashSet<SubArray>();
        Set<SubArray> linearSet = new HashSet<SubArray>();
        Set<SubArray> firstprivateSet = new HashSet<SubArray>();
        Set<SubArray> copySet = new HashSet<SubArray>();

        Expression numTeamsExpr = null;
        Expression threadLimitExpr = null;
        Expression condition = null;

        String defaultExpr = null;

        //Handle condition passed from enclosing target region.
        if(pragma.containsKey("if"))
        {
        	String condStr = pragma.get("if");
            condition =  ParserTools.strToExpression(condStr);
            pragma.remove("if");
        }

        if(pragma.containsKey("num_teams")) {
            String numTeamsStr = pragma.get("num_teams");
            numTeamsExpr = ParserTools.strToExpression(numTeamsStr);
            pragma.remove("num_teams");
        }

        if(pragma.containsKey("thread_limit")) {
            String threadLimitStr = pragma.get("thread_limit");
            threadLimitExpr = ParserTools.strToExpression(threadLimitStr);
            pragma.remove("thread_limit");
        }

        if(pragma.containsKey("default")) {
            String defaultStr = pragma.get("default");
            if(defaultStr.compareTo("none") == 0)
            {
                defaultExpr = "none";
            }
            pragma.remove("default");
        }

        if(pragma.containsKey("private")) {
            parseSet((Set<String>)pragma.get("private"), privateSet);
            pragma.remove("private");
        }
        if(pragma.containsKey("linear")) {
            parseSet((Set<String>)pragma.get("linear"), linearSet);
            pragma.remove("linear");
        }
        if(pragma.containsKey("firstprivate")) {
            parseSet((Set<String>)pragma.get("firstprivate"), firstprivateSet);
            pragma.remove("firstprivate");
        }
        if(pragma.containsKey("shared")) {
        	//[DEBUG] shared set in a teams construct should be used to identify 
        	//team-private variables; putting these variables into OpenACC copy set
        	//will conflict with OpenACC data clauses created for OpenMP map clauses.
            //parseSet((Set<String>)pragma.get("shared"), copySet);
            //pragma.remove("shared");
        }


        ACCAnnotation accAnnot = null;
        accAnnot = pragma.getAnnotatable().getAnnotation(ACCAnnotation.class, "parallel");
        if( accAnnot == null ) {
        	accAnnot = new ACCAnnotation();
        	accAnnot.put("parallel", "_directive");
        	pragma.getAnnotatable().annotate(accAnnot);
        }
        if(copySet.size() > 0)
            accAnnot.put("copy", copySet);
        if(privateSet.size() > 0)
            accAnnot.put("private", privateSet);
        if( linearSet.size() > 0) {
        	//[FIXME] Semantics of OpenMP linear clause is different from that of OpenACC private clause.
        	//OpenMP Specification Version 5.0, Section 2.19.4.6 linear Clause says that the value corresponding to 
        	//the sequentially last iteration of the associated loop(s) is assigned to the original list item.
            PrintTools.println("[WARNING in OMP4toACCTranslator.parse_teams_pragma()] the current translator converts "
            		+ "OpenMP linear clause to OpenACC private clause, which may break the original OpenMP semantics." +
                                        AnalysisTools.getEnclosingAnnotationContext(pragma), 0);
        	Set<SubArray> oldPrivateSet = accAnnot.get("private");
        	if( oldPrivateSet == null ) {
        		accAnnot.put("private", linearSet);
        	} else {
        		oldPrivateSet.addAll(linearSet);
        	}
        }
        if(firstprivateSet.size() > 0)
            accAnnot.put("firstprivate", firstprivateSet);
        if(numTeamsExpr != null)
            accAnnot.put("num_gangs", numTeamsExpr);
        if(threadLimitExpr != null)
            accAnnot.put("num_workers", threadLimitExpr);
        if(defaultExpr != null)
        	accAnnot.put("default", defaultExpr);
        if(condition != null) {
        	accAnnot.put("if", condition);
        }

        if(pragma.containsKey("reduction")) {
        	Map<String, Set> reduction_map = pragma.get("reduction");
        	parseReduction(accAnnot, reduction_map);
        	pragma.remove("reduction");
        }

        pragma.setSkipPrint(true);
        removeList.add(pragma);
    }

    private void parse_target_pragma(OmpAnnotation pragma, OmpRegion regionType)
    {
        Expression condition = null;
        boolean removePragma = true;
        boolean nowait = false;
        int numDependVariables = 0;
        Annotatable at = pragma.getAnnotatable();

        Set<SubArray> createSet = new HashSet<SubArray>();
        Set<SubArray> copyinSet = new HashSet<SubArray>();
        Set<SubArray> copyoutSet = new HashSet<SubArray>();
        Set<SubArray> copySet = new HashSet<SubArray>();
        Set<SubArray> presentSet = new HashSet<SubArray>();
        Set<SubArray> devicePtrSet = new HashSet<SubArray>();
        Set<SubArray> releaseSet = new HashSet<SubArray>();
        Set<SubArray> deleteSet = new HashSet<SubArray>();
        Set<SubArray> inSet = new HashSet<SubArray>();
        Set<SubArray> outSet = new HashSet<SubArray>();
        Set<SubArray> inoutSet = new HashSet<SubArray>();
        
        if(pragma.containsKey("nowait"))
        {
        	nowait = true;
        }

        //Handle condition
        if(pragma.containsKey("if"))
        {
        	String condStr = pragma.get("if");
            condition =  ParserTools.strToExpression(condStr);
        }

        if(pragma.containsKey("device"))
        {
            //device clause
            String deviceStr = pragma.get("device");
            deviceExpr = ParserTools.strToExpression(deviceStr);
            ACCAnnotation newAnnot = new ACCAnnotation("set", "_directive");
            newAnnot.put("device_num", deviceExpr);
            ((CompoundStatement)at.getParent()).addStatementBefore(
                    (Statement)at,
                    new AnnotationStatement(newAnnot));
        }

        //Handle Data Transfer
        boolean always = false;

        if(pragma.containsKey("alloc")) {
            parseSet((Set<String>)pragma.get("alloc"), createSet);
        }

        if(pragma.containsKey("always alloc")) {
            parseSet((Set<String>)pragma.get("alloc"), createSet);
            always = true;
        }

        if(pragma.containsKey("to")) {
            parseSet((Set<String>)pragma.get("to"), copyinSet);
        }

        if(pragma.containsKey("always to")) {
            parseSet((Set<String>)pragma.get("always to"), copyinSet);
            always = true;
        }

        if(pragma.containsKey("from")) {
            parseSet((Set<String>)pragma.get("from"), copyoutSet);
        }

        if(pragma.containsKey("always from")) {
            parseSet((Set<String>)pragma.get("always from"), copyoutSet);
            always = true;
        }

        if(pragma.containsKey("tofrom")) 
        {
            parseSet((Set<String>)pragma.get("tofrom"), copySet);
        }

        if(pragma.containsKey("always tofrom")) 
        {
            parseSet((Set<String>)pragma.get("always tofrom"), copySet);
            always = true;
        }

        if(pragma.containsKey("use_device_ptr"))
        {
            parseSet((Set<String>)pragma.get("use_device_ptr"), presentSet);
        }

        if(pragma.containsKey("is_device_ptr"))
        {
            parseSet((Set<String>)pragma.get("is_device_ptr"), devicePtrSet);
        }

        if(pragma.containsKey("delete"))
        {
            parseSet((Set<String>)pragma.get("delete"), deleteSet);
        }

        if(pragma.containsKey("always delete"))
        {
            parseSet((Set<String>)pragma.get("always delete"), deleteSet);
            always = true;
        }

        if(pragma.containsKey("release"))
        {
            parseSet((Set<String>)pragma.get("release"), releaseSet);
        }

        if(pragma.containsKey("always release"))
        {
            parseSet((Set<String>)pragma.get("always release"), releaseSet);
            always = true;
        }

        if(pragma.containsKey("in"))
        {
            parseSet((Set<String>)pragma.get("in"), inSet);
            numDependVariables += inSet.size();
        }

        if(pragma.containsKey("out"))
        {
            parseSet((Set<String>)pragma.get("out"), outSet);
            numDependVariables += outSet.size();
        }

        if(pragma.containsKey("inout"))
        {
            parseSet((Set<String>)pragma.get("inout"), inoutSet);
            numDependVariables += inoutSet.size();
        }

        boolean createAccDirective = false;
        if((createSet.size() > 0) || (copyinSet.size() > 0) || (copyoutSet.size() > 0) || 
        		(copySet.size() > 0) || (presentSet.size() > 0) || (devicePtrSet.size() > 0) ||
        		(releaseSet.size() > 0 ) || (deleteSet.size() > 0) )
        {
        	createAccDirective = true;
        }
        ACCAnnotation newAnnot = new ACCAnnotation();
        if( regionType == OmpRegion.Target ) {
        	ARCAnnotation devTaskAnnot = at.getAnnotation(ARCAnnotation.class, "devicetask");
        	String taskMapping = null;
        	String taskScheduling = null;
        	if( devTaskAnnot != null ) {
        		taskMapping = devTaskAnnot.get("map");
        		taskScheduling = devTaskAnnot.get("schedule");
        	}
        	//System.out.println("devicetask: " + devTaskAnnot + "\n");
        	//System.out.println("structured block: " + at + "\n");
        	if( taskMapping == null ) {
        		//[TODO] implement an analysis pass to check whether the target region is eligible for task parallelism. 
        		taskMapping = "included";
        	}
        	if( taskMapping.equals("coarse_grained") ) {
        		//removePragma = false;
        		Set<String> searchKeys = new HashSet<String>();
        		searchKeys.add("task");
        		searchKeys.add("taskwait");
        		List<OmpAnnotation> taskAnnotList = AnalysisTools.collectPragmas(at, OmpAnnotation.class, searchKeys, false);
        		if( taskAnnotList != null ) {
        			Set<SubArray> taskInSet = new HashSet<SubArray>();
        			Set<SubArray> taskOutSet = new HashSet<SubArray>();
        			Set<SubArray> taskInoutSet = new HashSet<SubArray>();
        			List<Statement> insertRefList = new LinkedList<Statement>();
        			List<Statement> insertTargetList = new LinkedList<Statement>();
        			for( OmpAnnotation tAnnot : taskAnnotList ) {
        				Annotatable att = tAnnot.getAnnotatable();
        				OmpAnnotation taskAnnot = att.getAnnotation(OmpAnnotation.class, "task");
        				OmpAnnotation taskwaitAnnot = att.getAnnotation(OmpAnnotation.class, "taskwait");
        				int taskNumDependVariables = 0;
        				if( taskAnnot != null ) {
        					ACCAnnotation parallelAnnot = new ACCAnnotation();
        					parallelAnnot.put("parallel", "_directive");
        					if( condition != null ) {
        						parallelAnnot.put("if", condition.clone());
        					}
        					att.annotate(parallelAnnot);
        					taskInSet.clear();
        					taskOutSet.clear();
        					taskInoutSet.clear();
        					if(taskAnnot.containsKey("in"))
        					{
        						parseSet((Set<String>)taskAnnot.get("in"), taskInSet);
        						taskNumDependVariables += taskInSet.size();
        					}

        					if(taskAnnot.containsKey("out"))
        					{
        						parseSet((Set<String>)taskAnnot.get("out"), taskOutSet);
        						taskNumDependVariables += taskOutSet.size();
        					}

        					if(taskAnnot.containsKey("inout"))
        					{
        						parseSet((Set<String>)taskAnnot.get("inout"), taskInoutSet);
        						taskNumDependVariables += taskInoutSet.size();
        					}
        					CompoundStatement taskCStmt = handleDependClause(taskAnnot, taskNumDependVariables, taskInSet, taskOutSet, taskInoutSet, taskType.innerTask);
        					parallelAnnot.put("async", new NameID("openarc_async"));
        					List<Expression> waitList = new ArrayList<Expression>(defaultNumAsyncQueues);
        					for(int i=0; i<defaultNumAsyncQueues; i++) {
        						waitList.add(new ArrayAccess(new NameID("openarc_waits"), new IntegerLiteral(i)));
        					}
        					parallelAnnot.put("wait", waitList);
        					insertRefList.add((Statement)att);
        					insertTargetList.add(taskCStmt);
        					DFIterator<Annotatable> titer = new DFIterator<Annotatable>(att, Annotatable.class);
        					while (titer.hasNext()) 
        					{
        						Annotatable tat = titer.next();
        						List<OmpAnnotation> pragmas = tat.getAnnotations(OmpAnnotation.class);
        						if( pragmas != null ) {
        							for (int i = 0; i < pragmas.size(); i++) 
        							{
        								OmpAnnotation tpragma = pragmas.get(i);
        								if (tpragma.containsKey("parallel")) {
        									//Enter Parallel region
        									PrintTools.println("Enter parallel region", 2);
        									parse_parallel_pragma(tpragma);
        								} else if (tpragma.containsKey("for")) {
        									//Enter for region
        									PrintTools.println("Enter for region", 2);
        									parse_for_pragma(tpragma);
        								} else if (tpragma.containsKey("simd")) {
        									//Enter SIMD region
        									PrintTools.println("Enter simd region", 2);
        									parse_simd_pragma(tpragma);
        								}
        								if (tpragma.containsKey("atomic")) {
        									parse_atomic(tpragma);
        								}

        							}
        						}
        					}
        					taskAnnot.setSkipPrint(true);
        					removeList.add(taskAnnot);
        				} else if( taskwaitAnnot != null ) {
        					if( !att.containsAnnotation(ACCAnnotation.class, "wait") ) {
        						ACCAnnotation newWaitAnnot = new ACCAnnotation();
        						newWaitAnnot.put("wait", "_directive");
        						att.annotate(newWaitAnnot);
        						//taskwaitAnnot.remove("taskwait");
        						//taskwaitAnnot.setSkipPrint(true);
        						//removeList.add(taskwaitAnnot);
        					}
        				}
        			}
        			if( !insertRefList.isEmpty() ) {
        				for( int k=0; k<insertRefList.size(); k++ ) {
        					Statement refStmt = insertRefList.get(k);
        					Statement taskCStmt = insertTargetList.get(k);
        					((CompoundStatement)refStmt.getParent()).addStatementBefore(
        							refStmt, taskCStmt);
        				}
        			}
        		}
        		if( createAccDirective ) {
        			newAnnot.put("data", "_directive");
        			at.annotate(newAnnot);
        		}
        		List<OmpAnnotation> delList = IRTools.collectPragmas(at, OmpAnnotation.class, null);
        		if( delList != null ) {
        			removeList.addAll(delList);
        		}
        	} else if( taskMapping.equals("fine_grained") ) {
        		//removePragma = false;
        		newAnnot.put("parallel", "_directive");
/*        		newAnnot.put("kernels", "_directive");
        		if( at instanceof Loop ) {
        			newAnnot.put("seq", "_clause");
        		}*/
        		at.annotate(newAnnot);
        		List<OmpAnnotation> taskAnnotList = IRTools.collectPragmas(at, OmpAnnotation.class, "task");
        		if( taskAnnotList != null ) {
        			for( OmpAnnotation tAnnot : taskAnnotList ) {
        				Annotatable att = tAnnot.getAnnotatable();
        				DFIterator<Annotatable> titer = new DFIterator<Annotatable>(att, Annotatable.class);
        				while (titer.hasNext()) 
        				{
        					Annotatable tat = titer.next();
        					List<OmpAnnotation> pragmas = tat.getAnnotations(OmpAnnotation.class);
        					if( pragmas != null ) {
        						for (int i = 0; i < pragmas.size(); i++) 
        						{
        							OmpAnnotation tpragma = pragmas.get(i);
        							if (tpragma.containsKey("parallel")) {
        								//Enter Parallel region
        								PrintTools.println("Enter parallel region", 2);
        								parse_parallel_pragma(tpragma);
        							} else if (tpragma.containsKey("for")) {
        								//Enter for region
        								PrintTools.println("Enter for region", 2);
        								parse_for_pragma(tpragma);
        							} else if (tpragma.containsKey("simd")) {
        								//Enter SIMD region
        								PrintTools.println("Enter simd region", 2);
        								parse_simd_pragma(tpragma);
        							}
        							if (tpragma.containsKey("atomic")) {
        								parse_atomic(tpragma);
        							}

        						}
        					}
        				}
        			}
        		}
 /*       		List<OmpAnnotation> delList = IRTools.collectPragmas(at, OmpAnnotation.class, null);
        		if( delList != null ) {
        			removeList.addAll(delList);
        		}*/
        	} else if( taskMapping.equals("included") ) {
        		createAccDirective = true;
        		boolean createAccKernel = true;
        		if( !at.containsAnnotation(OmpAnnotation.class, "teams") ) {
        			List<Traversable> children = null;
        			if( at instanceof CompoundStatement ) {
        				children = at.getChildren();
        				if( children != null ) {
        					Annotatable firstChild = (Annotatable)children.get(0);
        					if( firstChild.containsAnnotation(OmpAnnotation.class, "teams") ) {
        						createAccKernel = false;
        					}
        				}
        			}
        		}
        		if( createAccKernel ) {
        			newAnnot.put("parallel", "_directive");
        		} else {
        			newAnnot.put("data", "_directive");
        		}
        		at.annotate(newAnnot);
        	} else if ( taskMapping != null ) {
        		Tools.exit("[ERROR in OMP4toACCTranslator.start()] Unexpected argument (" + taskMapping + ") for the map clause in the following OpenARC devicetask construct; exit!\n"
        				+ "OpenARC construct: " + at + "\n" + AnalysisTools.getEnclosingContext(at));

        	}
        } else if( regionType == OmpRegion.Target_Data ) {
        	newAnnot.put("data", "_directive");
        	pragma.getAnnotatable().annotate(newAnnot);
        } else if( regionType == OmpRegion.Target_Enter_Data ) {
        	newAnnot.put("enter", "_directive");
        	newAnnot.put("data", "_directive");
        	((CompoundStatement)pragma.getAnnotatable().getParent()).addStatementBefore(
        			(Statement)pragma.getAnnotatable(),
        			new AnnotationStatement(newAnnot));
        } else if( regionType == OmpRegion.Target_Exit_Data ) {
        	newAnnot.put("exit", "_directive");
        	newAnnot.put("data", "_directive");
        	((CompoundStatement)pragma.getAnnotatable().getParent()).addStatementBefore(
        			(Statement)pragma.getAnnotatable(),
        			new AnnotationStatement(newAnnot));
        } else if( regionType == OmpRegion.Target_Update ) {
        	newAnnot.put("update", "_directive");
        	((CompoundStatement)pragma.getAnnotatable().getParent()).addStatementBefore(
        			(Statement)pragma.getAnnotatable(),
        			new AnnotationStatement(newAnnot));
        }
        if(createSet.size() > 0) {
        	newAnnot.put("pcreate", createSet);
        }
        if(copyinSet.size() > 0) {
        	if( regionType == OmpRegion.Target_Update ) {
        		newAnnot.put("device", copyinSet);
        	} else if( regionType == OmpRegion.Target_Exit_Data ) {
        		newAnnot.put("delete", copyinSet);
        	} else {
        		if( always ) {
        			newAnnot.put("copyin", copyinSet);
        		} else {
        			newAnnot.put("pcopyin", copyinSet);
        		}
        	}
        }
        if(copyoutSet.size() > 0) {
        	if( regionType == OmpRegion.Target_Update ) {
        		newAnnot.put("host", copyoutSet);
        	} else if( regionType == OmpRegion.Target_Enter_Data ) {
        		if( always ) {
        			newAnnot.put("pcreate", copyinSet);
        		} else {
        			newAnnot.put("pcreate", copyinSet);
        		}
        	} else {
        		if( always ) {
        			newAnnot.put("copyout", copyoutSet);
        		} else {
        			newAnnot.put("pcopyout", copyoutSet);
        		}
        	}
        }
        if(copySet.size() > 0) {
        	if( regionType == OmpRegion.Target_Enter_Data ) {
        		if( always ) {
        			newAnnot.put("copyin", copySet);
        		} else {
        			newAnnot.put("pcopyin", copySet);
        		}
        	} else if( regionType == OmpRegion.Target_Exit_Data ) {
        		if( always ) {
        			newAnnot.put("copyout", copySet);
        		} else {
        			newAnnot.put("pcopyout", copySet);
        		}
        	} else {
        		if( always ) {
        			newAnnot.put("copy", copySet);
        		} else {
        			newAnnot.put("pcopy", copySet);
        		}
        	}
        }
        if(presentSet.size() > 0) {
        	newAnnot.put("present", presentSet);
        }
        if(devicePtrSet.size() > 0) {
        	newAnnot.put("deviceptr", devicePtrSet);
        }
        if(deleteSet.size() > 0) {
        	newAnnot.put("delete", deleteSet);
        	newAnnot.put("finalize", "_clause");
        }
        if(releaseSet.size() > 0) {
        	newAnnot.put("delete", releaseSet);
        }
        if(condition != null) {
        	newAnnot.put("if", condition);
        }

        if( nowait ) {
        	if( createAccDirective ) {
        		CompoundStatement cStmt = handleDependClause(pragma, numDependVariables, inSet, outSet, inoutSet, taskType.innerTask);
        		((CompoundStatement)at.getParent()).addStatementBefore(
        				(Statement)at, cStmt);
        		newAnnot.put("async", new NameID("openarc_async"));
        		List<Expression> waitList = new ArrayList<Expression>(defaultNumAsyncQueues);
        		for(int i=0; i<defaultNumAsyncQueues; i++) {
        			waitList.add(new ArrayAccess(new NameID("openarc_waits"), new IntegerLiteral(i)));
        		}
        		newAnnot.put("wait", waitList);
        	} else { //[DEBUG] When this pass will be executed?
        		CompoundStatement cStmt = handleDependClause(pragma, numDependVariables, inSet, outSet, inoutSet, taskType.outerTask);
        		((CompoundStatement)at.getParent()).addStatementBefore(
        				(Statement)at, cStmt);
        		FunctionCall newFCall = null;
        		Symbol newFCallSym = SymbolTools.getSymbolOfName("omp_helper_task_exit", at);
        		if( newFCallSym == null ) {
        			newFCall = new FunctionCall(new NameID("omp_helper_task_exit"));
        		} else {
        			newFCall = new FunctionCall(new Identifier(newFCallSym));
        		}
        		Statement exitStmt = new ExpressionStatement(newFCall);
        		cStmt = new CompoundStatement();
        		cStmt.addStatement(exitStmt);
        		((CompoundStatement)at.getParent()).addStatementAfter(
        				(Statement)at, cStmt);
        	}
        }

        if( removePragma ) {
        	pragma.setSkipPrint(true);
        	removeList.add(pragma);
        }
    }

    private void parse_declare_target(OmpAnnotation pragma)
    {
    	Annotatable at = pragma.getAnnotatable();
    	boolean noClause = true;
        Set<SubArray> copyinSet = new HashSet<SubArray>();
        Set<SubArray> linkSet = new HashSet<SubArray>();
        List<Declaration> deviceFuncList = new LinkedList<Declaration>();
        List<VariableDeclaration> deviceVarList = new LinkedList<VariableDeclaration>();

        if(pragma.containsKey("to")) {
        	noClause = false;
        	if( procDeclList == null ) {
        		procDeclList = new LinkedList<Declaration>();
        		List<ProcedureDeclarator> procDeclrList = AnalysisTools.getProcedureDeclarators(program);
        		for(ProcedureDeclarator pDeclr : procDeclrList ) {
        			procDeclList.add(pDeclr.getDeclaration());
        		}
        		procDeclList.addAll(IRTools.getProcedureList(program));
        	}
        	Set<String> extListSet = (Set<String>)pragma.get("to");
        	for(Declaration tDecl : procDeclList ) {
        		String procName = null;
        		if( tDecl instanceof Procedure ) {
        			procName = ((Procedure)tDecl).getSymbolName();
        		} else if( tDecl instanceof VariableDeclaration ) {
        			procName = ((VariableDeclaration)tDecl).getDeclarator(0).getID().toString();
        		}
        		if( extListSet.contains(procName) ) {
        			extListSet.remove(procName);
        			deviceFuncList.add(tDecl);
        		}
        	}
        	if( !extListSet.isEmpty() ) {
        		parseSet(extListSet, copyinSet);
        	}
        }
        if(pragma.containsKey("link")) {
        	noClause = false;
            parseSet((Set<String>)pragma.get("link"), linkSet);
        }
        
        if( noClause ) {
        	//#pragma omp declare target clause; find matching end declare target directive.
        	Traversable tt = at.getParent();
        	if( tt instanceof CompoundStatement ) {
        		CompoundStatement cStmt = (CompoundStatement)tt;
        		List<Traversable> childList = cStmt.getChildren();
        		boolean declareTargetStart = false;
        		for( Traversable child : childList ) {
        			if( child == at ) {
        				declareTargetStart = true;
        			} else {
        				if( declareTargetStart ) {
        					if( child instanceof AnnotationStatement ) {
        						OmpAnnotation ompAnnot = ((Annotatable)child).getAnnotation(OmpAnnotation.class, "end");
        						if( ompAnnot != null ) {
        							declareTargetStart = false;
        							ompAnnot.setSkipPrint(true);
        							removeList.add(ompAnnot);
        							break;
        						}
        					} else if( child instanceof DeclarationStatement ) {
        						Declaration vDecl = ((DeclarationStatement)child).getDeclaration();
        						if( vDecl instanceof VariableDeclaration ) {
        							if( !((VariableDeclaration)vDecl).getSpecifiers().contains(Specifier.EXTERN) ) {
        								deviceVarList.add((VariableDeclaration)vDecl);
        							}
        						}
        					}
        				}
        			}
        		}
        	} else if( tt instanceof TranslationUnit ) {
        		TranslationUnit tUnt = (TranslationUnit)tt;
        		List<Traversable> childList = tUnt.getChildren();
        		boolean declareTargetStart = false;
        		for( Traversable child : childList ) {
        			if( child == at ) {
        				declareTargetStart = true;
        			} else {
        				if( declareTargetStart ) {
        					if( child instanceof AnnotationDeclaration ) {
        						OmpAnnotation ompAnnot = ((Annotatable)child).getAnnotation(OmpAnnotation.class, "end");
        						if( ompAnnot != null ) {
        							declareTargetStart = false;
        							ompAnnot.setSkipPrint(true);
        							removeList.add(ompAnnot);
        							break;
        						}
        					} else if( child instanceof VariableDeclaration ) {
        						Declarator tdeclr = ((VariableDeclaration)child).getDeclarator(0);
        						if( tdeclr instanceof ProcedureDeclarator ) {
        							deviceFuncList.add((VariableDeclaration)child);
        						} else {
        							if( !((VariableDeclaration)child).getSpecifiers().contains(Specifier.EXTERN) ) {
        								deviceVarList.add((VariableDeclaration)child);
        							}
        						}
        					} else if( child instanceof Procedure ) {
        						deviceFuncList.add((Procedure)child);
        					}
        				}
        			}
        		}
        	}
        }
        
        if( !deviceFuncList.isEmpty() ) {
        	for( Declaration tDecl : deviceFuncList ) {
        		if( !tDecl.containsAnnotation(ACCAnnotation.class, "routine") ) {
        			ACCAnnotation annot = new ACCAnnotation("routine", "_directive");
        			tDecl.annotate(annot);
        		}
        	}
        }
        if( !deviceVarList.isEmpty() ) {
        	for( VariableDeclaration tDecl : deviceVarList ) {
        		List<IDExpression> IDList = tDecl.getDeclaredIDs();
        		Traversable tAt = null;
        		if( tDecl.getParent() instanceof DeclarationStatement ) {
        			tAt = tDecl.getParent();
        		} else {
        			tAt = tDecl;
        		}
        		ACCAnnotation annot = new ACCAnnotation("declare", "_directive");
        		if( tAt == tDecl ) {
        			TranslationUnit tUnt = (TranslationUnit)tAt.getParent();
        			tUnt.addDeclarationAfter(tDecl, new AnnotationDeclaration(annot));
        		} else {
        			CompoundStatement cStmt = (CompoundStatement)tAt.getParent();
        			cStmt.addStatementAfter((Statement)tAt, new AnnotationStatement(annot));
        		}

        		Set<SubArray> copyinSASet = null;
        		if( annot.containsKey("copyin") ) {
        			copyinSASet = annot.get("copyin");
        		} else {
        			copyinSASet = new HashSet<SubArray>();
        			annot.put("copyin", copyinSASet);
        		}
        		for( IDExpression ID : IDList ) {
        			boolean IDExist = false;
        			for( SubArray subA : copyinSASet ) {
        				if( subA.getArrayName().equals(ID) ) {
        					IDExist = true;
        					break;
        				}
        			}
        			if( !IDExist ) {
        				SubArray subA = new SubArray(ID.clone());
        				copyinSASet.add(subA);
        			}
        		}
        	}
        }
        if( !copyinSet.isEmpty() ) {
        	ACCAnnotation annot = new ACCAnnotation("declare", "_directive");
        	annot.put("copyin", copyinSet);
        	if( at.getParent() instanceof TranslationUnit ) {
        		TranslationUnit tUnt = (TranslationUnit)at.getParent();
        		tUnt.addDeclarationAfter((Declaration)at, new AnnotationDeclaration(annot));
        	} else if( at instanceof AnnotationStatement ) {
        		CompoundStatement cStmt = (CompoundStatement)at.getParent();
        		cStmt.addStatementAfter((Statement)at, new AnnotationStatement(annot));
        	}
        }
        if( !linkSet.isEmpty() ) {
        	ACCAnnotation annot = new ACCAnnotation("declare", "_directive");
        	annot.put("link", linkSet);
        	if( at.getParent() instanceof TranslationUnit ) {
        		TranslationUnit tUnt = (TranslationUnit)at.getParent();
        		tUnt.addDeclarationAfter((Declaration)at, new AnnotationDeclaration(annot));
        	} else if( at instanceof AnnotationStatement ) {
        		CompoundStatement cStmt = (CompoundStatement)at.getParent();
        		cStmt.addStatementAfter((Statement)at, new AnnotationStatement(annot));
        	}
        }
        
        pragma.setSkipPrint(true);
        removeList.add(pragma);
    }

    private void parseSet(Set<String> orig, Set dest)
    {
        dest.addAll(ACCParser.parse_subarraylist(PrintTools.collectionToString(orig, ","), macroMap));
    }
    
    private void parseReduction(ACCAnnotation accAnnot, Map<String, Set> reduction_map)
    {
    	StringBuilder sb = new StringBuilder(80);
    	for (String op : reduction_map.keySet()) {
    		sb.append(op);
    		sb.append(": ");
    		sb.append(PrintTools.collectionToString(
    				reduction_map.get(op), ", "));
    	}
        ACCParser.parse_reductionclause(accAnnot, sb.toString(), macroMap);
    }
    
    private CompoundStatement handleDependClause ( OmpAnnotation pragma, int numDependVariables, Set<SubArray> inSet,
    		Set<SubArray> outSet, Set<SubArray> inoutSet, taskType tType ) {
    	Annotatable at = pragma.getAnnotatable();
    	//This target task can be executed asynchronously, and thus add appropriate async and wait clauses 
    	//to the generated OpenACC directive.
    	CompoundStatement cStmt = new CompoundStatement();
    	ArraySpecifier aspec = new ArraySpecifier(new IntegerLiteral(numDependVariables));
    	VariableDeclarator dependtype_declarator = new VariableDeclarator(new NameID("openarc_dependtypes"), aspec);
    	List<Specifier> specs = new LinkedList<Specifier>();
    	specs.add(Specifier.INT);
    	Declaration dependtype_decl = new VariableDeclaration(specs, dependtype_declarator);
    	List<Expression> dependtype_initlist = new ArrayList<Expression>(numDependVariables);

    	aspec = new ArraySpecifier(new IntegerLiteral(numDependVariables*2));
    	List<ArraySpecifier> aspecs = new ArrayList<ArraySpecifier>(1);
    	aspecs.add(aspec);
    	specs = new LinkedList<Specifier>();
    	specs.add(PointerSpecifier.UNQUALIFIED);
    	VariableDeclarator dependadd_declarator = new VariableDeclarator(specs, new NameID("openarc_dependaddresses"), aspecs);
    	specs = new LinkedList<Specifier>();
    	specs.add(Specifier.VOID);
    	Declaration dependadd_decl = new VariableDeclaration(specs, dependadd_declarator);
    	List<Expression> dependadd_initlist = new ArrayList<Expression>(numDependVariables*2);

    	for( SubArray subA : inSet ) {
    		ACCAnalysis.updateSymbolsInSubArray(subA, at, null, pragma);
    		Expression subAName = subA.getArrayName();
    		List<Expression> startList = new LinkedList<Expression>();
    		List<Expression> lengthList = new LinkedList<Expression>();
    		List<Expression> endList = new LinkedList<Expression>();
    		boolean foundDimensions = AnalysisTools.extractDimensionInfo(subA, startList, lengthList, false, at);
    		if( !foundDimensions ) {
    			Tools.exit("[ERROR in OMP4toACCTranslator.handleDependClause()] Dimension information " +
    					"of the following variable is unknown; exit.\n" + 
    					"Variable: " + subAName + "\n"+ AnalysisTools.getEnclosingAnnotationContext(pragma));
    		} else {
    			dependtype_initlist.add(new NameID("oh_in"));
    			if( subA.getArrayDimension() == 0 ) {
    				dependadd_initlist.add(new UnaryExpression(UnaryOperator.ADDRESS_OF, subAName.clone()));
    				dependadd_initlist.add(new UnaryExpression(UnaryOperator.ADDRESS_OF, subAName.clone()));
    			} else {
    				for( Expression texp : lengthList ) {
    					endList.add(new BinaryExpression(texp.clone(), BinaryOperator.SUBTRACT, new IntegerLiteral(1)));
    				}
    				dependadd_initlist.add(new UnaryExpression(UnaryOperator.ADDRESS_OF, new ArrayAccess(subAName.clone(), startList)));
    				dependadd_initlist.add(new UnaryExpression(UnaryOperator.ADDRESS_OF, new ArrayAccess(subAName.clone(), endList)));
    			}
    		}
    	}
    	for( SubArray subA : outSet ) {
    		ACCAnalysis.updateSymbolsInSubArray(subA, at, null, pragma);
    		Expression subAName = subA.getArrayName();
    		List<Expression> startList = new LinkedList<Expression>();
    		List<Expression> lengthList = new LinkedList<Expression>();
    		List<Expression> endList = new LinkedList<Expression>();
    		boolean foundDimensions = AnalysisTools.extractDimensionInfo(subA, startList, lengthList, false, at);
    		if( !foundDimensions ) {
    			Tools.exit("[ERROR in OMP4toACCTranslator.handleDependClause()] Dimension information " +
    					"of the following variable is unknown; exit.\n" + 
    					"Variable: " + subAName + "\n"+ AnalysisTools.getEnclosingAnnotationContext(pragma));
    		} else {
    			dependtype_initlist.add(new NameID("oh_out"));
    			if( subA.getArrayDimension() == 0 ) {
    				dependadd_initlist.add(new UnaryExpression(UnaryOperator.ADDRESS_OF, subAName.clone()));
    				dependadd_initlist.add(new UnaryExpression(UnaryOperator.ADDRESS_OF, subAName.clone()));
    			} else {
    				for( Expression texp : lengthList ) {
    					endList.add(new BinaryExpression(texp.clone(), BinaryOperator.SUBTRACT, new IntegerLiteral(1)));
    				}
    				dependadd_initlist.add(new UnaryExpression(UnaryOperator.ADDRESS_OF, new ArrayAccess(subAName.clone(), startList)));
    				dependadd_initlist.add(new UnaryExpression(UnaryOperator.ADDRESS_OF, new ArrayAccess(subAName.clone(), endList)));
    			}
    		}
    	}
    	for( SubArray subA : inoutSet ) {
    		ACCAnalysis.updateSymbolsInSubArray(subA, at, null, pragma);
    		Expression subAName = subA.getArrayName();
    		List<Expression> startList = new LinkedList<Expression>();
    		List<Expression> lengthList = new LinkedList<Expression>();
    		List<Expression> endList = new LinkedList<Expression>();
    		boolean foundDimensions = AnalysisTools.extractDimensionInfo(subA, startList, lengthList, false, at);
    		if( !foundDimensions ) {
    			Tools.exit("[ERROR in OMP4toACCTranslator.handleDependClause()] Dimension information " +
    					"of the following variable is unknown; exit.\n" + 
    					"Variable: " + subAName + "\n"+ AnalysisTools.getEnclosingAnnotationContext(pragma));
    		} else {
    			dependtype_initlist.add(new NameID("oh_inout"));
    			if( subA.getArrayDimension() == 0 ) {
    				dependadd_initlist.add(new UnaryExpression(UnaryOperator.ADDRESS_OF, subAName.clone()));
    				dependadd_initlist.add(new UnaryExpression(UnaryOperator.ADDRESS_OF, subAName.clone()));
    			} else {
    				for( Expression texp : lengthList ) {
    					endList.add(new BinaryExpression(texp.clone(), BinaryOperator.SUBTRACT, new IntegerLiteral(1)));
    				}
    				dependadd_initlist.add(new UnaryExpression(UnaryOperator.ADDRESS_OF, new ArrayAccess(subAName.clone(), startList)));
    				dependadd_initlist.add(new UnaryExpression(UnaryOperator.ADDRESS_OF, new ArrayAccess(subAName.clone(), endList)));
    			}
    		}
    	}

    	dependtype_declarator.setInitializer(new Initializer(dependtype_initlist));
    	dependadd_declarator.setInitializer(new Initializer(dependadd_initlist));

    	cStmt.addDeclaration(dependtype_decl);
    	cStmt.addDeclaration(dependadd_decl);

    	FunctionCall fCall = null;
    	if( tType == taskType.innerTask ) {
    		Symbol newFCallSym = SymbolTools.getSymbolOfName("omp_helper_task_exec", at);
    		if( newFCallSym == null ) {
    			fCall = new FunctionCall(new NameID("omp_helper_task_exec"));
    		} else {
    			fCall = new FunctionCall(new Identifier(newFCallSym));
    		}
    	} else {
    		Symbol newFCallSym = SymbolTools.getSymbolOfName("omp_helper_task_enter", at);
    		if( newFCallSym == null ) {
    			fCall = new FunctionCall(new NameID("omp_helper_task_enter"));
    		} else {
    			fCall = new FunctionCall(new Identifier(newFCallSym));
    		}
    	}
    	fCall.addArgument(new IntegerLiteral(numDependVariables));
    	if( numDependVariables == 0 ) {
    		fCall.addArgument(new IntegerLiteral(0));
    		fCall.addArgument(new IntegerLiteral(0));
    	} else {
    		fCall.addArgument(new Identifier(dependtype_declarator));
    		fCall.addArgument(new Identifier(dependadd_declarator));
    	}
    	if( tType == taskType.innerTask ) {
    		fCall.addArgument(new UnaryExpression(UnaryOperator.ADDRESS_OF, new NameID("openarc_async")));
    		fCall.addArgument(new NameID("openarc_waits"));
    	}
    	cStmt.addStatement(new ExpressionStatement(fCall));
    	return cStmt;
    }

    /**
     * [Convert critical sections into reduction sections]
     * For each critical section in a parallel region,
     *     if the critical section is a kind of reduction form, necessary reduction
     *     clause is added to the annotation of the enclosing parallel region, and
     *     the original critical construct is commented out.
     * A critical section is considered as a reduction form if reduction variables recognized
     * by Reduction.analyzeStatement2() are the only shared variables modified in the
     * critical section.
     * [CAUTION] Cetus compiler can recognize array reduction, but the array reduction
     * is not supported by standard OpenMP compilers. Therefore, below conversion may
     * not be handled correctly by other OpenMP compilers.
     * [FIXME] Reduction.analyzeStatement2() returns a set of reduction variables as expressions,
     * but this method converts them into a set of symbols. This conversion loses some information
     * and thus complex reduction expressions such as a[0][i] and a[i].b can not be handled properly;
     * current translator supports only simple scalar or array variables.
     */
    public void convertCritical2Reduction()
    {
        List<OmpAnnotation> ompPAnnots = IRTools.collectPragmas(program, OmpAnnotation.class, "parallel");
		//PrintTools.println("omp parallel annotations: "+ompPAnnots.toString(), 0);
        Reduction redAnalysis = new Reduction(program);
        for (OmpAnnotation omp_annot : ompPAnnots)
        {
            Statement pstmt = (Statement)omp_annot.getAnnotatable();
            HashSet<Symbol> shared_set = new HashSet<Symbol>();
            HashSet<String> symStrSet = (HashSet<String>)omp_annot.get("shared");
            if( symStrSet != null ) {
            	for(String symStr : symStrSet ) {
            		Symbol tSym = SymbolTools.getSymbolOfName(symStr, pstmt);
            		if( tSym != null ) {
            			shared_set.add(tSym);
            		}
            	}
            }
            HashMap pRedMap = (HashMap)omp_annot.get("reduction");
            List<OmpAnnotation> ompCAnnots = IRTools.collectPragmas(pstmt, OmpAnnotation.class, "critical");
            //PrintTools.println("omp critical anntations: "+ompCAnnots.toString(), 0);
            for (OmpAnnotation cannot : ompCAnnots)
            {
                boolean foundError = false;
                Statement cstmt = (Statement)cannot.getAnnotatable();
                Set<Symbol> definedSymbols = DataFlowTools.getDefSymbol(cstmt);
                HashSet<Symbol> shared_subset = new HashSet<Symbol>();
                shared_subset.addAll(shared_set);
                Map<String, Set<Expression>> reduce_map = redAnalysis.analyzeStatement2(cstmt);
                //PrintTools.println("reduction map: "+reduce_map.toString(), 0);
                Map<String, Set<Symbol>> reduce_map2 = new HashMap<String, Set<Symbol>>();
                if (!reduce_map.isEmpty())
                {
                    // Remove reduction variables from shared_subset.
                    for (String ikey : (Set<String>)(reduce_map.keySet())) {
                        if( foundError ) {
                            break;
                        }
                        Set<Expression> tmp_set = (Set<Expression>)reduce_map.get(ikey);
                        HashSet<Symbol> redSet = new HashSet<Symbol>();
                        for (Expression iexp : tmp_set) {
                            //Symbol redSym = findsSymbol(shared_set, iexp.toString());
                            Symbol redSym = SymbolTools.getSymbolOf(iexp);
                            if( redSym != null ) {
                                if( redSym instanceof VariableDeclarator ) {
                                    shared_subset.remove(redSym);
                                    redSet.add(redSym);
                                } else {
                                    PrintTools.println("[INFO in convertCritical2Reduction()] the following expression has reduction pattern" +
                                            " but not handled by current translator: " + iexp, 0);
                                    //Skip current critical section.
                                    foundError = true;
                                    break;

                                }
                            } else {
                                PrintTools.println("[WARNING in convertCritical2Reduction()] found unrecognizable reduction expression (" +
                                        iexp+")", 0);
                                //Skip current critical section.
                                foundError = true;
                                break;
                            }
                        }
                        reduce_map2.put(ikey, redSet);
                    }
                    //If error is found, skip current critical section.
                    if( foundError ) {
                        continue;
                    }
                    //////////////////////////////////////////////////////////////////////
                    // If shared_subset and definedSymbols are disjoint,                //
                    // it means that reduction variables are the only shared variables  //
                    // defined in the critical section.                                 //
                    //////////////////////////////////////////////////////////////////////
                    //PrintTools.println("shared_subset: " + PrintTools.collectionToString(shared_subset, ", "), 0);
                    //PrintTools.println("definedSymbols: " + PrintTools.collectionToString(definedSymbols, ", "), 0);
                    //PrintTools.println("new reduction map: " + reduce_map2.toString(), 0);
                    if( Collections.disjoint(shared_subset, definedSymbols) ) {
                        if( pRedMap == null ) {
                            pRedMap = new HashMap();
                            omp_annot.put("reduction", pRedMap);
                        }
                        for (String ikey : (Set<String>)(reduce_map2.keySet())) {
                            Set<Symbol> tmp_set = (Set<Symbol>)reduce_map2.get(ikey);
                            HashSet<Symbol> redSet = (HashSet<Symbol>)pRedMap.get(ikey);
                            if( redSet == null ) {
                                redSet = new HashSet<Symbol>();
                                pRedMap.put(ikey, redSet);
                            }
                            redSet.addAll(tmp_set);
                        }
                        // Remove omp critical annotation and add comment annotation.
                        CommentAnnotation comment = new CommentAnnotation(cannot.toString());
                        AnnotationStatement comment_stmt = new AnnotationStatement(comment);
                        CompoundStatement parent = (CompoundStatement)cstmt.getParent();
                        parent.addStatementBefore(cstmt, comment_stmt);
                        cannot.getAnnotatable().removeAnnotations(OmpAnnotation.class);
                    }
                }
            }
        }
    }

	private void parse_target_pragma_old(OmpAnnotation pragma, OmpRegion regionType)
	{
	    Expression condition = null;
	    boolean removePragma = true;
	    boolean nowait = false;
	    int numDependVariables = 0;
	    Annotatable at = pragma.getAnnotatable();
	
	    Set<SubArray> createSet = new HashSet<SubArray>();
	    Set<SubArray> copyinSet = new HashSet<SubArray>();
	    Set<SubArray> copyoutSet = new HashSet<SubArray>();
	    Set<SubArray> copySet = new HashSet<SubArray>();
	    Set<SubArray> presentSet = new HashSet<SubArray>();
	    Set<SubArray> devicePtrSet = new HashSet<SubArray>();
	    Set<SubArray> releaseSet = new HashSet<SubArray>();
	    Set<SubArray> deleteSet = new HashSet<SubArray>();
	    Set<SubArray> inSet = new HashSet<SubArray>();
	    Set<SubArray> outSet = new HashSet<SubArray>();
	    Set<SubArray> inoutSet = new HashSet<SubArray>();
	    
	    if(pragma.containsKey("nowait"))
	    {
	    	nowait = true;
	    }
	
	    //Handle condition
	    if(pragma.containsKey("if"))
	    {
	    	String condStr = pragma.get("if");
	        condition =  ParserTools.strToExpression(condStr);
	    }
	
	    if(pragma.containsKey("device"))
	    {
	        //device clause
	        String deviceStr = pragma.get("device");
	        deviceExpr = ParserTools.strToExpression(deviceStr);
	        ACCAnnotation newAnnot = new ACCAnnotation("set", "_directive");
	        newAnnot.put("device_num", deviceExpr);
	        ((CompoundStatement)at.getParent()).addStatementBefore(
	                (Statement)at,
	                new AnnotationStatement(newAnnot));
	    }
	
	    //Handle Data Transfer
	    boolean always = false;
	
	    if(pragma.containsKey("alloc")) {
	        parseSet((Set<String>)pragma.get("alloc"), createSet);
	    }
	
	    if(pragma.containsKey("always alloc")) {
	        parseSet((Set<String>)pragma.get("alloc"), createSet);
	        always = true;
	    }
	
	    if(pragma.containsKey("to")) {
	        parseSet((Set<String>)pragma.get("to"), copyinSet);
	    }
	
	    if(pragma.containsKey("always to")) {
	        parseSet((Set<String>)pragma.get("always to"), copyinSet);
	        always = true;
	    }
	
	    if(pragma.containsKey("from")) {
	        parseSet((Set<String>)pragma.get("from"), copyoutSet);
	    }
	
	    if(pragma.containsKey("always from")) {
	        parseSet((Set<String>)pragma.get("always from"), copyoutSet);
	        always = true;
	    }
	
	    if(pragma.containsKey("tofrom")) 
	    {
	        parseSet((Set<String>)pragma.get("tofrom"), copySet);
	    }
	
	    if(pragma.containsKey("always tofrom")) 
	    {
	        parseSet((Set<String>)pragma.get("always tofrom"), copySet);
	        always = true;
	    }
	
	    if(pragma.containsKey("use_device_ptr"))
	    {
	        parseSet((Set<String>)pragma.get("use_device_ptr"), presentSet);
	    }
	
	    if(pragma.containsKey("is_device_ptr"))
	    {
	        parseSet((Set<String>)pragma.get("is_device_ptr"), devicePtrSet);
	    }
	
	    if(pragma.containsKey("delete"))
	    {
	        parseSet((Set<String>)pragma.get("delete"), deleteSet);
	    }
	
	    if(pragma.containsKey("always delete"))
	    {
	        parseSet((Set<String>)pragma.get("always delete"), deleteSet);
	        always = true;
	    }
	
	    if(pragma.containsKey("release"))
	    {
	        parseSet((Set<String>)pragma.get("release"), releaseSet);
	    }
	
	    if(pragma.containsKey("always release"))
	    {
	        parseSet((Set<String>)pragma.get("always release"), releaseSet);
	        always = true;
	    }
	
	    if(pragma.containsKey("in"))
	    {
	        parseSet((Set<String>)pragma.get("in"), inSet);
	        numDependVariables += inSet.size();
	    }
	
	    if(pragma.containsKey("out"))
	    {
	        parseSet((Set<String>)pragma.get("out"), outSet);
	        numDependVariables += outSet.size();
	    }
	
	    if(pragma.containsKey("inout"))
	    {
	        parseSet((Set<String>)pragma.get("inout"), inoutSet);
	        numDependVariables += inoutSet.size();
	    }
	
	    boolean createAccDirective = false;
	    if((createSet.size() > 0) || (copyinSet.size() > 0) || (copyoutSet.size() > 0) || 
	    		(copySet.size() > 0) || (presentSet.size() > 0) || (devicePtrSet.size() > 0) ||
	    		(releaseSet.size() > 0 ) || (deleteSet.size() > 0) )
	    {
	    	createAccDirective = true;
	    }
	    ACCAnnotation newAnnot = new ACCAnnotation();
	    if( regionType == OmpRegion.Target ) {
	    	if( at.containsAnnotation(OmpAnnotation.class, "teams") || at.containsAnnotation(OmpAnnotation.class, "distribute") ) {
	    		newAnnot.put("parallel", "_directive");
	    		createAccDirective = true;
	    	} else if( at.containsAnnotation(OmpAnnotation.class, "parallel") ) {
	    		boolean createAccKernel = false;
	    		if( at.containsAnnotation(OmpAnnotation.class, "single") ) {
	    			List<Traversable> children = null;
	    			if( at instanceof CompoundStatement ) {
	    				children = at.getChildren();
	    			} else if( at instanceof Loop ) {
	    				children = ((Loop)at).getBody().getChildren();
	    			} else {
	    				createAccKernel = true;
	    			}
	    			if( (children != null) && (!children.isEmpty()) ) {
	    				for( Traversable tr : children ) {
	    					Annotatable att = (Annotatable)tr;	
	    					if(!att.containsAnnotation(OmpAnnotation.class, "task") && !att.containsAnnotation(OmpAnnotation.class, "taskwait")) {
	    						createAccKernel = true;
	    						break;
	    					}
	    				}
	    				if( !createAccKernel ) {
	    					removePragma = false;
	    					Set<SubArray> taskInSet = new HashSet<SubArray>();
	    					Set<SubArray> taskOutSet = new HashSet<SubArray>();
	    					Set<SubArray> taskInoutSet = new HashSet<SubArray>();
	    					List<Statement> insertRefList = new LinkedList<Statement>();
	    					List<Statement> insertTargetList = new LinkedList<Statement>();
	    					for( Traversable tr : children ) {
	    						Annotatable att = (Annotatable)tr;
	    						OmpAnnotation taskAnnot = att.getAnnotation(OmpAnnotation.class, "task");
	    						OmpAnnotation taskwaitAnnot = att.getAnnotation(OmpAnnotation.class, "taskwait");
	    						int taskNumDependVariables = 0;
	    						if( taskAnnot != null ) {
	    							ACCAnnotation parallelAnnot = new ACCAnnotation();
	    							parallelAnnot.put("parallel", "_directive");
	    							if( condition != null ) {
	    								parallelAnnot.put("if", condition.clone());
	    							}
	    							att.annotate(parallelAnnot);
	    							taskInSet.clear();
	    							taskOutSet.clear();
	    							taskInoutSet.clear();
	    							if(taskAnnot.containsKey("in"))
	    							{
	    								parseSet((Set<String>)taskAnnot.get("in"), taskInSet);
	    								taskNumDependVariables += taskInSet.size();
	    							}
	
	    							if(taskAnnot.containsKey("out"))
	    							{
	    								parseSet((Set<String>)taskAnnot.get("out"), taskOutSet);
	    								taskNumDependVariables += taskOutSet.size();
	    							}
	
	    							if(taskAnnot.containsKey("inout"))
	    							{
	    								parseSet((Set<String>)taskAnnot.get("inout"), taskInoutSet);
	    								taskNumDependVariables += taskInoutSet.size();
	    							}
	    							CompoundStatement taskCStmt = handleDependClause(taskAnnot, taskNumDependVariables, taskInSet, taskOutSet, taskInoutSet, taskType.innerTask);
	    							parallelAnnot.put("async", new NameID("openarc_async"));
	    							List<Expression> waitList = new ArrayList<Expression>(defaultNumAsyncQueues);
	    							for(int i=0; i<defaultNumAsyncQueues; i++) {
	    								waitList.add(new ArrayAccess(new NameID("openarc_waits"), new IntegerLiteral(i)));
	    							}
	    							parallelAnnot.put("wait", waitList);
	    							insertRefList.add((Statement)att);
	    							insertTargetList.add(taskCStmt);
	    						} else if( taskwaitAnnot != null ) {
	    							if( !att.containsAnnotation(ACCAnnotation.class, "wait") ) {
	    								ACCAnnotation newWaitAnnot = new ACCAnnotation();
	    								newWaitAnnot.put("wait", "_directive");
	    								att.annotate(newWaitAnnot);
	    								//taskwaitAnnot.remove("taskwait");
	    								//taskwaitAnnot.setSkipPrint(true);
	    								//removeList.add(taskwaitAnnot);
	    							}
	    						}
	    					}	
	    					if( !insertRefList.isEmpty() ) {
	    						for( int k=0; k<insertRefList.size(); k++ ) {
	    							Statement refStmt = insertRefList.get(k);
	    							Statement taskCStmt = insertTargetList.get(k);
	    							((CompoundStatement)refStmt.getParent()).addStatementBefore(
	    									refStmt, taskCStmt);
	    						}
	    					}
	    				}
	    			} else {
	    				createAccKernel = true;
	    			}
	    		} else {
	    			List<Traversable> children = null;
	    			if( at instanceof CompoundStatement ) {
	    				children = at.getChildren();
	    			} else if( at instanceof Loop ) {
	    				children = ((Loop)at).getBody().getChildren();
	    			} else {
	    				createAccKernel = true;
	    			}
	    			if( (children != null) && (children.size() == 1) ) {
	    				Annotatable att = (Annotatable)children.get(0);
	    				if( (att instanceof CompoundStatement) && att.containsAnnotation(OmpAnnotation.class, "single") ) {
	    					List<Traversable> gchildren = att.getChildren();
	    					if( (gchildren != null) && !gchildren.isEmpty() ) {
	    						for( Traversable tr : gchildren ) {
	    							Annotatable gatt = (Annotatable)tr;
	    							if(!gatt.containsAnnotation(OmpAnnotation.class, "task") && !gatt.containsAnnotation(OmpAnnotation.class, "taskwait")) {
	    								createAccKernel = true;
	    								break;
	    							}
	    						}
	    						if( !createAccKernel ) {
	    							removePragma = false;
	    							Set<SubArray> taskInSet = new HashSet<SubArray>();
	    							Set<SubArray> taskOutSet = new HashSet<SubArray>();
	    							Set<SubArray> taskInoutSet = new HashSet<SubArray>();
	    							List<Statement> insertRefList = new LinkedList<Statement>();
	    							List<Statement> insertTargetList = new LinkedList<Statement>();
	    							for( Traversable tr : gchildren ) {
	    								Annotatable gatt = (Annotatable)tr;
	    								OmpAnnotation taskAnnot = gatt.getAnnotation(OmpAnnotation.class, "task");
	    								OmpAnnotation taskwaitAnnot = gatt.getAnnotation(OmpAnnotation.class, "taskwait");
	    								int taskNumDependVariables = 0;
	    								if( taskAnnot != null ) {
	    									ACCAnnotation parallelAnnot = new ACCAnnotation();
	    									parallelAnnot.put("parallel", "_directive");
	    									if( condition != null ) {
	    										parallelAnnot.put("if", condition.clone());
	    									}
	    									gatt.annotate(parallelAnnot);
	    									taskInSet.clear();
	    									taskOutSet.clear();
	    									taskInoutSet.clear();
	    									if(taskAnnot.containsKey("in"))
	    									{
	    										parseSet((Set<String>)taskAnnot.get("in"), taskInSet);
	    										taskNumDependVariables += taskInSet.size();
	    									}
	
	    									if(taskAnnot.containsKey("out"))
	    									{
	    										parseSet((Set<String>)taskAnnot.get("out"), taskOutSet);
	    										taskNumDependVariables += taskOutSet.size();
	    									}
	
	    									if(taskAnnot.containsKey("inout"))
	    									{
	    										parseSet((Set<String>)taskAnnot.get("inout"), taskInoutSet);
	    										taskNumDependVariables += taskInoutSet.size();
	    									}
	    									CompoundStatement taskCStmt = handleDependClause(taskAnnot, taskNumDependVariables, taskInSet, taskOutSet, taskInoutSet, taskType.innerTask);
	    									parallelAnnot.put("async", new NameID("openarc_async"));
	    									List<Expression> waitList = new ArrayList<Expression>(defaultNumAsyncQueues);
	    									for(int i=0; i<defaultNumAsyncQueues; i++) {
	    										waitList.add(new ArrayAccess(new NameID("openarc_waits"), new IntegerLiteral(i)));
	    									}
	    									parallelAnnot.put("wait", waitList);
	    									insertRefList.add((Statement)gatt);
	    									insertTargetList.add(taskCStmt);
	    								} else if( taskwaitAnnot != null ) {
	    									if( !gatt.containsAnnotation(ACCAnnotation.class, "wait") ) {
	    										ACCAnnotation newWaitAnnot = new ACCAnnotation();
	    										newWaitAnnot.put("wait", "_directive");
	    										gatt.annotate(newWaitAnnot);
	    										//taskwaitAnnot.remove("taskwait");
	    										//taskwaitAnnot.setSkipPrint(true);
	    										//removeList.add(taskwaitAnnot);
	    									}
	    								}
	    							}	
	    							if( !insertRefList.isEmpty() ) {
	    								for( int k=0; k<insertRefList.size(); k++ ) {
	    									Statement refStmt = insertRefList.get(k);
	    									Statement taskCStmt = insertTargetList.get(k);
	    									((CompoundStatement)refStmt.getParent()).addStatementBefore(
	    											refStmt, taskCStmt);
	    								}
	    							}
	    						}
	    					} else {
	    						createAccKernel = true;
	    					}
	    				} else {
	    					createAccKernel = true;
	    				}
	    			} else {
	    				createAccKernel = true;
	    			}
	    		}
	    		if( createAccKernel ) {
	    			newAnnot.put("parallel", "_directive");
	    			createAccDirective = true;
	    		} else {
	    			//[FIXME] this will not be attached to any IR.
	    			newAnnot.put("data", "_directive");
	    		}
	    	} else if( at.containsAnnotation(OmpAnnotation.class, "simd") ) {
	    		newAnnot.put("parallel", "_directive");
	    		createAccDirective = true;
	    	} else {
	    		//[FIXME] this will not be attached to any IR.
	    		newAnnot.put("data", "_directive");
	    	}
	    	if( createAccDirective ) {
	    		at.annotate(newAnnot);
	    	}
	    } else if( regionType == OmpRegion.Target_Data ) {
	    	newAnnot.put("data", "_directive");
	    	pragma.getAnnotatable().annotate(newAnnot);
	    } else if( regionType == OmpRegion.Target_Enter_Data ) {
	    	newAnnot.put("enter", "_directive");
	    	newAnnot.put("data", "_directive");
	    	((CompoundStatement)pragma.getAnnotatable().getParent()).addStatementBefore(
	    			(Statement)pragma.getAnnotatable(),
	    			new AnnotationStatement(newAnnot));
	    } else if( regionType == OmpRegion.Target_Exit_Data ) {
	    	newAnnot.put("exit", "_directive");
	    	newAnnot.put("data", "_directive");
	    	((CompoundStatement)pragma.getAnnotatable().getParent()).addStatementBefore(
	    			(Statement)pragma.getAnnotatable(),
	    			new AnnotationStatement(newAnnot));
	    } else if( regionType == OmpRegion.Target_Update ) {
	    	newAnnot.put("update", "_directive");
	    	((CompoundStatement)pragma.getAnnotatable().getParent()).addStatementBefore(
	    			(Statement)pragma.getAnnotatable(),
	    			new AnnotationStatement(newAnnot));
	    }
	    if(createSet.size() > 0) {
	    	newAnnot.put("pcreate", createSet);
	    }
	    if(copyinSet.size() > 0) {
	    	if( regionType == OmpRegion.Target_Update ) {
	    		newAnnot.put("device", copyinSet);
	    	} else if( regionType == OmpRegion.Target_Exit_Data ) {
	    		newAnnot.put("delete", copyinSet);
	    	} else {
	    		if( always ) {
	    			newAnnot.put("copyin", copyinSet);
	    		} else {
	    			newAnnot.put("pcopyin", copyinSet);
	    		}
	    	}
	    }
	    if(copyoutSet.size() > 0) {
	    	if( regionType == OmpRegion.Target_Update ) {
	    		newAnnot.put("host", copyoutSet);
	    	} else if( regionType == OmpRegion.Target_Enter_Data ) {
	    		if( always ) {
	    			newAnnot.put("pcreate", copyinSet);
	    		} else {
	    			newAnnot.put("pcreate", copyinSet);
	    		}
	    	} else {
	    		if( always ) {
	    			newAnnot.put("copyout", copyoutSet);
	    		} else {
	    			newAnnot.put("pcopyout", copyoutSet);
	    		}
	    	}
	    }
	    if(copySet.size() > 0) {
	    	if( regionType == OmpRegion.Target_Enter_Data ) {
	    		if( always ) {
	    			newAnnot.put("copyin", copySet);
	    		} else {
	    			newAnnot.put("pcopyin", copySet);
	    		}
	    	} else if( regionType == OmpRegion.Target_Exit_Data ) {
	    		if( always ) {
	    			newAnnot.put("copyout", copySet);
	    		} else {
	    			newAnnot.put("pcopyout", copySet);
	    		}
	    	} else {
	    		if( always ) {
	    			newAnnot.put("copy", copySet);
	    		} else {
	    			newAnnot.put("pcopy", copySet);
	    		}
	    	}
	    }
	    if(presentSet.size() > 0) {
	    	newAnnot.put("present", presentSet);
	    }
	    if(devicePtrSet.size() > 0) {
	    	newAnnot.put("deviceptr", devicePtrSet);
	    }
	    if(deleteSet.size() > 0) {
	    	newAnnot.put("delete", deleteSet);
	    	newAnnot.put("finalize", "_clause");
	    }
	    if(releaseSet.size() > 0) {
	    	newAnnot.put("delete", releaseSet);
	    }
	    if(condition != null) {
	    	newAnnot.put("if", condition);
	    }
	
	    if( nowait ) {
	    	if( createAccDirective ) {
	    		CompoundStatement cStmt = handleDependClause(pragma, numDependVariables, inSet, outSet, inoutSet, taskType.innerTask);
	    		((CompoundStatement)at.getParent()).addStatementBefore(
	    				(Statement)at, cStmt);
	    		newAnnot.put("async", new NameID("openarc_async"));
	    		List<Expression> waitList = new ArrayList<Expression>(defaultNumAsyncQueues);
	    		for(int i=0; i<defaultNumAsyncQueues; i++) {
	    			waitList.add(new ArrayAccess(new NameID("openarc_waits"), new IntegerLiteral(i)));
	    		}
	    		newAnnot.put("wait", waitList);
	    	} else {
	    		CompoundStatement cStmt = handleDependClause(pragma, numDependVariables, inSet, outSet, inoutSet, taskType.outerTask);
	    		((CompoundStatement)at.getParent()).addStatementBefore(
	    				(Statement)at, cStmt);
	    		FunctionCall newFCall = null;
	    		Symbol newFCallSym = SymbolTools.getSymbolOfName("omp_helper_task_exit", at);
	    		if( newFCallSym == null ) {
	    			newFCall = new FunctionCall(new NameID("omp_helper_task_exit"));
	    		} else {
	    			newFCall = new FunctionCall(new Identifier(newFCallSym));
	    		}
	    		Statement exitStmt = new ExpressionStatement(newFCall);
	    		cStmt = new CompoundStatement();
	    		cStmt.addStatement(exitStmt);
	    		((CompoundStatement)at.getParent()).addStatementAfter(
	    				(Statement)at, cStmt);
	    	}
	    }
	
	    if( removePragma ) {
	    	pragma.setSkipPrint(true);
	    	removeList.add(pragma);
	    }
	}

}
