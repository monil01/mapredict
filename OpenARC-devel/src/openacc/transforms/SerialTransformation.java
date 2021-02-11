/**
 * 
 */
package openacc.transforms;

import cetus.hir.Tools;
import openacc.analysis.AnalysisTools;
import cetus.hir.*;
import cetus.exec.Driver;
import openacc.hir.*;

import java.util.HashSet;
import java.util.List;
import java.util.LinkedList;
import java.util.ArrayList;
import java.util.Set;

import cetus.transforms.TransformPass;
import cetus.hir.*;
import java.util.*;

/**
 * This pass converts "serial" directives to "parallel num_gangs(1) num_workers(1) vector_length(1)" directives 
 * 
 * @author f6l
 *
 */
public class SerialTransformation extends TransformPass {

  private static String pass_name = "[SerialTransformation]";
  private static Statement insertedInitStmt = null;

  public SerialTransformation(Program program) {
    super(program);
  }

  /* (non-Javadoc)
   * @see cetus.transforms.TransformPass#getPassName()
   */
  @Override
    public String getPassName() {
      return pass_name;
    }

  private void convSerialRegions( List<ACCAnnotation> serialRegions ) {
    //Handle serial regions.
    for( ACCAnnotation cAnnot : serialRegions ) {
      Annotatable at = cAnnot.getAnnotatable();
      PrintTools.println("Before: " + cAnnot.toString(), 0);

      cAnnot.put("num_workers", new IntegerLiteral(1));
      cAnnot.put("num_gangs", new IntegerLiteral(1));
      cAnnot.put("vector_length", new IntegerLiteral(1));
      cAnnot.put("parallel", "_directive");
      cAnnot.remove("serial");

      PrintTools.println("After: " + cAnnot.toString(), 0);
    }
  }


  /* (non-Javadoc)
   * @see cetus.transforms.TransformPass#start()
   */
  @Override
    public void start() {

      List<ACCAnnotation>  cRegionAnnots = AnalysisTools.collectPragmas(program, ACCAnnotation.class, ACCAnnotation.computeRegions, false);

      List<ACCAnnotation> serialRegions = new ArrayList<ACCAnnotation>();
      for( ACCAnnotation cAnnot : cRegionAnnots ) {
        if ( cAnnot.containsKey("serial")) {
          serialRegions.add(cAnnot);
        }

      }

      convSerialRegions(serialRegions);

    }

}
