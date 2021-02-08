import aspen.Aspen;
import aspen.AppModel;
import aspen.Expression;
import aspen.MachModel;
import aspen.MachComponent;

public class Test
{ 
    public static void main(String[] args)
    {
        //AppModel app = Aspen.LoadAppModel("../models/examples/table.aspen");
        AppModel app = Aspen.LoadAppModel("../models/matmul/matmul.aspen");
        System.out.format("AppModel name: %s\n", app.GetName());

        //MachModel amm = Aspen.LoadMachineModel("../models/machine/1cpu1gpu.aspen");
        MachModel amm = Aspen.LoadMachineModel("../models/machine/simple.aspen");
        MachComponent m = amm.GetMachine();
        System.out.format("Machine name: %s\n", m.GetName());

        Expression f = app.GetResourceRequirementExpression("flops");
        System.out.format("App flops: %s\n", f.GetText());
        
    }
}
