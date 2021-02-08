/* Copyright 2013-2015 UT-Battelle, LLC.  See LICENSE.txt for more information. */
#include "aspenc.h"

#include "stdio.h"
#include "stdlib.h"

int main()
{
    AppModel_p app;
    app = Aspen_LoadAppModel("../models/matmul/matmul.aspen");

    const char *name = AppModel_GetName(app);
    printf("Model name = %s\n", name);

    Expression_p globalsize;
    globalsize = AppModel_GetGlobalArraySizeExpression(app);

    printf("\n");

    char *str_size = Expression_GetText(globalsize);
    printf("Global array size = %s\n", str_size);
    free(str_size);

    printf("\n");

    Kernel_p kernel;
    kernel = AppModel_GetMainKernel(app);
    printf("Main kernel name = %s\n", Kernel_GetName(kernel));

    Expression_p flops;
    flops = AppModel_GetResourceRequirementExpression(app, "flops");

    printf("\n");

    char *str_flops = Expression_GetText(flops);
    printf("FLOPS count = %s\n", str_flops);
    free(str_flops);

    printf("\n");

    ParamMap_p appParams = AppModel_GetParamMap(app);
    printf("Param map:\n");
    ParamMap_BeginIteration(appParams);
    const char *n;
    Expression_p e;
    while (ParamMap_NextValue(appParams, &n, &e))
    {
        printf("   \"%s\" => %s\n", n, Expression_GetText(e));
    }

    printf("\n");

    Expression_p flops_exp;
    flops_exp = Expression_Expanded(flops, appParams);

    char *str_flopsexp = Expression_GetText(flops_exp);
    printf("FLOPS count (expanded) = %s\n", str_flopsexp);
    free(str_flopsexp);

    printf("\n");

    MachModel_p amm;
    amm = Aspen_LoadMachModel("../models/machine/simple.aspen");

    MachComponent_p mach;
    mach = MachModel_GetMachine(amm);
    printf("Machine name = %s (type='%s')\n",
           MachComponent_GetName(mach),
           MachComponent_GetType(mach));

    printf("\n");

    Expression_p runtime;
    runtime = Kernel_GetTimeExpression(kernel, app, amm, "SimpleCPU");
    char *str_rt = Expression_GetText(runtime);
    printf("Predicted runtime = %s\n", str_rt);
    free(str_rt);

    ParamMap_p machParams = MachModel_GetParamMap(amm);

    Expression_p rt_exp1;
    rt_exp1 = Expression_Expanded(runtime, ParamMap_Create("n",277));
    printf("\n");
    printf("runtime expanded by n=277 = %s\n", Expression_GetText(rt_exp1));

    Expression_p rt_exp2;
    rt_exp2 = Expression_Expanded(Expression_Expanded(rt_exp1,appParams),machParams);
    printf("\n");
    printf("runtime expanded by app,mach = %s\n", Expression_GetText(rt_exp2));
    printf("\n");
    printf("runtime as value = %lf\n", Expression_Evaluate(rt_exp2));

    printf("\n");

    return 0;
}
