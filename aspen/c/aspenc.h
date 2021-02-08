/* Copyright 2013-2015 UT-Battelle, LLC.  See LICENSE.txt for more information. */
#ifndef ASPEN_C_H
#define ASPEN_C_H

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

    typedef struct       AppModel_t      *AppModel_p;
    typedef struct       MachModel_t     *MachModel_p;
    typedef const struct MachComponent_t *MachComponent_p;
    typedef const struct Expression_t    *Expression_p;
    typedef const struct Kernel_t        *Kernel_p;
    typedef struct       ParamMap_t      *ParamMap_p;

    /* --------------------------------------------------------------------- */
    /* Parser                                                                */
    AppModel_p    Aspen_LoadAppModel(const char *fn);
    MachModel_p   Aspen_LoadMachModel(const char *fn);

    /* --------------------------------------------------------------------- */
    /* AppModel                                                              */
    const char   *AppModel_GetName(AppModel_p);
    Kernel_p      AppModel_GetMainKernel(AppModel_p);
    ParamMap_p    AppModel_GetParamMap(AppModel_p a);
    Expression_p  AppModel_GetGlobalArraySizeExpression(AppModel_p);
    Expression_p  AppModel_GetResourceRequirementExpression(AppModel_p a,
                                                           const char *res);

    /* --------------------------------------------------------------------- */
    /* Kernel                                                                */
    const char   *Kernel_GetName(Kernel_p);
    Expression_p  Kernel_GetTimeExpression(Kernel_p, AppModel_p,
                                          MachModel_p, const char*);

    /* --------------------------------------------------------------------- */
    /* MachModel                                                             */
    MachComponent_p MachModel_GetMachine(MachModel_p);
    ParamMap_p      MachModel_GetParamMap(MachModel_p a);

    /* --------------------------------------------------------------------- */
    /* MachComponent                                                         */
    const char   *MachComponent_GetName(MachComponent_p);
    const char   *MachComponent_GetType(MachComponent_p);

    /* --------------------------------------------------------------------- */
    /* ParamMap                                                              */
    ParamMap_p    ParamMap_Create(const char *e, double v);
    void          ParamMap_BeginIteration(ParamMap_p);
    int           ParamMap_NextValue(ParamMap_p,
                                     const char**, Expression_p*);

    /* --------------------------------------------------------------------- */
    /* Expression                                                            */
    char         *Expression_GetText(Expression_p e);
    double        Expression_Evaluate(Expression_p e);
    Expression_p  Expression_Expanded(Expression_p e, ParamMap_p p);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
