import aspen

print "Loading application model"
app = aspen.LoadAppModel("../models/matmul/matmul.aspen")
print "Successfully loaded app model:", app.GetName()

print
print "Getting app global array size expression"
globalsize = app.GetGlobalArraySizeExpression()

print "result = ",globalsize
print "as text = ",globalsize.GetText()


print
print "Getting global statements (as python list)"
statements = app.GetGlobals()
print "len(statements) =",len(statements)
print "statements      =",statements
print "statements1[0]  =",statements[0]
#print dir(statements)

print
print "Getting app parameter map (as python dict)"
app_params = app.GetParamMap()
print "len(app_params) =",len(app_params)
print "app_params      =",app_params
print "app_params['n'] =",app_params['n'].GetText()
#print dir(app_params)

print
print "Getting flops expression"
flops = app.GetResourceRequirementExpression("flops")

print "result = ",flops

print "as text(default) = ",flops.GetText()
print "as text(C style) = ",flops.GetText(aspen.TextStyle.C)
print "as text(ASPEN)   = ",flops.GetText(aspen.TextStyle.ASPEN)
print "as text(GNUPLOT) = ",flops.GetText(aspen.TextStyle.GNUPLOT)

print "simplified = ",flops.Simplified().GetText()

print "as a number = ",flops.Expanded(app_params).Evaluate()

k = app.GetMainKernel()
print
print "main kernel name =",k.GetName()

print
print "Loading machine model"
amm = aspen.LoadMachineModel("../models/machine/simple.aspen")
#amm = aspen.LoadMachineModel("../models/machine/keeneland.aspen")
print "Successfully loaded mach model"

print
print "Getting AMM parameter map (as python dict)"
amm_params = amm.GetParamMap()
print "amm_params =",amm_params
#print "amm_params:"
#for (x,y) in amm_params.items(): print " ",x,"\t->",y

print
mach = amm.GetMachine()
print "Machine: ",mach
print "   name:",mach.GetName()
print "   type:",mach.GetType()

print
print "Runtime:"
rt1 = k.GetTimeExpression(app,amm,"SimpleCPU")
print rt1

print
print "... raw expression:"
print rt1.GetText()

print
print "... expanded with n -> 277:"
rt2 = rt1.Expanded( {'n':277} )
print rt2.GetText()

print
print "... expanded by app parameters:"
rt3 = rt2.Expanded(app_params)
print rt3.GetText()

print
print "... expanded by mach parameters:"
rt4 = rt3.Expanded(amm_params)
print rt4.GetText()

print
print "... evaluated:"
rt5 = rt4.Evaluate()
print rt5


