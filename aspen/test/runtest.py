#! /usr/bin/env python

# *****************************************************************************
# File: runtest.py
#
# Purpose:
#    Runs a set of tests against sample data files and validates the
#    results against known baselines.  For now, the results are simply
#    a summary of the generated data set (showing the overall structure
#    and a small set of values for each field).
#
# Programmer:  Jeremy Meredith
# Creation:    June 27, 2013
#
# Modifications:
# *****************************************************************************

# -----------------------------------------------------------------------------
#                              Infrastructure
# -----------------------------------------------------------------------------

import subprocess, os, sys

def AddResult(category, exitcode, coutfn, cerrfn):
    runcounts.setdefault(category, 0)
    runcounts[category] += 1

    successcounts.setdefault(category, 0)
    errorcounts.setdefault(category, 0)

    basefile = "baseline/"+coutfn
    currfile = "current/"+coutfn

    if not os.path.exists(basefile):
        # missing baseline result; shouldn't happen
        logfile.write("%s: Missing baseline result\n" % coutfn)
        rebasefile.write("cp %s %s\n" % (currfile, basefile))
    elif not os.path.exists(currfile):
        # missing current result; shouldn't happen
        logfile.write("%s: Missing current result\n" % coutfn)
    else:
        devnull=open("/dev/null","w")
        diff = subprocess.call(["diff", "-q", basefile, currfile], stdout=devnull, stderr=devnull)
        if exitcode == 0:
            if diff:
                logfile.write("%s: Different result; regression possibly introduced.\n" % coutfn)
                difffile.write("echo; echo \"------------ %s ------------\"\n" % coutfn)
                difffile.write("diff ${prefix}%s ${prefix}%s\n" % (basefile, currfile))
                tkdifffile.write("echo; echo \"------------ %s ------------\"\n" % coutfn)
                tkdifffile.write("tkdiff ${prefix}%s ${prefix}%s\n" % (basefile, currfile))
                rebasefile.write("cp %s %s\n" % (currfile, basefile))
            else:
                successcounts[category] += 1
                #logfile.write("%s: Success\n" % coutfn)
        else:
            errorcounts[category] += 1
            if diff:
                logfile.write("%s: Failure, exit code was %d.  See file %s for details\n" % (coutfn, exitcode, "current/"+cerrfn))
            else:
                logfile.write("%s: Failure: same result but exit code was %d.  See file %s for details\n" % (coutfn, exitcode, "current/"+cerrfn))

def RunTest(category, bn, args):
    # assume bn is basename of file name
    coutfn = "%s/%s.out" % (category,bn)
    cerrfn = "%s/%s.err" % (category,bn)
    cout = open("current/" + coutfn, "w")
    cerr = open("current/" + cerrfn, "w")

    try:
        exitcode = subprocess.call(args, stdout=cout, stderr=cerr)
    except:
        cerr.write("In runtest: error executing: %s\n" % args)
        exitcode = 99

    cout.close()
    cerr.close()
    AddResult(category, exitcode, coutfn, cerrfn)

def PrintResults():
    print ""
    totalrun = 0
    totalsuccess = 0
    totalerror = 0
    for k in sections:
        run = runcounts[k]
        success = successcounts[k]
        error = errorcounts[k]
        totalrun += run
        totalsuccess += success
        totalerror += error
        if success == run:
            print "Test case   %-14s:   all %3d tests passed" % ("'%s'"%k, run)
        else:
            print "ERROR: Test %-14s: " % ("'%s'"%k),success,"/",run," tests passed"
    if totalsuccess < totalrun:
        print "\nTesting failure:",totalsuccess,"/",totalrun," tests passed overall"
        if totalerror != 0:
            print "Additionally:",totalerror,"of these were serious failures with non-zero exit code."
        print "\nSee results.txt for more information on failures."
        print "\nRun diffs.sh or tkdiffs.sh to see detailed differences baseline results.\n"
    else:
        if totalerror != 0:
            print "\Failure:",totalerror,"serious failures with non-zero exit code\n"
        else:
            print "\nSUCCESS: all",totalrun," tests passed"
    return totalrun - totalsuccess

def StartSection(category):
    sections.append(category)
    logfile.write("\n===== %s =====\n" % ("%s tests") % category)
    print "Running %s tests" % category

# -----------------------------------------------------------------------------
#                             Test Categories
# -----------------------------------------------------------------------------

#
# Basic machine model only analysis
#
def TestMachine(fn):
    bn = os.path.basename(fn)
    RunTest("machine", bn,
            ["./bigtest", fn])

#
# Basic application model only analysis
#
def TestApplication(fn):
    bn = os.path.basename(fn)
    RunTest("application", bn,
            ["./bigtest", fn])


#
# Joint machine + application model analysis
#
def TestCombined(fn1, fn2):
    bn1 = os.path.basename(fn1)
    bn2 = os.path.basename(fn2)
    RunTest("combined", "%s_with_%s" % (bn1, bn2),
            ["./bigtest", fn1, fn2])

#
# Tests for examples
#
def TestExampleRuntime(fn1, fn2, socket):
    bn1 = os.path.basename(fn1)
    bn2 = os.path.basename(fn2)
    RunTest("exampleruntime", "%s_with_%s" % (bn1, bn2),
            ["../examples/basic_runtime", fn1, fn2, socket])

#
# Test roofline models
#
def TestRoofline(fn, socket):
    bn = os.path.basename(fn)
    RunTest("rooflineplot", "%s_%s.gplot" % (bn, socket),
            ["../tools/analysis/roofline", fn, socket])

#
# Test in-memory parsing.
#
def TestInMemory():
    RunTest("inmemory", "inmemory",
            ["./parsestring"])


#
# Basic resource analysis for applications.
#
def TestResource(fn, res, var="", values=[]):
    bn = os.path.basename(fn)
    if (var == ""):
        RunTest("resource", bn + "_" + res,
                ["../tools/analysis/resourcecount", fn, res])
    else:
        RunTest("resource", bn + "_" + res,
                ["../tools/analysis/resourcecount", fn, res, var] + values)


#
# Black-box modeling tool tests.
#
def TestBlackbox(csv,template,output):
    bn = os.path.basename(csv)
    RunTest("modeling", csv + "_" + template,
            ["../tools/modeling/blackbox", csv, template, output])


# -----------------------------------------------------------------------------
#                               Run Tests
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    runcounts = {}
    successcounts = {}
    errorcounts = {}
    sections = []
    logfile = open("results.txt", "w")
    difffile = open("diffs.sh", "w")
    tkdifffile = open("tkdiffs.sh", "w")
    rebasefile = open("rebaseline.sh", "w")

    tkdifffile.write('prefix=""\n')
    tkdifffile.write('if (test `basename \`pwd\`` != "test"); then\n')
    tkdifffile.write('   prefix="test/"\n')
    tkdifffile.write('fi\n');

    difffile.write('prefix=""\n')
    difffile.write('if (test `basename \`pwd\`` != "test"); then\n')
    difffile.write('   prefix="test/"\n')
    difffile.write('fi\n');

    StartSection("machine")
    TestMachine("../models/machine/keeneland.aspen")
    TestMachine("../models/machine/fermiCluster.aspen")
    TestMachine("../models/machine/1cpu1gpu.aspen")

    StartSection("application")
    TestApplication("../models/md/md.aspen")
    TestApplication("../models/md/miniMD.aspen")
    TestApplication("../models/comd/CoMD.aspen")
    TestApplication("../models/fft/1D_FFT.aspen")
    TestApplication("../models/matmul/matmul.aspen")
    TestApplication("../models/3dfft/3D_FFT.aspen")
    TestApplication("../models/echelon/tiling/tile.aspen")
    TestApplication("../models/echelon/full/sscp.aspen")
    TestApplication("../models/examples/callstack1.aspen")
    TestApplication("../models/examples/import1_a.aspen")
    TestApplication("../models/examples/import2_a.aspen")
    TestApplication("../models/examples/import3_a.aspen")
    TestApplication("../models/examples/import4_a.aspen")
    TestApplication("../models/examples/import5_a.aspen")
    TestApplication("../models/examples/import6_a.aspen")
    TestApplication("../models/examples/import7_a.aspen")
    TestApplication("../models/examples/array.aspen")

    StartSection("combined")
    TestCombined("../models/machine/keeneland.aspen", "../models/matmul/matmul.aspen")
    TestCombined("../models/machine/keeneland.aspen", "../models/fft/1D_FFT.aspen")
    TestCombined("../models/machine/fermiCluster.aspen", "../models/md/miniMD.aspen")
    TestCombined("../models/machine/fermiCluster.aspen", "../models/3dfft/3D_FFT.aspen")

    StartSection("exampleruntime")
    TestExampleRuntime("../models/machine/simple.aspen", "../models/examples/maxflops_1.aspen", "SimpleCPU")
    TestExampleRuntime("../models/machine/simple.aspen", "../models/examples/maxflops_2.aspen", "SimpleCPU")
    TestExampleRuntime("../models/machine/simple.aspen", "../models/examples/maxloads_1.aspen", "SimpleCPU")
    TestExampleRuntime("../models/machine/simple.aspen", "../models/examples/maxloads_2.aspen", "SimpleCPU")
    TestExampleRuntime("../models/machine/simple_2nodes.aspen", "../models/examples/messages_1.aspen", "SimpleCPU")
    TestExampleRuntime("../models/machine/simple_2nodes.aspen", "../models/examples/messages_2.aspen", "SimpleCPU")

    StartSection("inmemory")
    TestInMemory()

    StartSection("rooflineplot")
    TestRoofline("../models/machine/keeneland.aspen", "nvidia_m2090")
    TestRoofline("../models/machine/keeneland.aspen", "intel_xeon_x5660")
    TestRoofline("../models/machine/fermiCluster.aspen", "nvidia_m2090")
    TestRoofline("../models/machine/1cpu1gpu.aspen", "nvidia_m2090")
    TestRoofline("../models/machine/1cpu1gpu.aspen", "intel_xeon_x5660")

    StartSection("resource")
    TestResource("../models/examples/callstack1.aspen", "flops")
    TestResource("../models/examples/callstack1.aspen", "loads")
    TestResource("../models/examples/lookup.aspen", "flops", "n", ["10", "90", "100", "110", "200", "250", "300", "400", "440", "450", "460"])

    if os.path.isfile("../tools/modeling/blackbox"):
        StartSection("modeling")
        TestBlackbox("blackbox_example.csv","blackbox_template.aspen","blackbox_output.aspen")
        TestBlackbox("blackbox_new_example.csv","blackbox_template.aspen","blackbox_output.aspen")
    else:
        print "NOTE: Skipping modeling tests: because the tool was not built."

    errors = PrintResults()

    logfile.close()
    difffile.close()
    tkdifffile.close()
    rebasefile.close()

    sys.exit(errors)
