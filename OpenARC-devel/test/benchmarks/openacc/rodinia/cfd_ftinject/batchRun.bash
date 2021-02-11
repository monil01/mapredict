#! /bin/bash
#############################################################
# This script batch-runs the benchmark.                     #
# - Output Binaries will be stored in $binaryDir directory. #
#############################################################
if [ $# -ge 1 ]; then
	if [ "$1" = "-h" ] || [ "$1" = "-help" ]; then
		echo "====> Usage of this script"
		echo "      $ batchRun.bash [TARGET] [FTVAR] [ITR] [RR] [RMODE] [FTKIND] [NEL] [BLOCK_LENGTH] [NELR] [INPUTFILE]"
		echo "      where TARGET is a fault-injection target (ACC, TCPU, or LLVM),"
		echo "      FTVAR is non-negative number indicating target fault data"
		echo "      (FTVAR < 0 will run binaries for all available target data),"
		echo "      ITR is the number of repetitions per test,"
		echo "      RR represents target resilience regions," 
		echo "      (the right-most bit in its binary representation refer to the RR0, the second-right-most to RR1, etc.)"
        echo "      RMODE sets R_MODE macro," 
        echo "      FTKIND is non-negative number indicating target fault kind"
        echo "      (FTKIND < 0 will generate outputs for all available target kinds),"
		echo "      and NEL, BLOCK_LENGTH and NELR are the input sizes for the input file, INPUTFILE."
		echo "      (default: NEL = 97046, BLOCK_LENGTH = 256 and NELR = 97280 for INPUTFILE = fvcorr.domn.097K)."
		echo ""
		exit
	fi
fi

if [ "$rodinia" = "" ]; then
	rodinia="[path-to-rodinia-benchmark]"
fi

if [ $# -eq 10 ]; then
	target=$1
	ftvar=$2
	numIterations=$3
	RR=$4
	RMODE=$5
	ftkind=$6
	inputData1=$7
	inputData2=$8
	inputData3=$9
	inputFile=${10}
elif [ $# -eq 6 ]; then
	target=$1
	ftvar=$2
	numIterations=$3
	RR=$4
	RMODE=$5
	ftkind=$6
	inputData1=97046
	inputData2=256
	inputData3=97280
	inputFile="${rodinia}/data/cfd/fvcorr.domn.097K"
elif [ $# -eq 5 ]; then
	target=$1
	ftvar=$2
	numIterations=$3
	RR=$4
	RMODE=$5
	ftkind=5
	inputData1=97046
	inputData2=256
	inputData3=97280
	inputFile="${rodinia}/data/cfd/fvcorr.domn.097K"
elif [ $# -eq 4 ]; then
	target=$1
	ftvar=$2
	numIterations=$3
	RR=$4
	RMODE=0
	ftkind=5
	inputData1=97046
	inputData2=256
	inputData3=97280
	inputFile="${rodinia}/data/cfd/fvcorr.domn.097K"
elif [ $# -eq 3 ]; then
	target=$1
	ftvar=$2
	numIterations=$3
	RR=1
	RMODE=0
	ftkind=5
	inputData1=97046
	inputData2=256
	inputData3=97280
	inputFile="${rodinia}/data/cfd/fvcorr.domn.097K"
elif [ $# -eq 2 ]; then
	target=$1
	ftvar=$2
	numIterations=3
	RR=1
	RMODE=0
	ftkind=5
	inputData1=97046
	inputData2=256
	inputData3=97280
	inputFile="${rodinia}/data/cfd/fvcorr.domn.097K"
elif [ $# -eq 1 ]; then
	target=$1
	ftvar=-1
	numIterations=3
	RR=1
	RMODE=0
	ftkind=5
	inputData1=97046
	inputData2=256
	inputData3=97280
	inputFile="${rodinia}/data/cfd/fvcorr.domn.097K"
else
	target="ACC"
	ftvar=-1
	numIterations=3
	RR=1
	RMODE=0
	ftkind=5
	inputData1=97046
	inputData2=256
	inputData3=97280
	inputFile="${rodinia}/data/cfd/fvcorr.domn.097K"
	echo "No command-line input is specified, default values will be used."
	echo "[Default values] TARGET=$target, FTVAR=$ftvar, ITR=$numIterations, RR=$RR, RMODE=$RMODE, FTKIND=$ftkind"
	echo ""
fi	
echo "TARGET: $target"
echo "ITR per test: $numIterations"
echo "RMODE: $RMODE"
echo "NEL: $inputData1"
echo "BLOCK_LENGTH: $inputData2"
echo "NELR: $inputData3"
echo "INPUTFILE: $inputFile"

benchmark="cfd"

#Reference output data to be compared
refOutputs=( "density_NEL_${inputData1}" "density_energy_NEL_${inputData1}" "momentum_NEL_${inputData1}" )
#Output data to be generated.
tOutputs=( "density" "density_energy" "momentum" )

printprogress=0

# minimum minutes considered as program hang.
minutes_tokill=60

#numFaultsSet=( 1 2 4 8 16 128 1024 )
numFaultsSet=( 1 )
nfCNT=${#numFaultsSet[@]}
if [ $RMODE -eq 2 ] || [ $RMODE -eq 3 ]; then
numFTBitsSet=( 1 )
else
numFTBitsSet=( 1 2 )
fi
nfbCNT=${#numFTBitsSet[@]}
#Below is used to select which resilience region to inject faults.
#Set only one region at a time.
if ((($RR & 1) == 0)); then
    RES_REGION0=0
else
    RES_REGION0=1
fi
if ((($RR & 2) == 0)); then
    RES_REGION1=0
else
    RES_REGION1=1
fi
if ((($RR & 4) == 0)); then
    RES_REGION2=0
else
    RES_REGION2=1
fi
if ((($RR & 8) == 0)); then
    RES_REGION3=0
else
    RES_REGION3=1
fi
if ((($RR & 16) == 0)); then
    RES_REGION4=0
else
    RES_REGION4=1
fi
targetRegion="${RES_REGION0}${RES_REGION1}${RES_REGION2}${RES_REGION3}${RES_REGION4}"
miscMacros="RES_REGION0=${RES_REGION0},RES_REGION1=${RES_REGION1},RES_REGION2=${RES_REGION2},RES_REGION3=${RES_REGION3},RES_REGION4=${RES_REGION4}"

FTVAR0Set=( "ff_variable" "variables" )
FTVAR1Set=( "old_variables" "variables" )
FTVAR2Set=( "areas" "step_factors" "variables" )
FTVAR3Set=( "elements_surrounding_elements" "fluxes" "normals" "variables" "ff_variable" )
FTVAR4Set=( "fluxes" "old_variables" "step_factors" "variables" )

startFTVAR=0
if [ $RES_REGION0 -eq 1 ]; then
	endFTVAR=${#FTVAR0Set[@]}
	endFTVAR=$((endFTVAR-1))
	FTVARSet=( "${FTVAR0Set[@]}" )
elif [ $RES_REGION1 -eq 1 ]; then
	endFTVAR=${#FTVAR1Set[@]}
	endFTVAR=$((endFTVAR-1))
	FTVARSet=( "${FTVAR1Set[@]}" )
elif [ $RES_REGION2 -eq 1 ]; then
	endFTVAR=${#FTVAR2Set[@]}
	endFTVAR=$((endFTVAR-1))
	FTVARSet=( "${FTVAR2Set[@]}" )
elif [ $RES_REGION3 -eq 1 ]; then
	endFTVAR=${#FTVAR3Set[@]}
	endFTVAR=$((endFTVAR-1))
	FTVARSet=( "${FTVAR3Set[@]}" )
elif [ $RES_REGION4 -eq 1 ]; then
	endFTVAR=${#FTVAR4Set[@]}
	endFTVAR=$((endFTVAR-1))
	FTVARSet=( "${FTVAR4Set[@]}" )
else
	endFTVAR=-1
fi
if [ $ftvar -ge 0 ]; then
	startFTVAR=$ftvar
	endFTVAR=$ftvar
fi
echo "startFTVAR = $startFTVAR"
echo "endFTVAR = $endFTVAR"

if [ $ftkind -ge 0 ]; then
    startFTKIND=$ftkind
    endFTKIND=$ftkind
else
    startFTKIND=0
    endFTKIND=7
fi
echo "startFTKIND = $startFTKIND"
echo "endFTKIND = $endFTKIND"

if [ "$openarc" = "" ] || [ ! -d "$openarc" ]; then
	echo "[ERROR] Environment variable, openarc, should be set up correctly to run this script; exit."
	echo ""
	exit 1
fi

baseDir="${openarc}/test/benchmarks/openacc/rodinia/${benchmark}_ftinject"
binaryDirBase="${baseDir}/bin"
if [ $RMODE -eq 0 ] || [ $RMODE -eq 1 ]; then
    binaryDir="${binaryDirBase}/${target}"
elif [ $RMODE -eq 2 ] || [ $RMODE -eq 3 ]; then
    #ftprofile mode
    binaryDir="${binaryDirBase}/${target}_PF"
elif [ $RMODE -eq 4 ] || [ $RMODE -eq 5 ]; then
    #ftprofile mode
    binaryDir="${binaryDirBase}/${target}_PD"
fi

if [ ! -d "$binaryDir" ]; then
	echo "[ERROR] Target benchmark output directory ($binaryDir) does not exists; exit!"
	echo ""
	exit 1
fi

function getNewName {
    local i=0
    local tnamebase="ARCrun"
    local tname="${tnamebase}.log"
    while [ -f "$tname" ]; do
        tname="${tnamebase}_${i}.log"
    i=$((i+1)) 
    done
    echo "$tname"
}   
runlog=$(getNewName)

cd ${baseDir}
date > $runlog
echo "" >> $runlog
if [ $RMODE -eq 0 ] || [ $RMODE -eq 1 ]; then
	binary_target="${benchmark}_${target}"
elif [ $RMODE -eq 2 ] || [ $RMODE -eq 3 ]; then
	binary_target="${benchmark}_${target}_PF"
elif [ $RMODE -eq 4 ] || [ $RMODE -eq 5 ]; then
	binary_target="${benchmark}_${target}_PD"
fi
echo "====> Start ${benchmark}" >> ${baseDir}/$runlog
echo "" >> ${baseDir}/$runlog
if [ $printprogress -gt 0 ]; then
	echo "====> Start ${benchmark}"
	echo ""
fi
workDir="${baseDir}"
cd ${workDir}
# Generate reference output if not existing.
if [ ! -f ${binaryDirBase}/${refOutputs[0]} ]; then
	make CPU NEL=${inputData1} BLOCK_LENGTH=${inputData2} NELR=${inputData3}
	cd ${binaryDirBase}
	./cfd_CPU ${inputFile}
	mv "density" "density_NEL_${inputData1}"
	mv "density_energy" "density_energy_NEL_${inputData1}"
	mv "momentum" "momentum_NEL_${inputData1}"
fi

cd ${workDir}

q0=$startFTKIND
while [ $q0 -le $endFTKIND ]
do
FTKIND=$q0
echo "" >> ${baseDir}/$runlog
echo "====> Target Kind: ${FTKIND}" >> ${baseDir}/$runlog
n0=$startFTVAR
while [ $n0 -le $endFTVAR ]
do
	FTVAR=${FTVARSet[$n0]}
	if [ "$FTVAR" = "" ]; then
		FTVAR="default"
	fi  
	echo "" >> ${baseDir}/$runlog
	echo "====> Target Data: ${FTVAR}" >> ${baseDir}/$runlog
	echo "" >> ${baseDir}/$runlog
	if [ $printprogress -gt 0 ]; then
		echo ""
		echo "====> Target Data ${FTVAR}"
		echo ""
	fi
	k=0
	while [ $k -lt $nfCNT ]
	do	
		numFaults=${numFaultsSet[$k]}
		m=0
		while [ $m -lt $nfbCNT ]
		do	
			numFTBits=${numFTBitsSet[$m]}

			cd ${binaryDir}
			targetdir="$FTVAR/FK${q0}_NF${numFaults}_FB${numFTBits}_RR${targetRegion}_RM${RMODE}"
			if [ $printprogress -gt 0 ]; then
				echo "Target Dir: ${binaryDir}/${targetdir}" | tee -a ${baseDir}/$runlog
			else
				echo "Target Dir: ${binaryDir}/${targetdir}" >> ${baseDir}/$runlog
			fi
			if [ -f "${binaryDir}/${targetdir}/${binary_target}" ]; then
				cd "${binaryDir}/${targetdir}"
				outputlog="${benchmark}_out_${FTVAR}_FK${q0}_NF${numFaults}_FB${numFTBits}_RR${targetRegion}_RM${RMODE}.log"
				date > "${outputlog}"
				echo "" >> "${outputlog}"
				it=0
				while [ $it -lt $numIterations ]
				do
					if [ $printprogress -gt 0 ]; then
						echo "" | tee -a "${outputlog}"
						echo "__START__ Benchmark: ${benchmark} Itr: ${it}" | tee -a "${outputlog}"
						echo "" | tee -a "${outputlog}"
					else
						echo "" >> "${outputlog}"
						echo "__START__ Benchmark: ${benchmark} Itr: ${it}" >> "${outputlog}"
						echo "" >> "${outputlog}"
					fi
    				$openarc/test/bin/detect_system_hang.bash ${binary_target} ${outputlog} ${minutes_tokill}&
					exeCmd="./${binary_target} ${inputFile}"
					if [ $printprogress -gt 0 ]; then
						${exeCmd} 2>&1 | tee -a "${outputlog}"
					else
						${exeCmd} 2>&1 >> "${outputlog}"
					fi
     				pid=`ps aux | grep "detect_system_hang.bash ${binary_target}" | grep -v "grep" | awk '{print $2}' | head -1`
    				kill -9 $pid
     				#pid=`ps aux | grep "sleep" | grep -v "grep" | grep $USER | awk '{print $2}' | head -1`
    				#kill -9 $pid
					pp=0
					for refOutput in ${refOutputs[*]}
					do
						tOutput=${tOutputs[$pp]}
						if [ -f ${tOutput} ]; then
							${openarc}/test/bin/compareResults.pl ${binaryDirBase}/${refOutput} ${tOutput} >> "${outputlog}"
						else
							echo "Output file does not exist" >> "${outputlog}"	
						fi
						pp=$((pp+1))
					done
					if [ $printprogress -gt 0 ]; then
						echo "" | tee -a "${outputlog}"
						echo "__END__ Benchmark: ${benchmark} Itr: ${it}" | tee -a "${outputlog}"
						echo "" | tee -a "${outputlog}"
					else
						echo "" >> "${outputlog}"
						echo "__END__ Benchmark: ${benchmark} Itr: ${it}" >> "${outputlog}"
						echo "" >> "${outputlog}"
					fi
				it=$((it+1))
				done

				if [ $printprogress -gt 0 ]; then
					echo "" | tee -a "${outputlog}"
					date | tee -a "${outputlog}"
					$openarc/test/bin/extractSegFt.pl ${outputlog}
				else
					echo "" >> "${outputlog}"
					date >> "${outputlog}"
					$openarc/test/bin/extractSegFt.pl ${outputlog} >> /dev/null
				fi
			else
				echo "[WARNING] target binary is mssing: ${binaryDir}/${targetdir}/${binary_target}" | tee -a ${outputlog} 
			fi
		m=$((m+1))
		done
	k=$((k+1))
	done
n0=$((n0+1))
done
q0=$((q0+1))
done

cd ${baseDir}
echo "====> End ${benchmark}" >> ${baseDir}/$runlog
date >> ${baseDir}/$runlog
echo ""
echo "====> End ${benchmark}"
echo ""
