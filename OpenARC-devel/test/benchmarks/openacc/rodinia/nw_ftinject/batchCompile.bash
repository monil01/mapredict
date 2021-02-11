#! /bin/bash
#############################################################
# This script batch-translates/compiles the benchmark.      #
# - Output Binaries will be stored in $binaryDir directory. #
#############################################################
if [ $# -ge 1 ]; then
	if [ "$1" = "-h" ] || [ "$1" = "-help" ]; then
		echo "====> Usage of this script"
		echo "      $ batchCompile.bash [TARGET] [FTVAR] [RR] [RMODE] [FTKIND]"
		echo "      where TARGET is a fault-injection target (ACC, TCPU, or LLVM),"
		echo "      FTVAR is non-negative number indicating target fault data"
		echo "      (FTVAR < 0 will generate outputs for all available target data),"
		echo "      RR represents target resilience regions, " 
		echo "      (the right-most bit in its binary representation refer to the RR0, the second-right-most to RR1, etc.)"
        echo "      RMODE sets R_MODE macro," 
		echo "      and FTKIND is non-negative number indicating target fault kind"
		echo "      (FTVAR < 0 will generate outputs for all available target kinds)."
		echo ""
		exit
	elif [ "$1" = "-numRR" ]; then
		#return the number of resilience regions
		exit 2
	fi
fi

if [ $# -eq 5 ]; then
	target=$1
	ftvar=$2
	RR=$3
	RMODE=$4
	ftkind=$5
elif [ $# -eq 4 ]; then
	target=$1
	ftvar=$2
	RR=$3
	RMODE=$4
	ftkind=5
elif [ $# -eq 3 ]; then
	target=$1
	ftvar=$2
	RR=$3
	RMODE=0
	ftkind=5
elif [ $# -eq 2 ]; then
	target=$1
	ftvar=$2
	RR=1
	RMODE=0
	ftkind=5
elif [ $# -eq 1 ]; then
	target=$1
	ftvar=-1
	RR=1
	RMODE=0
	ftkind=5
else
	target="ACC"
	ftvar=-1
	RR=1
	RMODE=0
	ftkind=5
	echo "No command-line input is specified, default values will be used."
	echo "[Default values] TARGET=$target, FTVAR=$ftvar, RR=$RR, RMODE=$RMODE, FTKIND=$ftkind"
	echo ""
fi	
echo "TARGET: $target"
echo "RR: $RR"
echo "RMODE: $RMODE"

if [ "$target" = "TCPU" ]; then
	conffile="openarcConf_NORMAL_TCPU.txt"
elif [ "$target" = "LLVM" ]; then
	conffile=""
else
	conffile="openarcConf_NORMAL.txt"
fi

benchmark="nw"

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
targetRegion="${RES_REGION0}${RES_REGION1}"
miscMacros="RES_REGION0=${RES_REGION0},RES_REGION1=${RES_REGION1}"

FTVAR0Set=( "input_itemsets" "referrence" )
FTVAR1Set=( "input_itemsets" "referrence" )

startFTVAR=0
if [ $RES_REGION0 -eq 1 ]; then
	endFTVAR=${#FTVAR0Set[@]}
	endFTVAR=$((endFTVAR-1))
	FTVARSet=( "${FTVAR0Set[@]}" )
elif [ $RES_REGION1 -eq 1 ]; then
	endFTVAR=${#FTVAR1Set[@]}
	endFTVAR=$((endFTVAR-1))
	FTVARSet=( "${FTVAR1Set[@]}" )
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

if [ ! -d "$baseDir" ]; then
	echo "[ERROR] Target benchmark directory ($baseDir) does not exists; exit!"
	echo ""
	exit 1
fi

if [ ! -d "$binaryDir" ]; then
	mkdir -p "$binaryDir"
fi

function getNewName {
    local i=0 
    local tnamebase="ARCcompile"
    local tname="${tnamebase}.log"
    while [ -f "$tname" ]; do
        tname="${tnamebase}_${i}.log"
    i=$((i+1))
    done
    echo "$tname"
}
compilelog=$(getNewName)
    
function getNewName2 {
    local i=0  
    local tnamebase="Makefile"
    local tname="${tnamebase}.local"
    while [ -f "$tname" ]; do
        tname="${tnamebase}_${i}.local"
    i=$((i+1))
    done
    echo "$tname"
}
localmakefile=$(getNewName2)

cd ${baseDir}
date > $compilelog
echo "" >> $compilelog
if [ $RMODE -eq 0 ] || [ $RMODE -eq 1 ]; then
	binary_target="${benchmark}_${target}"
elif [ $RMODE -eq 2 ] || [ $RMODE -eq 3 ]; then
	binary_target="${benchmark}_${target}_PF" 
elif [ $RMODE -eq 4 ] || [ $RMODE -eq 5 ]; then
	binary_target="${benchmark}_${target}_PD" 
fi
echo "====> Start ${benchmark}" >> ${baseDir}/$compilelog
echo "" >> ${baseDir}/$compilelog
echo "====> Start ${benchmark}"
echo ""
workDir="${baseDir}"
cd ${workDir}

q0=$startFTKIND
while [ $q0 -le $endFTKIND ]
do
FTKIND=$q0
echo "" >> ${baseDir}/$compilelog
echo "====> Target Kind: ${FTKIND}" >> ${baseDir}/$compilelog
n0=$startFTVAR
while [ $n0 -le $endFTVAR ]
do
	FTVAR=${FTVARSet[$n0]}
	if [ "$FTVAR" = "" ]; then
		FTVAR="default"
	fi
	echo "" >> ${baseDir}/$compilelog
	echo "====> Target Data: ${FTVAR}" >> ${baseDir}/$compilelog
	echo "" >> ${baseDir}/$compilelog
	echo ""
	echo "====> Target Data ${FTVAR}"
	echo ""
	k=0
	while [ $k -lt $nfCNT ]
	do	
		numFaults=${numFaultsSet[$k]}
		echo "" >> ${baseDir}/$compilelog
		echo "====> Number of Faults: ${numFaults}" >> ${baseDir}/$compilelog
		echo "" >> ${baseDir}/$compilelog
		echo ""
		echo "====> Number of Faults: ${numFaults}"
		echo ""
		m=0
		while [ $m -lt $nfbCNT ]
		do	
			cd ${workDir}
			numFTBits=${numFTBitsSet[$m]}
			echo "" >> ${baseDir}/$compilelog
			echo "====> Number of Faulty Bits: ${numFTBits}" >> ${baseDir}/$compilelog
			echo "" >> ${baseDir}/$compilelog
			echo ""
			echo "====> Number of Faulty Bits: ${numFTBits}"
			echo ""
			FTMacroString="TOTAL_NUM_FAULTS=${numFaults},NUM_FAULTYBITS=${numFTBits},${miscMacros},_FTVAR=${n0},_FTKIND=${q0},R_MODE=${RMODE}"
			if [ "$conffile" = "" ]; then
				#LLVM mode
				make clean
				cp Makefile Makefile.temp
				cat "Makefile.temp" | sed "s/\$(ARCMACRO)/-Warc,-macro=${FTMacroString}/1" > "$localmakefile"
				rm "Makefile.temp"
				make $target -f "$localmakefile" R_MODE=${RMODE} 2>&1 | tee -a ${baseDir}/$compilelog
			else
				cp ${conffile} "openarcConf.txt_tmp"
				cat "openarcConf.txt_tmp" | sed "s/^macro=/&${FTMacroString},/1" > "openarcConf.txt"
				if [ "$target" = "TCPU" ]; then
					mv "openarcConf.txt" "openarcConf.txt_tmp"
					cat "openarcConf.txt_tmp" | sed "s/^macro=.*/&,_FTTHREAD=1/1" > "openarcConf.txt"
				fi
				rm "openarcConf.txt_tmp"
				make clean
				./O2GBuild.script 2>&1 | tee -a ${baseDir}/$compilelog
				make $target R_MODE=${RMODE} 2>&1 | tee -a ${baseDir}/$compilelog
			fi
			cd ${binaryDir}
			targetdir="$FTVAR/FK${q0}_NF${numFaults}_FB${numFTBits}_RR${targetRegion}_RM${RMODE}"
			mkdir -p "$targetdir"
			cd ${binaryDirBase}
			mv "${benchmark}_${target}" "${binaryDir}/${targetdir}/${binary_target}"
			if [ "$target" = "ACC" ]; then
				ls *.ptx > /dev/null 2>&1
				if [ $? -eq 0 ]; then
					mv *.ptx "${binaryDir}/${targetdir}/"
				fi
				ls *.cu > /dev/null 2>&1
				if [ $? -eq 0 ]; then
					mv *.cu "${binaryDir}/${targetdir}/"
				fi
				ls *.cl > /dev/null 2>&1
				if [ $? -eq 0 ]; then
					mv *.cl "${binaryDir}/${targetdir}/"
				fi
				ls *.h > /dev/null 2>&1
				if [ $? -eq 0 ]; then
					mv *.h "${binaryDir}/${targetdir}/"
				fi
			fi
			#Copy other scripts.
		m=$((m+1))
		done
	k=$((k+1))
	done
n0=$((n0+1))
done
q0=$((q0+1))
done

cd ${baseDir}
echo "====> End ${benchmark}" >> ${baseDir}/$compilelog
date >> ${baseDir}/${compilelog}
echo ""
echo "====> End ${benchmark}"
echo ""
