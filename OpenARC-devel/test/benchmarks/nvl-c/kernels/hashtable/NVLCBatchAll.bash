#! /bin/bash

function usage()
{
    echo "./NVLCBatchAll.bash"
    echo "List of options:"
    echo -e "\t-h --help"
    echo -e "\t-tx --tx-only"
	echo -e "\t-n=[iterations] --nItr=[iterations]"
    echo -e "\tall --run-all"
    echo -e "\t[list of targets to run]"
    echo ""
    echo "List of targets:"
    echo -e "\tNVM RD HDD"
    echo ""
}

TXOPTION=""
nItr=3
while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | awk -F= '{print $2}'`
    case $PARAM in
        -h | --help)
            usage
            exit
            ;;  
        -tx | --tx-only)
			TXOPTION="-tx"
            ;;  
        -n | --nItr)
            nItr=$VALUE
            ;;  
        all | --run-all)
            RUN_ALL=1
            RUN_TARGETS=( "NVM" "RD" "HDD" )
            echo "Run All"
            ;;  
        NVM | RD | HDD )
            if [ ! -n "$RUN_ALL" ]; then
                RUN_TARGETS=( "${RUN_TARGETS[@]}" $PARAM )
            fi  
            ;;  
        *)  
            echo "ERROR: unknown parameter \"$PARAM\""
            usage
            exit 1
            ;;  
    esac
    shift
done

if [ ${#RUN_TARGETS[@]} -eq 0 ]; then
    RUN_TARGETS=( "NVM" )
fi

###################################
# Modify below for each benchmark #
###################################
benchmark="hashtable"
inputFile="hashtable.c"
inputSize1=1000
inputSize2=1024
makeOption=""
outputFileBase="${benchmark}_results_${inputSize1}_${inputSize2}"

cp "${inputFile}" "${inputFile}_org"

for TARGET in ${RUN_TARGETS[@]}
do
	if [ "$TARGET" = "NVM" ]; then
		if [ ! -d "/opt/fio/scratch/${USER}" ]; then
		mkdir -p "/opt/fio/scratch/${USER}"
		fi
		cat "${inputFile}_org" | sed "s|_NVLFILEPATH_|/opt/fio/scratch/${USER}/${benchmark}.nvl|g" > "${inputFile}"
	elif [ "$TARGET" = "RD" ]; then
		if [ ! -d "/opt/rd/scratch/${USER}" ]; then
		mkdir -p "/opt/rd/scratch/${USER}"
		fi
		cat "${inputFile}_org" | sed "s|_NVLFILEPATH_|/opt/rd/scratch/${USER}/${benchmark}.nvl|g" > "${inputFile}"
	elif [ "$TARGET" = "HDD" ]; then
		if [ ! -d "/tmp/${USER}" ]; then
		mkdir -p "/tmp/${USER}"
		fi
		cat "${inputFile}_org" | sed "s|_NVLFILEPATH_|/tmp/${USER}/${benchmark}.nvl|g" > "${inputFile}"
	fi

	make clean
	make ${makeOption} ROWS_PER_TX=1
	NVLCBatchRun.bash -d=${TARGET} -n=${nItr} ${TXOPTION}
	#mv ${outputFileBase}_${TARGET}.txt ${outputFileBase}_1_${TARGET}.txt

done

mv "${inputFile}_org" "${inputFile}"
