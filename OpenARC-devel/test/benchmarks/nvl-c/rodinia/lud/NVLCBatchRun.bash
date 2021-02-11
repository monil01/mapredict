#! /bin/bash

function usage()
{
    echo "./NVLCBatchRun.bash"
    echo "List of options:"
    echo -e "\t-h --help"
    echo -e "\t-d=[device] --device=[device]"
    echo -e "\t-n=[iterations] --nItr=[iterations]"
    echo -e "\t-tx --txonly"
    echo ""
    echo "List of target devices:"
    echo -e "\tNVM RD HDD"
    echo ""
}

TX_ONLY=0
device="NVM"
nItr=3
while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | awk -F= '{print $2}'`
    case $PARAM in
        -h | --help)
            usage
            exit
            ;;  
        -d | --device)
			device=$VALUE
            ;;  
        -n | --nItr)
			nItr=$VALUE
            ;;  
        -tx | --txonly)
			TX_ONLY=1
            ;;  
        *)  
            echo "ERROR: unknown parameter \"$PARAM\""
            usage
            exit 1
            ;;  
    esac
    shift
done

arglist=( "64.dat" )
benchmark="lud"

if [ $TX_ONLY -eq 0 ]; then
executablesSFX=( malloc vheap nosafe-norefs nosafe-norefs-poor safe-norefs safe-norefs-poor safe-refs safe-refs-poor safe-refs-persist safe-refs-persist-poor safe-refs-txs1 safe-refs-txs2 safe-refs-txs3 safe-refs-txs1-poor safe-refs-txs2-poor safe-refs-txs3-poor  pmem pmem-poor pmem-txs pmem-txs-poor )
else
executablesSFX=( safe-refs-txs1 safe-refs-txs2 safe-refs-txs3 safe-refs-txs1-poor safe-refs-txs2-poor safe-refs-txs3-poor )
fi

if [ "$device" = "RD" ]; then
device="RD"
NVLFILE="/opt/rd/scratch/${USER}/${benchmark}.nvl"
elif [ "$device" = "HDD" ]; then
device="HDD"
NVLFILE="/tmp/${USER}/${benchmark}.nvl"
else
device="NVM"
NVLFILE="/opt/fio/scratch/${USER}/${benchmark}.nvl"
fi

for arg2 in ${arglist[*]}
do
output="${benchmark}_results_${arg2}_${device}.txt" 
for binarySFX in ${executablesSFX[*]}
do
	binary="${benchmark}-${binarySFX}"
	exeCmd="./${binary} -v -i ../../../../data/rodinia1.0/data/lud/${arg2}"
	if [ ! -f "${binary}" ]; then
		echo "${binary} does not exist; skip!"
		continue
	fi
	i=0
	while [ $i -lt $nItr ]
	do
		rm -f $NVLFILE
		echo "$binary" | grep -E persist\|txs >> /dev/null
		if [ $? -eq 0 ]; then
			echo "PMEM_IS_PMEM_FORCE=1 ${exeCmd}" | tee -a $output
			echo "" | tee -a $output
			PMEM_IS_PMEM_FORCE=1 ${exeCmd} | tee -a $output
		else
			echo "${exeCmd}" | tee -a $output
			echo "" | tee -a $output
			${exeCmd} | tee -a $output
		fi
		i=$((i+1))
		echo "" | tee -a $output
	done
	echo "$binary" | grep -E persist\|txs >> /dev/null
	if [ $? -eq 0 ]; then
		i=0
		while [ $i -lt $nItr ]
		do
			rm -f $NVLFILE
			echo "PMEM_IS_PMEM_FORCE=0 ${exeCmd}" | tee -a $output
			echo "" | tee -a $output
			PMEM_IS_PMEM_FORCE=0 ${exeCmd} | tee -a $output
			i=$((i+1))
			echo "" | tee -a $output
		done
	fi
done
done
	
