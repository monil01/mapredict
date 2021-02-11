##/usr/bin
#cd ..
rm models/application/xsbench_big_corrected.aspen
rm applications/memory_research_ornl/aspen_model_generation/xsbench_big_corrected/cetus_output/xsbench_big_corrected.aspen 

make clean
make

pushd ./applications/memory_research_ornl/aspen_model_generation/xsbench_big_corrected
./O2GBuild.script
cp cetus_output/xsbench_big_corrected.aspen ../../../../models/application/
popd
pwd

#./mapmc models/application/xsbench_big_corrected.aspen models/machine/oswald00.aspen
#./mapmc models/application/xsbench_big_corrected.aspen models/machine/oswald00_no_prefetch.aspen


#./mapmc models/application/xsbench_big_corrected.aspen models/machine/quad00.aspen
#./mapmc models/application/xsbench_big_corrected.aspen models/machine/quad00_no_prefetch.aspen



#./mapmc models/application/xsbench_big_corrected.aspen models/machine/apachepass.aspen
#./mapmc models/application/xsbench_big_corrected.aspen models/machine/apachepass_no_prefetch.aspen


#./mapmc models/application/xsbench_big_corrected.aspen models/machine/jupiter.aspen
#./mapmc models/application/xsbench_big_corrected.aspen models/machine/jupiter_no_prefetch.aspen


./mapmc models/application/xsbench_big_corrected.aspen models/machine/pegasus.aspen
#./mapmc models/application/xsbench_big_corrected.aspen models/machine/pegasus_no_prefetch.aspen



