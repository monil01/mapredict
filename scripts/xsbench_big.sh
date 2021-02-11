##/usr/bin
#cd ..
rm models/application/xsbench_big.aspen
rm applications/memory_research_ornl/aspen_model_generation/xsbench_big/cetus_output/xsbench_big.aspen 

make clean
make

pushd ./applications/memory_research_ornl/aspen_model_generation/xsbench_big
./O2GBuild.script
cp cetus_output/xsbench_big.aspen ../../../../models/application/
popd
pwd


./mapmc models/application/xsbench_big.aspen models/machine/apachepass.aspen
#./mapmc models/application/vecmul_strided_200.aspen models/machine/apachepass_no_prefetch.aspen


