##/usr/bin
#cd ..
rm models/application/xsbench.aspen
rm applications/memory_research_ornl/aspen_model_generation/xsbench/cetus_output/xsbench.aspen 

make clean
make

pushd ./applications/memory_research_ornl/aspen_model_generation/xsbench
./O2GBuild.script
cp cetus_output/xsbench.aspen ../../../../models/application/
popd
pwd


./mapmc models/application/xsbench.aspen models/machine/apachepass.aspen
#./mapmc models/application/vecmul_strided_200.aspen models/machine/apachepass_no_prefetch.aspen


