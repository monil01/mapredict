##/usr/bin
#cd ..
rm models/application/vecmul_strided_50_50M.aspen
rm applications/memory_research_ornl/aspen_model_generation/vecmul_strided_50_50M/cetus_output/vecmul_strided_50_50M.aspen 

make clean
make

pushd ./applications/memory_research_ornl/aspen_model_generation/vecmul_strided_50_50M
./O2GBuild.script
cp cetus_output/vecmul_strided_50_50M.aspen ../../../../models/application/
popd
pwd


./mapmc models/application/vecmul_strided_50_50M.aspen models/machine/apachepass_no_prefetch.aspen
#./mapmc models/application/vecmul_strided_50_50M.aspen models/machine/apachepass.aspen


