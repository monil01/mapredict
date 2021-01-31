##/usr/bin
#cd ..
rm models/application/vecmul_strided_200_200M.aspen
rm applications/memory_research_ornl/aspen_model_generation/vecmul_strided_200_200M/cetus_output/vecmul_strided_200_200M.aspen 

make clean
make

pushd ./applications/memory_research_ornl/aspen_model_generation/vecmul_strided_200_200M
./O2GBuild.script
cp cetus_output/vecmul_strided_200_200M.aspen ../../../../models/application/
popd
pwd


./mapmc models/application/vecmul_strided_200_200M.aspen models/machine/apachepass_no_prefetch.aspen
#./mapmc models/application/vecmul_strided_200_200M.aspen models/machine/apachepass.aspen


