##/usr/bin
#cd ..
rm models/application/jacobi_16384.aspen
rm applications/memory_research_ornl/aspen_model_generation/jacobi_16384/cetus_output/jacobi_16384.aspen 

make clean
make

pushd ./applications/memory_research_ornl/aspen_model_generation/jacobi_16384
./O2GBuild.script
cp cetus_output/jacobi_16384.aspen ../../../../models/application/
popd
pwd


./mapmc models/application/jacobi_16384.aspen models/machine/monil.aspen


