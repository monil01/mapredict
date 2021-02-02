##/usr/bin
#cd ..
rm models/application/jacobi_4096.aspen
rm applications/memory_research_ornl/aspen_model_generation/jacobi_4096/cetus_output/jacobi_4096.aspen 

make clean
make

pushd ./applications/memory_research_ornl/aspen_model_generation/jacobi_4096
./O2GBuild.script
cp cetus_output/jacobi_4096.aspen ../../../../models/application/
popd
pwd


time ./mapmc models/application/jacobi_4096.aspen models/machine/monil.aspen


