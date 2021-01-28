##/usr/bin
#cd ..
rm models/application/jacobi.aspen
rm applications/memory_research_ornl/aspen_model_generation/jacobi/cetus_output/jacobi.aspen 

make clean
make

pushd ./applications/memory_research_ornl/aspen_model_generation/jacobi
./O2GBuild.script
cp cetus_output/jacobi.aspen ../../../../models/application/
popd
pwd


./mapmc models/application/jacobi.aspen models/machine/monil.aspen


