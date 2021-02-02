##/usr/bin
#cd ..
rm models/application/lulesh_250.aspen
rm applications/memory_research_ornl/aspen_model_generation/lulesh_250/cetus_output/lulesh_250.aspen 

make clean
make

pushd ./applications/memory_research_ornl/aspen_model_generation/lulesh_250
./O2GBuild.script
cp cetus_output/lulesh_250.aspen ../../../../models/application/
popd
pwd


./mapmc models/application/lulesh_250.aspen models/machine/monil.aspen


