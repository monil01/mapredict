##/usr/bin
#cd ..
rm models/application/lulesh.aspen
rm applications/memory_research_ornl/aspen_model_generation/lulesh/cetus_output/lulesh.aspen 

make clean
make

pushd ./applications/memory_research_ornl/aspen_model_generation/lulesh
./O2GBuild.script
cp cetus_output/lulesh.aspen ../../../../models/application/
popd
pwd


./mapmc models/application/lulesh.aspen models/machine/monil.aspen


