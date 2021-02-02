##/usr/bin
#cd ..
rm models/application/lulesh_400.aspen
rm applications/memory_research_ornl/aspen_model_generation/lulesh_400/cetus_output/lulesh_400.aspen 

make clean
make

pushd ./applications/memory_research_ornl/aspen_model_generation/lulesh_400
./O2GBuild.script
cp cetus_output/lulesh_400.aspen ../../../../models/application/
popd
pwd


./mapmc models/application/lulesh_400.aspen models/machine/monil.aspen


