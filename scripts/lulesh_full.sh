##/usr/bin
#cd ..
rm models/application/lulesh_full.aspen
rm applications/memory_research_ornl/aspen_model_generation/lulesh_full/cetus_output/lulesh_full.aspen 

make clean
make

pushd ./applications/memory_research_ornl/aspen_model_generation/lulesh_full
./O2GBuild.script
cp cetus_output/lulesh_full.aspen ../../../../models/application/
popd
pwd


time ./mapmc models/application/lulesh_full.aspen models/machine/apachepass.aspen


