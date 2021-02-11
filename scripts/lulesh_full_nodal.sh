##/usr/bin
#cd ..
rm models/application/lulesh_full_nodal.aspen
rm applications/memory_research_ornl/aspen_model_generation/lulesh_full_nodal/cetus_output/lulesh_full_nodal.aspen 

make clean
make

pushd ./applications/memory_research_ornl/aspen_model_generation/lulesh_full_nodal
./O2GBuild.script
cp cetus_output/lulesh_full_nodal.aspen ../../../../models/application/
popd
pwd


time ./mapmc models/application/lulesh_full_nodal.aspen models/machine/apachepass.aspen


