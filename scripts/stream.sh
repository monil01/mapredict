##/usr/bin
#cd ..
rm models/application/stream.aspen
rm applications/memory_research_ornl/aspen_model_generation/stream-CPU/cetus_output/stream.aspen 

make clean
make

pushd ./applications/memory_research_ornl/aspen_model_generation/stream-CPU
./O2GBuild.script
cp cetus_output/stream.aspen ../../../../models/application/
popd
pwd


./mapmc models/application/stream.aspen models/machine/monil.aspen


