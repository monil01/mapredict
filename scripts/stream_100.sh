##/usr/bin
#cd ..
rm models/application/stream_100.aspen
rm applications/memory_research_ornl/aspen_model_generation/stream-CPU_100/cetus_output/stream_100.aspen 

make clean
make

pushd ./applications/memory_research_ornl/aspen_model_generation/stream-CPU_100
./O2GBuild.script
cp cetus_output/stream_100.aspen ../../../../models/application/
popd
pwd


./mapmc models/application/stream_100.aspen models/machine/apachepass.aspen


