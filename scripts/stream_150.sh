##/usr/bin
#cd ..
rm models/application/stream_150.aspen
rm applications/memory_research_ornl/aspen_model_generation/stream-CPU_150/cetus_output/stream_150.aspen 

make clean
make

pushd ./applications/memory_research_ornl/aspen_model_generation/stream-CPU_150
./O2GBuild.script
cp cetus_output/stream_150.aspen ../../../../models/application/
popd
pwd


./mapmc models/application/stream_150.aspen models/machine/apachepass.aspen


