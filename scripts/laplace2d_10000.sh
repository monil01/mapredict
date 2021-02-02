##/usr/bin
#cd ..
rm models/application/laplace2d_10000.aspen
rm applications/memory_research_ornl/aspen_model_generation/laplace2d_10000/cetus_output/laplace2d_10000.aspen 

make clean
make

pushd ./applications/memory_research_ornl/aspen_model_generation/laplace2d_10000
./O2GBuild.script
cp cetus_output/laplace2d_10000.aspen ../../../../models/application/
popd
pwd


./mapmc models/application/laplace2d_10000.aspen models/machine/apachepass.aspen


