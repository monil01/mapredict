##/usr/bin
#cd ..
rm models/application/laplace2d.aspen
rm applications/memory_research_ornl/aspen_model_generation/laplace2d/cetus_output/laplace2d.aspen 

make clean
make

pushd ./applications/memory_research_ornl/aspen_model_generation/laplace2d
./O2GBuild.script
cp cetus_output/laplace2d.aspen ../../../../models/application/
popd
pwd


./mapmc models/application/laplace2d.aspen models/machine/monil.aspen


