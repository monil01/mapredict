################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/apps/sparselu/sparselu-global.cu \
../src/apps/sparselu/sparselu-run.cu \
../src/apps/sparselu/sparselu-seq.cu \
../src/apps/sparselu/sparselu-task-dep.cu 

CU_DEPS += \
./src/apps/sparselu/sparselu-global.d \
./src/apps/sparselu/sparselu-run.d \
./src/apps/sparselu/sparselu-seq.d \
./src/apps/sparselu/sparselu-task-dep.d 

OBJS += \
./src/apps/sparselu/sparselu-global.o \
./src/apps/sparselu/sparselu-run.o \
./src/apps/sparselu/sparselu-seq.o \
./src/apps/sparselu/sparselu-task-dep.o 


# Each subdirectory must supply rules for building sources it contributes
src/apps/sparselu/%.o: ../src/apps/sparselu/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/opt/cuda/8.0/cuda/bin/nvcc -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc/ -G -g -lineinfo -O0 -std=c++11 -gencode arch=compute_60,code=sm_60 -m64 -odir "src/apps/sparselu" -M -o "$(@:%.o=%.d)" "$<"
	/opt/cuda/8.0/cuda/bin/nvcc -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc/ -G -g -lineinfo -O0 -std=c++11 --compile --relocatable-device-code=true -gencode arch=compute_60,code=compute_60 -gencode arch=compute_60,code=sm_60 -m64  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


