################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/apps/jacobi/jacobi-block-global.cu \
../src/apps/jacobi/jacobi-block-task.cu \
../src/apps/jacobi/jacobi-seq.cu \
../src/apps/jacobi/poisson.cu 

CU_DEPS += \
./src/apps/jacobi/jacobi-block-global.d \
./src/apps/jacobi/jacobi-block-task.d \
./src/apps/jacobi/jacobi-seq.d \
./src/apps/jacobi/poisson.d 

OBJS += \
./src/apps/jacobi/jacobi-block-global.o \
./src/apps/jacobi/jacobi-block-task.o \
./src/apps/jacobi/jacobi-seq.o \
./src/apps/jacobi/poisson.o 


# Each subdirectory must supply rules for building sources it contributes
src/apps/jacobi/%.o: ../src/apps/jacobi/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/opt/cuda/8.0/cuda/bin/nvcc -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc/ -G -g -lineinfo -O0 -std=c++11 -gencode arch=compute_60,code=sm_60 -m64 -odir "src/apps/jacobi" -M -o "$(@:%.o=%.d)" "$<"
	/opt/cuda/8.0/cuda/bin/nvcc -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc/ -G -g -lineinfo -O0 -std=c++11 --compile --relocatable-device-code=true -gencode arch=compute_60,code=compute_60 -gencode arch=compute_60,code=sm_60 -m64  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


