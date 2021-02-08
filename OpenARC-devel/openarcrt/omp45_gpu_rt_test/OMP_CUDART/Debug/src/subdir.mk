################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/rt_device.cu \
../src/rt_host.cu 

CU_DEPS += \
./src/rt_device.d \
./src/rt_host.d 

OBJS += \
./src/rt_device.o \
./src/rt_host.o 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/opt/cuda/8.0/cuda/bin/nvcc -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc/ -G -g -lineinfo -O0 -std=c++11 -gencode arch=compute_60,code=sm_60 -m64 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/opt/cuda/8.0/cuda/bin/nvcc -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc/ -G -g -lineinfo -O0 -std=c++11 --compile --relocatable-device-code=true -gencode arch=compute_60,code=compute_60 -gencode arch=compute_60,code=sm_60 -m64  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


