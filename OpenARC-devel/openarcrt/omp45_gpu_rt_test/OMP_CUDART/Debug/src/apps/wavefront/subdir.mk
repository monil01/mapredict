################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/apps/wavefront/wavefront-global.cu \
../src/apps/wavefront/wavefront-run.cu \
../src/apps/wavefront/wavefront-task.cu 

CU_DEPS += \
./src/apps/wavefront/wavefront-global.d \
./src/apps/wavefront/wavefront-run.d \
./src/apps/wavefront/wavefront-task.d 

OBJS += \
./src/apps/wavefront/wavefront-global.o \
./src/apps/wavefront/wavefront-run.o \
./src/apps/wavefront/wavefront-task.o 


# Each subdirectory must supply rules for building sources it contributes
src/apps/wavefront/%.o: ../src/apps/wavefront/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/opt/cuda/8.0/cuda/bin/nvcc -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc/ -G -g -lineinfo -O0 -std=c++11 -gencode arch=compute_60,code=sm_60 -m64 -odir "src/apps/wavefront" -M -o "$(@:%.o=%.d)" "$<"
	/opt/cuda/8.0/cuda/bin/nvcc -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc/ -G -g -lineinfo -O0 -std=c++11 --compile --relocatable-device-code=true -gencode arch=compute_60,code=compute_60 -gencode arch=compute_60,code=sm_60 -m64  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


