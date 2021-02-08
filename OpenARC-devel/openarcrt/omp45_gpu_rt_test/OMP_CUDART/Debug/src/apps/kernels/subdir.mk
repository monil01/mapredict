################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../src/apps/kernels/profiler.c 

CU_SRCS += \
../src/apps/kernels/initialization.cu \
../src/apps/kernels/kernels.cu \
../src/apps/kernels/pragmatic.cu \
../src/apps/kernels/validate.cu 

CU_DEPS += \
./src/apps/kernels/initialization.d \
./src/apps/kernels/kernels.d \
./src/apps/kernels/pragmatic.d \
./src/apps/kernels/validate.d 

OBJS += \
./src/apps/kernels/initialization.o \
./src/apps/kernels/kernels.o \
./src/apps/kernels/pragmatic.o \
./src/apps/kernels/profiler.o \
./src/apps/kernels/validate.o 

C_DEPS += \
./src/apps/kernels/profiler.d 


# Each subdirectory must supply rules for building sources it contributes
src/apps/kernels/%.o: ../src/apps/kernels/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/opt/cuda/8.0/cuda/bin/nvcc -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc/ -G -g -O3 -std=c++11 -gencode arch=compute_60,code=sm_60 -m64 -odir "src/apps/kernels" -M -o "$(@:%.o=%.d)" "$<"
	/opt/cuda/8.0/cuda/bin/nvcc -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc/ -G -g -O3 -std=c++11 --compile --relocatable-device-code=true -gencode arch=compute_60,code=compute_60 -gencode arch=compute_60,code=sm_60 -m64  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/apps/kernels/%.o: ../src/apps/kernels/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/opt/cuda/8.0/cuda/bin/nvcc -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc/ -G -g -O3 -std=c++11 -gencode arch=compute_60,code=sm_60 -m64 -odir "src/apps/kernels" -M -o "$(@:%.o=%.d)" "$<"
	/opt/cuda/8.0/cuda/bin/nvcc -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc/ -G -g -O3 -std=c++11 --compile -m64  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


