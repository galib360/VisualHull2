################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../include/Cube.cpp \
../include/IO.cpp \
../include/MeshReconstruction.cpp \
../include/Triangulation.cpp 

OBJS += \
./include/Cube.o \
./include/IO.o \
./include/MeshReconstruction.o \
./include/Triangulation.o 

CPP_DEPS += \
./include/Cube.d \
./include/IO.d \
./include/MeshReconstruction.d \
./include/Triangulation.d 


# Each subdirectory must supply rules for building sources it contributes
include/%.o: ../include/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I"\usr\local\include\opencv" -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


