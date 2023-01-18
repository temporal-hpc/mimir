# EDIT THIS LINE --> set CUDAVIEW according to your system
CUDAVIEW=../../.

# INCLUDE DIRS
INC=-I${CUDAVIEW}/include -I${CUDAVIEW}/contrib/slang/include/

# STATIC LIBS
CV_STATIC_LIB=${CUDAVIEW}/build/src/libcudaview.a 
IMGUI_STATIC_LIB=${CUDAVIEW}/build/contrib/libimgui.a

# DYNAMIC LIBS
DYN_LIBDIRS=-L${CUDAVIEW}/contrib/slang/lib/
DYN_LIBS=-lslang -lvulkan -lglfw

all:
	nvcc -Xcompiler -fopenmp -O3 ${INC} ${DYN_LIBDIRS} ${DYN_LIBS} ${CV_STATIC_LIB} ${IMGUI_STATIC_LIB} main.cu -o prog
