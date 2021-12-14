DEBUG=no

all:
	nvcc -O3 -D${DEBUG} -o prog main.cu libcudaview.a -lcurand -lvulkan -lglfw -I./include
