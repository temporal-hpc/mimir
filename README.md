# Mìmir
Library for visualization of CUDA code with Vulkan

# Dependencies
* Vulkan 1.2 or higher
* CUDA 10 or higher (for Vulkan interop)
* GLFW (https://www.glfw.org/)

## Building

From the cloned source directory:

```bash
mkdir build
cd build
cmake ..
make
```

When successful, the above commands should create the `cudaview` library, which
links to a number of included sample programs making use of it.

# Installing
TODO

# Current features
* Visualization of 2D structured and non-structured data
* Synchronous and asynchronous (on separate thread) rendering
