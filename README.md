# cudaview
Library for visualization of CUDA code with Vulkan

# Build Instructions
To compile the libary and sample CUDA programs that use it, run: `fbuild all`\
The library is built with [fastbuild](https://www.fastbuild.org/docs/home.html).
The required `fbuild` binary is included at the `bin` folder, and can be called
from the project root as `./bin/fbuild`. 

For now, CUDA programs using the library must be compiled using fastbuild.
The `fbuild.bff` file at the source root contains sample compilation rules for
such projects.

# Current features
* Visualization of 2D structured and non-structured data
* Synchronous and asynchronous (on separate thread) rendering
