# cudaview
Library for visualization of CUDA code with Vulkan

# Dependencies
* Vulkan 1.2 or higher
* CUDA 10 or higher (for Vulkan interop)
* GLFW (https://www.glfw.org/)

# Build Instructions
To compile the libary and sample CUDA programs that use it, run: `fbuild all`\
The library is built with [fastbuild](https://www.fastbuild.org/docs/home.html).
The required `fbuild` binary is included at the `bin` folder, and can be called
from the project root as `./bin/fbuild`.

For now, CUDA programs using the library must be compiled using fastbuild.
The `fbuild.bff` file at the source root contains sample compilation rules for
such projects.

# Installing
To install the library, run: `fbuild install`
The installation folder can be changed by modifying the `.InstallPrefix`
variable at `config.bff`. Currently, a bug in fastbuild raises an error when
attempting to install to a system location, which can be solved by running the
included `install.sh` script.

# Current features
* Visualization of 2D structured and non-structured data
* Synchronous and asynchronous (on separate thread) rendering
