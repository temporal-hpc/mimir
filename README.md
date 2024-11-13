# Mìmir
Library for visualization of CUDA code with Vulkan

## Dependencies

### Required
* Vulkan 1.2 or higher
* CUDA 10 or higher (for Vulkan interop)

### Included
Mìmir downloads additional dependencies via the CMake `FetchContent` command:
* [Slang](https://github.com/shader-slang/slang)
* [GLFW](https://github.com/glfw/glfw)
* [ImGui](https://github.com/ocornut/imgui)
* [ImGuiFileDialog](https://github.com/aiekick/ImGuiFileDialog)

## Building

Building from source requires a `C++20` host compiler, `CUDA SDK >= 10.0` with the included `nvcc`
device compiler and `cmake >= 3.24`. After cloning or downlading the repository located at
`<mimir_dir>`, to generate a build in folder `<build_dir>` run:
```cmake
cmake -B <build_dir> -S <mimir_dir>
cmake --build <build_dir> --config <mode>
```

Passing `-j` to the build command will use available cores to speed up compilation.

## Installing

From a successful build placed at `<build_dir>`, run:
```cmake
cmake --install <build_dir> --prefix <install_prefix>
```

By default, this will install the library at the standard system library location. Use the
`--prefix` option to change the installation root folder to `<install_prefix>` (this defaults
to the system library path). Another CMake application can then use `find_package(mimir)` to
link with the installed library. If the library was installed to a non-default path
`<install_prefix>`, run:

```cmake
cmake build -DCMAKE_PREFIX_PATH=<install_prefix>
```

## Running samples

A successful build with the above steps allows to run samples from the `build` directory, so
running the unstructured domain sample should be executed as `./samples/<sample_name> <args>`. The
library currently provides the following samples:

* `run_unstructured`: Displays a 2D brownian moving point cloud with various sizes.
* `run_structured`: As above, but executing and a Jump Flood Algorithm (JFA) CUDA kernel to compute
a Distance Transform (DT) over a regular structured grid, using point positions as JFA seeds.
* `run_image [path_to_image]`: Simple image viewer for visualizing RGBA image formats. No CUDA
kernel is executed in here.
* `run_texture [path_to_image]`: As above, but executes a box filter with periodically varying
radii over the loaded image. Kernel code was ported from the
[vulkanImageCUDA](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/vulkanImageCUDA)
CUDA sample program.
* `run_mesh [path_to_off]`: Simple `.off` mesh viewer with no kernel execution. Support for more
mesh formats is planned.
* `points3d`: As the unstructured sample, but over 3D space.
* `points3dcpy`: As `points3d`, but performing computation in host with OpenMP and transferring
iteration results to device before displaying each time step.
* `automata3d`: Demonstrates the mapping of double buffered experiments in a ping-pong fashion to
display the evolution of a 3D cellular automata. Press enter to advance the simulation.

# Current features
* Visualization of 2D structured and non-structured data
* Synchronous and asynchronous (on separate thread) rendering
