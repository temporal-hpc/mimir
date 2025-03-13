# Mìmir
Library for visualization of CUDA code with Vulkan

## Dependencies

### Required
* [Vulkan](https://vulkan.lunarg.com/sdk/home) 1.2 or higher
* [CUDA](https://developer.nvidia.com/cuda-downloads) 10 or higher (for Vulkan interop)

### Included
Mìmir downloads additional dependencies via the CMake `FetchContent` command:
* [Slang shading language](https://github.com/shader-slang/slang)
* [ImGui](https://github.com/ocornut/imgui)
* [GLFW](https://github.com/glfw/glfw)
* [GLM](https://github.com/g-truc/glm)
* [ImGuiFileDialog](https://github.com/aiekick/ImGuiFileDialog)

## Building

Building from source requires a `C++20` host compiler, `CUDA SDK >= 10.0` and `cmake >= 3.24`.
From the cloned or downloaded source code folder, run:
```cmake
cmake -B build
cmake --build <build_dir> -j
```

The above commands will generate the corresponding makefiles and build the library,
using all available cores to speed up compilation.
Refer to the [CMake documentation](https://cmake.org/cmake/help/latest/manual/cmake.1.html)
for additional command-line settings.

### Build options

Additional options can be passed with `-D` at the build-system generation step.
The following options are currently provided:

`MIMIR_ENABLE_ASAN`:
Enables the address sanitizer [(ASan)](https://github.com/google/sanitizers/wiki/addresssanitizer)
for debugging (slow!). When this option is on, `ASAN_OPTIONS=protect_shadow_gap=0` must be passed
to any program that links to this library to avoid crashed due to interactions with CUDA
[(source)](https://github.com/google/sanitizers/issues/629).

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

* `run_unstructured`: Displays a 2D brownian moving point cloud with various point sizes.
* `run_structured`: As above, but executing and a Jump Flood Algorithm (JFA) CUDA kernel to compute
a Distance Transform (DT) over a regular structured grid, using point positions as JFA seeds.
* `run_image [path_to_image]`: Simple image viewer for visualizing RGBA image formats. No CUDA
kernel is executed in here.
* `run_texture [path_to_image]`: As above, but executes a box filter with periodically varying
radii over the loaded image. Kernel code was ported from the CUDA-Vulkan interop sample program
[vulkanImageCUDA](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/vulkanImageCUDA).
* `run_mesh [path_to_obj]`: Triangle mesh loader and viewer that uses
[tinyobjloader](https://github.com/tinyobjloader/tinyobjloader) to process `.obj` mesh files.
Smooth per-vertex normals are calculated for the loaded mesh and copied to device memory
along with vertex position and triangle vertex indices.
After that, a CUDA kernel deforms the mesh repeatedly along the normal directions for each vertex,
displaying the results in real time after each kernel call.
* `points3d`: As the unstructured sample, but over 3D space.
* `points3dcpy`: As `points3d`, but performing computation in host with OpenMP and transferring
iteration results to device before displaying each time step.
* `run_automata3d`: Demonstrates the mapping of double buffered algorithms using a ping-pong scheme
to display the evolution of a 3D cellular automata. Press enter to advance the simulation.

# Current features
* Visualization of 2D structured and non-structured data
* Synchronous and asynchronous (on separate thread) rendering
* Camera manipulation
* Model transformations (translation, rotation, scale) per view