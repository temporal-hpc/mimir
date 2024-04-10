# Mìmir
Library for visualization of CUDA code with Vulkan

# Dependencies

### Required
* Vulkan 1.2 or higher
* CUDA 10 or higher (for Vulkan interop)
* CMake 3.24 or higher (for the FindVulkan module)

### Included
The following dependencies are included in the `contrib` folder. Further versions will include them as submodules and provide the option to use user-side installations as well.
* [Slang](https://github.com/shader-slang/slang)
* [GLFW](https://github.com/glfw/glfw)
* [ImGui](https://github.com/ocornut/imgui)
* [ImGuiFileDialog](https://github.com/aiekick/ImGuiFileDialog)

## Building

From the cloned source directory:
```bash
mkdir build
cd build
cmake ..
make
```

Compilation in the last step can be sped up by passing the number of cores to `make` (e.g. `make -j8`). When successful, the above commands should create the `cudaview` library, which links to a number of included sample programs making use of it.

## Running samples

A successful build with the above steps allows to run samples from the `build` directory, so running the unstructured domain sample should be executed as `./samples/run_structured`. The library currently provides the following samples:

* `run_unstructured`: Displays a 2D brownian moving point cloud with various sizes.
* `run_structured`: As above, but executing and a Jump Flood Algorithm (JFA) CUDA kernel to compute a Distance Transform (DT) over a regular structured grid, using point positions
as JFA seeds.
* `run_image [path_to_image]`: Simple image viewer for visualizing RGBA image formats. No CUDA kernel is executed in here.
* `run_texture [path_to_image]`: As above, but executes a box filter with periodically varying radii over the loaded image. Kernel code was ported from the [vulkanImageCUDA](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/vulkanImageCUDA) CUDA sample.
* `run_mesh [path_to_off]`: Simple `.off` mesh viewer with no kernel execution. Support for more mesh formats is planned.
* `points3d`: As the unstructured sample, but over 3D space.
* `points3dcpy`: As `points3d`, but performing computation in host with OpenMP and transferring iteration results to device before displaying each time step.
* `automata3d`: Demonstrates the mapping of double buffered experiments in a ping-pong fashion to display the evolution of a 3D cellular automata. Press enter to advance the simulation.

# Installing
TODO

# Current features
* Visualization of 2D structured and non-structured data
* Synchronous and asynchronous (on separate thread) rendering
