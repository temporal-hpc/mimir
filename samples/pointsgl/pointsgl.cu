#include <experimental/source_location> // std::source_location
#include <chrono> // std::chrono
#include <fstream> // std::ifstream
#include <iostream> // std::cout
#include <string> // std::stoul
#include <thread> // std::thread
#include <vector> // std::vector

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "nvmlPower.hpp"
#include "camera.hpp"

#include <curand_kernel.h>
#include <cuda_gl_interop.h>

using chrono_tp = std::chrono::time_point<std::chrono::high_resolution_clock>;
using source_location = std::experimental::source_location;

constexpr void checkCuda(cudaError_t code, bool panic = true,
    source_location src = source_location::current())
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA assertion: %s in function %s at %s(%d)\n",
            cudaGetErrorString(code), src.function_name(), src.file_name(), src.line()
        );
        if (panic)
        {
            throw std::runtime_error("CUDA failure!");
        }
    }
}

__global__ void initSystem(float *coords, size_t point_count,
    curandState *global_states, uint3 extent, unsigned seed)
{
    auto points = reinterpret_cast<float3*>(coords);
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx < point_count)
    {
        auto local_state = global_states[tidx];
        curand_init(seed, tidx, 0, &local_state);
        auto rx = extent.x * curand_uniform(&local_state);
        auto ry = extent.y * curand_uniform(&local_state);
        auto rz = extent.z * curand_uniform(&local_state);
        points[tidx] = {rx, ry, rz};
        global_states[tidx] = local_state;
    }
}

__global__ void integrate3d(float *coords, size_t point_count,
    curandState *global_states, uint3 extent)
{
    auto points = reinterpret_cast<float3*>(coords);
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx < point_count)
    {
        auto local_state = global_states[tidx];
        auto p = points[tidx];
        p.x += curand_normal(&local_state);
        if (p.x > extent.x) p.x = extent.x;
        if (p.x < 0) p.x = 0;
        p.y += curand_normal(&local_state);
        if (p.y > extent.x) p.y = extent.y;
        if (p.y < 0) p.y = 0;
        p.z += curand_normal(&local_state);
        if (p.z > extent.z) p.z = extent.z;
        if (p.z < 0) p.z = 0;
        points[tidx] = p;
        global_states[tidx] = local_state;
    }
}

GLuint loadShader(const std::string& shader_path, GLenum shader_type)
{
    std::ifstream file(shader_path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("failed to open file!");
    }

    // Use read position to determine filesize and allocate output buffer
    auto filesize = static_cast<size_t>(file.tellg());
    std::string buffer(filesize, ' ');
    file.seekg(0);
    file.read(&buffer[0], filesize);
    file.close();
    const char *shader_code = buffer.c_str();

    auto shader_id = glCreateShader(shader_type);
    glShaderSource(shader_id, 1, &shader_code, nullptr);
    glCompileShader(shader_id);
    // Compile SPIR-V code
    //glShaderBinary(1, &shader_id, GL_SHADER_BINARY_FORMAT_SPIR_V_ARB, buffer.data(), buffer.size());
    //glSpecializeShader(shader_id, "main", 0, 0, 0);

    int compiled = 0;
    glGetShaderiv(shader_id, GL_COMPILE_STATUS, &compiled);
    if (!compiled)
    {
        GLchar infolog[1024];
        glGetShaderInfoLog(shader_id, 1024, nullptr, infolog);
        printf("Could not compile shader %s: %s\n", shader_path.c_str(), infolog);
        throw std::runtime_error("Shader compilation error!");
    }

    return shader_id;
}

int main(int argc, char *argv[])
{
    cudaGraphicsResource *vbo_res;
    GLuint shader_program;
    GLuint vao;

    float *d_coords       = nullptr;
    curandState *d_states = nullptr;
    unsigned block_size   = 256;
    unsigned seed         = 123456;
    uint3 extent          = {200, 200, 200};

    // Default values for this program
    int width = 1920;
    int height = 1080;
    size_t point_count = 1000;
    int iter_count = 10000;
    if (argc >= 3) { width = std::stoi(argv[1]); height = std::stoi(argv[2]); }
    if (argc >= 4) point_count = std::stoul(argv[3]);
    if (argc >= 5) iter_count = std::stoi(argv[4]);

    printf("opengl,%lu,", point_count);
    GLFWwindow *window = nullptr;
    // Here we would call engine.init(options);
    {
        if (!glfwInit()) { return EXIT_FAILURE; }
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        window = glfwCreateWindow(width, height, "OpenGL/CUDA Interop", nullptr, nullptr);        
        glfwMakeContextCurrent(window);
        gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

        int gl_device_id;
        uint gl_device_count;
        checkCuda(cudaGLGetDevices(&gl_device_count, &gl_device_id, 1, cudaGLDeviceListAll));
        checkCuda(cudaSetDevice(gl_device_id));

        glClearColor(.5f, .5f, .5f, 1.f);
        glEnable(GL_DEPTH_TEST);
    }

    {
        auto vert_shader = loadShader("cudaview/build/samples/shaders/marker_vert.glsl", GL_VERTEX_SHADER);
        auto frag_shader = loadShader("cudaview/build/samples/shaders/marker_geom.glsl", GL_GEOMETRY_SHADER);
        auto geom_shader = loadShader("cudaview/build/samples/shaders/marker_frag.glsl", GL_FRAGMENT_SHADER);
        shader_program = glCreateProgram();
        glAttachShader(shader_program, vert_shader);
        glAttachShader(shader_program, frag_shader);
        glAttachShader(shader_program, geom_shader);
        glLinkProgram(shader_program);
        int success = 0;
        glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
        if (!success)
        {
            GLchar infolog[1024];
            glGetProgramInfoLog(shader_program, 1024, nullptr, infolog);
            printf("Could not link shader program: %s\n", infolog);
            throw std::runtime_error("Shader program error!");
        }

        // Compiled shaders are no longer needed after linking 
        glDeleteShader(vert_shader);
        glDeleteShader(frag_shader);
        glDeleteShader(geom_shader);

        // TODO: Set uniforms
        Camera camera;
        camera.type = Camera::CameraType::LookAt;
        //camera.flipY = true;
        camera.setPosition(glm::vec3(0.f, 0.f, -2.85f)); //(glm::vec3(0.f, 0.f, -3.75f));
        camera.setRotation(glm::vec3(1.5f, -2.5f, 0.f)); //(glm::vec3(15.f, 0.f, 0.f));
        camera.setRotationSpeed(0.5f);
        camera.setPerspective(60.f, (float)width / (float)height, 0.1f, 256.f);

        ModelViewProjection mvp{};
        mvp.model = glm::mat4(1.f);
        mvp.view  = glm::transpose(camera.matrices.view);
        mvp.proj  = glm::transpose(camera.matrices.perspective);

        GLuint ubo;
        glGenBuffers(1, &ubo);
        glBindBuffer(GL_UNIFORM_BUFFER, ubo);
        glBufferData(GL_UNIFORM_BUFFER, 2 * sizeof(glm::mat4), nullptr, GL_STATIC_DRAW);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
        glBindBufferRange(GL_UNIFORM_BUFFER, 0, ubo, 0, 2 * sizeof(glm::mat4));

        glBindBuffer(GL_UNIFORM_BUFFER, ubo);
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(glm::mat4), glm::value_ptr(mvp.view));
        glBufferSubData(GL_UNIFORM_BUFFER, sizeof(glm::mat4), sizeof(glm::mat4), glm::value_ptr(mvp.proj));
        glBindBuffer(GL_UNIFORM_BUFFER, 0);

        // Setup VBO
        GLuint vbo;
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        //glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * point_count, nullptr, GL_DYNAMIC_DRAW);
        checkCuda(cudaGraphicsGLRegisterBuffer(&vbo_res, vbo, cudaGraphicsRegisterFlagsNone));
        
        // Setup VAO
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        GLuint attr_idx = 0;
        glVertexAttribPointer(attr_idx, 3, GL_FLOAT, GL_FALSE, sizeof(float3), (void*)0);
        glEnableVertexAttribArray(attr_idx);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        //checkCuda(cudaMalloc((void**)&d_coords, sizeof(float3) * point_count));
    }

    checkCuda(cudaGraphicsMapResources(1, &vbo_res));
    size_t buffer_size;
    checkCuda(cudaGraphicsResourceGetMappedPointer((void**)&d_coords, &buffer_size, vbo_res));

    checkCuda(cudaMalloc(&d_states, sizeof(curandState) * point_count));
    unsigned grid_size = (point_count + block_size - 1) / block_size;
    initSystem<<<grid_size, block_size>>>(d_coords, point_count, d_states, extent, seed);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaGraphicsUnmapResources(1, &vbo_res)); 

    GPUPowerBegin("gpu", 100);

    // Here we would call engine.display();
    float total_graphics_time = 0;
    chrono_tp last_time = {};
    std::array<float,240> frame_times{};
    size_t total_frame_count = 0;
    for (int i = 0; i < iter_count; ++i)
    {
        glfwPollEvents();

        // Measure frame time
        static chrono_tp start_time = std::chrono::high_resolution_clock::now();
        chrono_tp current_time = std::chrono::high_resolution_clock::now();
        if (i == 0)
        {
            last_time = start_time;
        }
        float frame_time = std::chrono::duration<float, std::chrono::seconds::period>(current_time - last_time).count();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shader_program);
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, point_count);    
        glfwSwapBuffers(window);

        total_frame_count++;
        total_graphics_time += frame_time;
        frame_times[i % frame_times.size()] = frame_time;
        last_time = current_time;

        //if (display) engine.prepareWindow();
        float *d_coords = nullptr;
        checkCuda(cudaGraphicsMapResources(1, &vbo_res));
        size_t buffer_size;
        checkCuda(cudaGraphicsResourceGetMappedPointer((void**)&d_coords, &buffer_size, vbo_res));

        integrate3d<<<grid_size, block_size>>>(d_coords, point_count, d_states, extent);
        checkCuda(cudaDeviceSynchronize());

        //if (display) engine.updateWindow();
        checkCuda(cudaGraphicsUnmapResources(1, &vbo_res));
    }

    auto frame_sample_size = std::min(frame_times.size(), total_frame_count);
    float total_frame_time = 0;
    for (size_t i = 0; i < frame_sample_size; ++i) total_frame_time += frame_times[i];
    auto framerate = frame_times.size() / total_frame_time;

    // Nvml memory report
    {
        nvmlMemory_v2_t meminfo;
        meminfo.version = (unsigned int)(sizeof(nvmlMemory_v2_t) | (2 << 24U));
        nvmlDeviceGetMemoryInfo_v2(getNvmlDevice(), &meminfo);
        
        constexpr double gigabyte = 1024.0 * 1024.0 * 1024.0;
        double freemem = meminfo.free / gigabyte;
        double reserved = meminfo.reserved / gigabyte;
        double totalmem = meminfo.total / gigabyte;
        double usedmem = meminfo.used / gigabyte;
        printf("%f,%lf,%lf,", framerate, freemem, usedmem);
    }

    GPUPowerEnd();

    // Here we would call engine.exit();
    {
        glfwMakeContextCurrent(window);
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    checkCuda(cudaFree(d_states));
    //checkCuda(cudaFree(d_coords));

    return EXIT_SUCCESS;
}
