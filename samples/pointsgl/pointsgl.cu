#include <experimental/source_location> // std::source_location
#include <fstream> // std::ifstream
#include <iostream> // std::cout
#include <string> // std::stoul
#include <thread> // std::thread
#include <vector> // std::vector

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "nvmlPower.hpp"

#include <curand_kernel.h>
#include <cuda_gl_interop.h>

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

GLuint loadShader(const std::string& spirv_path, GLenum shader_type)
{
    std::ifstream file(spirv_path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("failed to open file!");
    }

    // Use read position to determine filesize and allocate output buffer
    auto filesize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(filesize);

    file.seekg(0);
    file.read(buffer.data(), filesize);
    file.close();

    auto shader_id = glCreateShader(shader_type);
    // Compile SPIR-V code
    glShaderBinary(1, &shader_id, GL_SHADER_BINARY_FORMAT_SPIR_V_ARB, buffer.data(), buffer.size());
    glSpecializeShader(shader_id, "main", 0, 0, 0);

    
    int compiled = 0;
    glGetShaderiv(shader_id, GL_COMPILE_STATUS, &compiled);
    if (!compiled)
    {
        printf("Could not compile shader %s\n", spirv_path.c_str());
        throw std::runtime_error("Shader compilation error!");
    }

    return shader_id;
}

void renderFrame(GLFWwindow *window, GLuint shader_program, GLuint vao, size_t point_count)
{
    glClearColor(.5f, .5f, .5f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(shader_program);
    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, 3);
 
    glfwSwapBuffers(window);
}

int main(int argc, char *argv[])
{
    cudaGraphicsResource *resource;
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
    size_t point_count = 100;
    int iter_count = 10000;
    //PresentOptions present_mode = PresentOptions::Immediate;
    //size_t target_fps = 0;
    bool enable_sync = true;
    bool use_interop = true;
    if (argc >= 3) { width = std::stoi(argv[1]); height = std::stoi(argv[2]); }
    if (argc >= 4) point_count = std::stoul(argv[3]);
    if (argc >= 5) iter_count = std::stoi(argv[4]);
    //if (argc >= 6) present_mode = static_cast<PresentOptions>(std::stoi(argv[5]));
    //if (argc >= 7) target_fps = std::stoul(argv[6]);
    if (argc >= 8) enable_sync = static_cast<bool>(std::stoi(argv[7]));
    if (argc >= 9) use_interop = static_cast<bool>(std::stoi(argv[8]));

    // Determine execution mode for benchmarking
    std::string mode;
    if (width == 0 && height == 0) mode = use_interop? "interop" : "cudaMalloc";
    else mode = enable_sync? "sync" : "desync";
    printf("%s,%lu,", mode.c_str(), point_count);

    /*bool display = true;
    if (width == 0 || height == 0)
    {
        width = height = 10;
        display = false;
    }*/

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
        checkCuda(cudaSetDevice(0));
    }

    if (use_interop)
    {
        auto vert_shader = loadShader("marker_vertexMain.spv", GL_VERTEX_SHADER);
        auto geom_shader = loadShader("marker_geometryMain.spv", GL_GEOMETRY_SHADER);
        auto frag_shader = loadShader("marker_fragmentMain.spv", GL_FRAGMENT_SHADER);
        shader_program = glCreateProgram();
        glAttachShader(shader_program, vert_shader);
        glAttachShader(shader_program, frag_shader);
        glAttachShader(shader_program, geom_shader);
        glLinkProgram(shader_program);
        int success = 0;
        glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
        if (!success)
        {
            throw std::runtime_error("Shader program error!");
        }

        // Compiled shaders are no longer needed after linking 
        glDeleteShader(vert_shader);
        glDeleteShader(frag_shader);
        glDeleteShader(geom_shader);

        // TODO: Set uniforms

/*const char *vertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "void main()\n"
    "{\n"
    "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
    "}\0";
const char *fragmentShaderSource = "#version 330 core\n"
    "out vec4 FragColor;\n"
    "void main()\n"
    "{\n"
    "   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
    "}\n\0";

    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    // check for shader compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    // fragment shader
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // check for shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    // link shaders
    shader_program = glCreateProgram();
    glAttachShader(shader_program, vertexShader);
    glAttachShader(shader_program, fragmentShader);
    glLinkProgram(shader_program);
    // check for linking errors
    glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shader_program, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);*/

    float vertices[] = {
        -0.5f, -0.5f, 0.0f, // left  
         0.5f, -0.5f, 0.0f, // right 
         0.0f,  0.5f, 0.0f  // top   
    }; 

        // Setup VBO
        GLuint vbo;
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        //glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * point_count, nullptr, GL_DYNAMIC_DRAW);
        
        // Setup VAO
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        GLuint attr_idx = 0;
        glVertexAttribPointer(attr_idx, 3, GL_FLOAT, GL_FALSE, sizeof(float3), (void*)0);
        glEnableVertexAttribArray(attr_idx);

        checkCuda(cudaMalloc((void**)&d_coords, sizeof(float3) * point_count));
        /*checkCuda(cudaGraphicsGLRegisterBuffer(&resource, vbo, cudaGraphicsRegisterFlagsNone));
        checkCuda(cudaGraphicsMapResources(1, &resource));
        size_t buffer_size;
        checkCuda(cudaGraphicsResourceGetMappedPointer((void**)&d_coords, &buffer_size, resource));
        checkCuda(cudaGraphicsUnmapResources(1, &resource));*/
    }
    else // Run the simulation without display
    {
        checkCuda(cudaMalloc((void**)&d_coords, sizeof(float3) * point_count));
    }

    checkCuda(cudaMalloc(&d_states, sizeof(curandState) * point_count));
    unsigned grid_size = (point_count + block_size - 1) / block_size;
    initSystem<<<grid_size, block_size>>>(d_coords, point_count, d_states, extent, seed);
    checkCuda(cudaDeviceSynchronize());

    GPUPowerBegin("gpu", 100);

    // Here we would call engine.displayAsync();
    glfwMakeContextCurrent(nullptr);
    auto rendering_thread = std::thread([&]()
    {
        glfwMakeContextCurrent(window);
        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            renderFrame(window, shader_program, vao, point_count);
        }
        glfwMakeContextCurrent(nullptr);
    });

    for (size_t i = 0; i < iter_count; ++i)
    {
        //if (display) engine.prepareWindow();
        integrate3d<<<grid_size, block_size>>>(d_coords, point_count, d_states, extent);
        checkCuda(cudaDeviceSynchronize());
        //if (display) engine.updateWindow();
    }
    //engine.showMetrics();

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
        printf("%lf,%lf,", freemem, usedmem);
    }

    GPUPowerEnd();

    // Here we would call engine.exit();
    {
        rendering_thread.join();
        glfwMakeContextCurrent(window);
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    checkCuda(cudaFree(d_states));
    //checkCuda(cudaFree(d_coords));

    return EXIT_SUCCESS;
}
