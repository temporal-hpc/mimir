#include "check_cuda.hpp" // checkCuda

enum class NBodyConfig { Random, Shell, Expand };

struct DeviceData
{
    float4 *dPos[2];  // mapped host pointers
    float4 *dVel;
};

cudaError_t setSofteningSquared(float value);

void integrateNbodySystem(DeviceData data, unsigned int current_read,
    float delta_time, float damping, unsigned int body_count, int block_size
);

void randomizeBodies(NBodyConfig config, float *pos, float *vel, float *color,
    float cluster_scale, float velocity_scale, int body_count, bool vec4vel
);


