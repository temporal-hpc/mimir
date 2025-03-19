struct HostData
{
    float *pos;
    float *vel;
    float *force;
};

void integrateNBodySystemCpu(HostData host_data, float delta_time,
    float damping, float softening_sqr, int body_count
);