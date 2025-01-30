#include <cmath> // sqrtf

struct HostData
{
    float *pos;
    float *vel;
    float *force;
};

void bodyBodyInteraction(float accel[3], float pos_mass0[4], float pos_mass1[4],
    float softening_sqr)
{
    float r[3];

    // r_01  [3 FLOPS]
    r[0] = pos_mass1[0] - pos_mass0[0];
    r[1] = pos_mass1[1] - pos_mass0[1];
    r[2] = pos_mass1[2] - pos_mass0[2];

    // d^2 + e^2 [6 FLOPS]
    float dist_sqr = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    dist_sqr += softening_sqr;

    // inv_dist_cube =1/dist_sqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    float inv_dist = 1.f / sqrtf((double)dist_sqr);
    float inv_dist_cube = inv_dist * inv_dist * inv_dist;

    // s = m_j * inv_dist_cube [1 FLOP]
    float s = pos_mass1[3] * inv_dist_cube;

    // (m_1 * r_01) / (d^2 + e^2)^(3/2)  [6 FLOPS]
    accel[0] += r[0] * s;
    accel[1] += r[1] * s;
    accel[2] += r[2] * s;
}

void computeNBodyGravitation(float *pos, float *force, float softening_sqr, int body_count)
{
#ifdef OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < body_count; i++)
    {
        int indexForce = 3 * i;

        float acc[3] = {0, 0, 0};

        // We unroll this loop 4X for a small performance boost.
        int j = 0;

        while (j < body_count)
        {
            bodyBodyInteraction(acc, &pos[4 * i], &pos[4 * j], softening_sqr);
            j++;
            bodyBodyInteraction(acc, &pos[4 * i], &pos[4 * j], softening_sqr);
            j++;
            bodyBodyInteraction(acc, &pos[4 * i], &pos[4 * j], softening_sqr);
            j++;
            bodyBodyInteraction(acc, &pos[4 * i], &pos[4 * j], softening_sqr);
            j++;
        }

        force[indexForce] = acc[0];
        force[indexForce + 1] = acc[1];
        force[indexForce + 2] = acc[2];
    }
}

void integrateNBodySystemCpu(HostData host_data, float deltaTime, float damping,
    float softening_sqr, int body_count)
{
    computeNBodyGravitation(host_data.pos, host_data.force, softening_sqr, body_count);

#ifdef OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < body_count; ++i)
    {
        int index = 4 * i;
        int indexForce = 3 * i;

        float pos[3], vel[3], force[3];
        pos[0] = pos[index + 0];
        pos[1] = pos[index + 1];
        pos[2] = pos[index + 2];
        float invMass = pos[index + 3];

        vel[0] = vel[index + 0];
        vel[1] = vel[index + 1];
        vel[2] = vel[index + 2];

        force[0] = force[indexForce + 0];
        force[1] = force[indexForce + 1];
        force[2] = force[indexForce + 2];

        // acceleration = force / mass;
        // new velocity = old velocity + acceleration * deltaTime
        vel[0] += (force[0] * invMass) * deltaTime;
        vel[1] += (force[1] * invMass) * deltaTime;
        vel[2] += (force[2] * invMass) * deltaTime;

        vel[0] *= damping;
        vel[1] *= damping;
        vel[2] *= damping;

        // new position = old position + velocity * deltaTime
        pos[0] += vel[0] * deltaTime;
        pos[1] += vel[1] * deltaTime;
        pos[2] += vel[2] * deltaTime;

        pos[index + 0] = pos[0];
        pos[index + 1] = pos[1];
        pos[index + 2] = pos[2];

        vel[index + 0] = vel[0];
        vel[index + 1] = vel[1];
        vel[index + 2] = vel[2];
    }
}