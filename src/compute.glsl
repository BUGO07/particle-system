#version 430

layout(local_size_x = 1024) in;

layout(std430, binding = 0) buffer Particles {
    vec4 particles[];
};

layout(std430, binding = 1) buffer Velocities {
    vec4 velocities[];
};

uniform bool uForce;
uniform mat4 uCam;

const float HALF_EXTENT = 5.0;
const vec3 CENTER = vec3(0.0, 0.0, 0.0);
const vec3 GRAVITY = vec3(0.0, -9.81, 0.0);
const float RESTITUTION = 0.04;
const float FORCE_STRENGTH = 30.0;
const float FRICTION = 0.96;
const float DELTA = 1.0 / 60.0;

void main() {
    uint id = gl_GlobalInvocationID.x;
    uint particleCount = particles.length();
    if (id >= particleCount) return;

    vec3 pos = particles[id].xyz;
    float radius = particles[id].w;
    vec3 vel = velocities[id].xyz;

    vel += GRAVITY * DELTA;
    pos += vel * DELTA;

    for (int axis = 0; axis < 3; axis++) {
        float minB = -HALF_EXTENT + radius;
        float maxB =  HALF_EXTENT - radius;

        if (pos[axis] < minB) {
            pos[axis] = minB;
            vel[axis] = -vel[axis] * (1.0 + RESTITUTION);
        } else if (pos[axis] > maxB) {
            pos[axis] = maxB;
            vel[axis] = -vel[axis] * (1.0 + RESTITUTION);
        }
    }

    if (uForce) {
        vec3 to_center = CENTER - pos + normalize(uCam[2].xyz) * 50.0;
        vel += normalize(to_center) * FORCE_STRENGTH * DELTA;
    }

    for (uint j = 0; j < particleCount; j++) {
        if (j == id) continue;

        vec3 other_pos = particles[j].xyz;
        float other_radius = particles[j].w;

        vec3 delta = pos - other_pos;
        float distSq = dot(delta, delta);
        float min_dist = radius + other_radius;
        float min_distSq = min_dist * min_dist;

        if (distSq < min_distSq && distSq > 0.0001) {
            float dist = sqrt(distSq);
            vec3 n = delta / dist; 
            float overlap = min_dist - dist;

            float push = min(overlap * 0.5, 0.01);
            pos += push * n;

            vec3 vj = velocities[j].xyz;
            vel *= FRICTION;
            vel = vel - dot(vel - vj, n) * n * 0.5;
        }
    }

    particles[id].xyz = pos;
    velocities[id].xyz = vel;
}
