#version 460

uniform mat4 view;
uniform mat4 projection;

layout(std430, binding = 0) buffer Particles {
    vec4 particles[];
};

void main() {
    vec4 particle = particles[gl_VertexID];
    vec3 world_pos = particle.xyz + particle.w;

    gl_Position = projection * view * vec4(world_pos, 1.0);
}