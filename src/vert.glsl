#version 330

// Per-vertex attributes (from sphere mesh)
in vec3 position;

// Per-instance attributes (from your sphere data)
in vec4 particle; // xyz = center, w = radius

uniform mat4 view;
uniform mat4 projection;

out vec3 v_normal;
out vec3 v_position;

void main() {
    // Scale unit sphere by radius and translate to world position
    vec3 world_pos = position * particle.w + particle.xyz;

    v_normal = position; // normal of unit sphere
    v_position = world_pos;

    gl_Position = projection * view * vec4(world_pos, 1.0);
}