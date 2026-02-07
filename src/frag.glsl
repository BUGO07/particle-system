#version 330

in vec3 v_normal;
in vec3 v_position;

out vec4 color;

void main() {
    vec3 normal = normalize(v_normal);
    vec3 light_dir = normalize(vec3(1.0, 1.0, 1.0));
    float diffuse = max(dot(normal, light_dir), 0.0);

    color = vec4(vec3(0.5 + 0.5 * diffuse), 1.0);
}