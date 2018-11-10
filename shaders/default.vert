#version 330 core

in vec4 Position;
in vec2 Texcoords;
out vec2 v_Texcoords;

void main(void) { 
    v_Texcoords = Texcoords;
    gl_Position = Position;
}
