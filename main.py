import os
import time
import math
import cv2
import numpy as np
import glfw
import glfw.GLFW as GLFW
from OpenGL.GL import *
from OpenGL.GL import shaders
import ctypes
from perlin_noise import PerlinNoise

def clamp(x) -> float:
    return min(max(x, 0.0), 1.0)

if __name__ == "__main__":
    src = 'src/'
    dst = 'dst/'

    # Determines how many triangles are in the distortion mesh
    # A resolution of 100 means there's 100 triangles horizontally and 100 vertically
    res = 100

    # Determines how large the Perlin noise pattern is
    # A higher number means more detailed textures
    zoom = 20
    noise_coeff = 2.0 / zoom**2

    # Determines how many variations of each image will be created
    variants = 3

    # Sets up glfw
    glfw.init()
    glfw.window_hint(GLFW.GLFW_CONTEXT_VERSION_MAJOR,3)
    glfw.window_hint(GLFW.GLFW_CONTEXT_VERSION_MINOR,3)
    glfw.window_hint(
        GLFW.GLFW_OPENGL_PROFILE, 
        GLFW.GLFW_OPENGL_CORE_PROFILE
    )
    glfw.window_hint(
        GLFW.GLFW_OPENGL_FORWARD_COMPAT, 
        GLFW.GLFW_TRUE
    )
    glfw.window_hint(
        GLFW.GLFW_VISIBLE, 
        GLFW.GLFW_FALSE
    )
    window = glfw.create_window(1000, 1000, "Window", None, None)
    glfw.make_context_current(window)
    
    # Creates the shader
    vertex_shader = shaders.compileShader(
        """
        #version 330 core
        
        layout (location=0) in vec2 vertex_pos;
        layout (location=1) in vec2 noise;
        uniform float noise_coeff;
        out vec2 tex_pos;

        void main() {
            tex_pos = vertex_pos;
            gl_Position = vec4((max(min(vertex_pos + noise*noise_coeff, 1.0), 0.0))*2.0 - 1.0, 0.0, 1.0);
        }
        """, GL_VERTEX_SHADER
    )
    fragment_shader = shaders.compileShader(
        """
        #version 330 core

        in vec2 tex_pos;
        uniform sampler2D tex;
        out vec4 color;

        void main() {
            color = texture(tex, tex_pos);
        }
        """, GL_FRAGMENT_SHADER
    )
    shader = shaders.compileProgram(vertex_shader, fragment_shader)

    start_time = time.time()
    # Defines mesh information
    vertex_count = (res + 1)**2  + (res + 1) // 2
    triangle_count = (2*res + 1) * res
    vertices = np.zeros(vertex_count*4, dtype=np.float32)
    indices = np.zeros(triangle_count*3, dtype=np.int32)

        # Creates vertices
    index = 0
    cols = (res + 1)*4

            # Precomputes rows
    row_a = np.zeros(cols)
    row_a[::4] = [j / res for j in range(res + 1)]
    row_b = np.zeros(cols + 1)
    row_b[::4] = [clamp((2*j - 1)/(2*res)) for j in range(res + 2)]

    for i in range(res + 1):
        y = i / res
        vertices[index:index + cols + i%2] = row_a if i%2==0 else row_b
        vertices[index + 1: index + cols + i%2 + 1:4] = y
        index += (res + 1 + i%2)*4

        # Creates indices
    index = 0
    a = np.array([0, res + 1, res + 2, 0, res + 2, 1])
    for i in range(res):
        indices[index : index+3] = a[0:3] + res if i%2 == 0 else a[3:6]
        index += 3
        quad = a + i%2
        for j in range(res):
            indices[index : index+6] = quad + j
            index += 6
        a += res + 1 + i%2

    # Buffers data to pass to shaders
        # VAO information
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

        # Vertex information
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    
        # Index information
    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # Vertex formatting
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
    
        # Creates a texture object for dst render
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        # Creates a frame buffer object. This is necessary to explicitly set pixel ownership when writing large number of pixels to an image.
    drb = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, drb)
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, drb)

    # Main loop
    vertex_coordinates = list(zip(vertices[::4], vertices[1::4]))
    glUseProgram(shader)
    glUniform1f(glGetUniformLocation(shader, 'noise_coeff'), noise_coeff)
    for path in os.listdir(src):
        # Reads a texture
        image = cv2.imread(f'{src}{path}')
        image = cv2.flip(image, 0)
        width = len(image[0])
        height = len(image)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1 if image.itemsize & 3 else 4)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, image)

        # Updates the rendering size
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height)
        glfw.set_window_size(window, width, height)
        glViewport(0, 0, width, height)

        for version in range(variants):
            # Creates new noise
            noise_sample_x = PerlinNoise(octaves=zoom)
            noise_sample_y = PerlinNoise(octaves=zoom)

            vertices[2::4] = [math.ceil(x%1) * noise_sample_x([x, y]) for x, y in vertex_coordinates]
            vertices[3::4] = [math.ceil(y%1) * noise_sample_y([x, y]) for x, y in vertex_coordinates]
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
            
            # Renders image off-screen
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)
            glBindVertexArray(vao)
            glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
        
            # Saves image
            output = np.zeros((height, width, 3), np.uint8)
            glReadBuffer(GL_COLOR_ATTACHMENT0)
            glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, output)
            output = cv2.flip(output, 0)
            path_extension = path.split('.')
            cv2.imwrite(f'{dst}{path_extension[0]}_{version}.{path_extension[1]}', output)
    print(time.time() - start_time)

    # Clean-up
    glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(2, [vbo, ebo])
    glDeleteFramebuffers(1, [fbo])
    glDeleteTextures(1, [texture])
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    glDeleteProgram(shader)
    glfw.destroy_window(window)
    glfw.terminate()
