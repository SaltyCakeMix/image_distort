import os
import cv2
import numpy as np
import glfw
import glfw.GLFW as GLFW
from OpenGL.GL import *
from OpenGL.GL import shaders
import ctypes

class Triangle:
    def __init__(self, shader):
        # Defines mesh information
        detail = 10
        self.vertex_count = (detail + 1)**2  + (detail + 1) // 2
        self.triangle_count = (2*detail + 1) * detail
        self.vertices = np.zeros(self.vertex_count*2, dtype=np.float32)
        self.indices = np.zeros(self.triangle_count*3, dtype=np.int32)

        index = 0
        for i in range(detail + 1):
            for j in range(detail + (1 if i%2==0 else 2)):
                self.vertices[index] = j / detail if i%2==0 else max(min((2*j - 1)/(2*detail), 1.0), 0.0)
                self.vertices[index + 1] = i / detail
                index += 2

        index = 0
        a = np.array([0, detail + 1, detail + 2, 0, detail + 2, 1])
        for i in range(detail):
            self.indices[index : index+3] = (a[0:3] if i%2 == 0 else a[3:6] + 1) + detail
            index += 3
            quad = a if i%2 == 0 else a + detail + 2
            for j in range(detail):
                self.indices[index : index+6] = quad + j
                index += 6
            if i%2 == 0:
                a += 2*detail + 3

        # Buffers data to pass to shaders
            # VAO information
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

            # Vertex information
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        
            # Index information
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

            # Vertex formatting
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        
        # Reads a texture
        image = cv2.imread(f'src/miku.jpeg')
        image = cv2.flip(image, 0)
        self.width = len(image[0])
        self.height = len(image)
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1 if image.itemsize & 3 else 4)
        glTexImage2D(
            GL_TEXTURE_2D, 
            0, 
            GL_RGB, 
            self.width, self.height,
            0,
            GL_BGR, 
            GL_UNSIGNED_BYTE, 
            image)

    def destroy(self):
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo, self.ebo])
        glDeleteTextures(1, [self.texture])

if __name__ == "__main__":
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
    # glfw.window_hint(
    #    GLFW.GLFW_VISIBLE, 
    #    GLFW.GLFW_FALSE
    # )
    window = glfw.create_window(1000, 1000, "Window", None, None)
    glfw.make_context_current(window)
    glClearColor(0.1, 0.2, 0.2, 1)
    
    # Creates the shader
    vertex_shader = shaders.compileShader(
        """
        #version 330 core
        
        layout (location=0) in vec2 vertex_pos;
        out vec2 tex_pos;

        void main() {
            tex_pos = vertex_pos;
            gl_Position = vec4(pow(vertex_pos.x, 2.0)*2.0 - 1.0, pow(vertex_pos.y, 2.0)*2.0 - 1.0, 0.0, 1.0);
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

    src = 'src/'
    dst = 'dst/'

    t = Triangle(shader)
    
    glUseProgram(shader)

    # Main loop
    while not glfw.window_should_close(window):
        if glfw.get_key(window, GLFW.GLFW_KEY_ESCAPE) \
            == GLFW.GLFW_PRESS:
            break
        glfw.poll_events()
        
        glClear(GL_COLOR_BUFFER_BIT)
        glBindVertexArray(t.vao)
        glDrawElements(GL_TRIANGLES, len(t.indices), GL_UNSIGNED_INT, None)
        glfw.swap_buffers(window)
    
    output = np.zeros((1000, 1000, 3), np.uint8)
    glReadPixels(0, 0, 1000, 1000, GL_BGR, GL_UNSIGNED_BYTE, output)
    output = cv2.flip(output, 0)
    cv2.imwrite('dst/test.png', output)

    # Clean-up
    t.destroy()
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    glDeleteProgram(shader)
    glfw.destroy_window(window)
    glfw.terminate()

# Determines how many vertices are in the distortion mesh
# A resolution of 100 means there's 100 vertices horizontally and 100 vertices vertically
#res = 50

# Determines how large the Perlin noise pattern is
# A higher number means more detailed textures
#scale = 10

#for path in os.listdir(src):
#    break

'''
(2) 3 2 2 3

0 3 4   0 4 1           3 7 4
1 4 5   0 5 2   4 7 8   4 8 5
2 5 6           5 8 9   5 9 6


(3) 4 3 3 4 4 3
0 4 5   0 5 1           4 9 5   9 13 14     9 14 10
1 5 6   1 6 2   5 9 10  5 10 6  10 14 15    10 15 11
2 6 7   2 7 3   6 10 11 6 11 7  11 15 16    11 16 12
3 7 8           7 11 12 7 12 8  12 16 17


(4) 5 4 4 5 5 4 4 5
0 5 6   0 6 1               5 11 6  11 16 17    11 17 12                16 22 17
1 6 7   1 7 2   6 11 12     6 12 7  12 17 18    12 18 13    17 22 23    17 23 18
2 7 8   2 8 3   7 12 13     7 13 8  13 18 19    13 19 14    18 23 24    18 24 19
3 8 9   3 9 4   8 13 14     8 14 9  14 19 20    14 20 15    19 24 25    19 25 20
4 9 10          9 14 15     9 15 10 15 20 21                20 25 26    20 26 21

(5)
0 6 7   0 7 1   7 13 14     6 13 7  13 19 20    13 20 14    20 26 27    19 26 20    26 32 33    26 33 27
'''