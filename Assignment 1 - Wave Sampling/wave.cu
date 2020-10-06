//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/*
    This example demonstrates how to use the Cuda OpenGL bindings to
    dynamically modify a vertex buffer using a Cuda kernel.

    The steps are:
    1. Create an empty vertex buffer object (VBO)
    2. Register the VBO with Cuda
    3. Map the VBO for writing from Cuda
    4. Run Cuda kernel to modify the vertex positions
    5. Unmap the VBO
    6. Render the results using OpenGL

    Host code
First Assignment - Wave Sampling
GPGPU & Accelerator Programming COMP10065 
B00268411 - Paul McLaughlin
*/

#define GLEW_STATIC
#define FREEGLUT_STATIC

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined (__APPLE__) || defined(MACOSX)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#include <vector_types.h>

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms

using namespace std;

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width  = 512;
const unsigned int window_height = 512;

 unsigned int mesh_width    = 256;
 unsigned int mesh_height   = 256;

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

float g_fAnim = 0.0;
//realm of calm globals
float radius = 0.05f;
float radius2 = 0.05f;
float circle_X =0.0f, circle_Y = 0.6f;
float circle_X2 = 0.0f, circle_Y2 = -0.6f;


// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

StopWatchInterface *timer = NULL;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;

int *pArgc = NULL;
char **pArgv = NULL;

#define MAX(a,b) ((a > b) ? a : b)

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv, char *ref_file);
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

// Cuda functionality
void runCuda(struct cudaGraphicsResource **vbo_resource);
void runAutoTest(int devID, char **argv, char *ref_file);
void checkResultCuda(int argc, char **argv, const GLuint &vbo);

const char *sSDKsample = "simpleGL (VBO)";

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void simple_vbo_kernel(float4 *pos, unsigned int width, float radius, float radius2, unsigned int height, float time, float circle_X, float circle_Y, float circle_X2, float circle_Y2)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // calculate uv coordinates
    float u = x / (float) width;
    float v = y / (float) height;
    u = u*2.0f - 1.0f;
    v = v*2.0f - 1.0f;

 
    // calculate simple sine wave pattern
    float freq = 4.0f;
    float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.55f;

	//region of calm
	float r2 = radius2;
	float r = radius;
	float circle_u =u - circle_X, circle_v = v - circle_Y;
	float circle_u2 = u - circle_X2, circle_v2 = v - circle_Y2;
    bool region = (circle_u * circle_u) + (circle_v * circle_v) <= r;
	bool region2 = (circle_u2 * circle_u2) + (circle_v2 * circle_v2) <= r;
	for (int i = 0; i < region; i++) {
		w = 0;
	}

	for (int i = 0; i < region2; i++) {
		w = 0;
	}
	

    // write output vertex
    pos[y*width+x] = make_float4(u, w, v, 1.0f);
}

//vertex addition for jitter kernel
__global__ void jitter_kernel(float4* pos, int mesh_width, float radius, float radius2, float mesh_height, float time, float* vertex_X, float* vertex_Y, float* vertex_Z)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	// write output vertex
	pos[mesh_width + x * y].x += vertex_X[mesh_width + x * y];
	pos[mesh_width + x * y].y += vertex_Y[mesh_width + x *y];
	pos[mesh_width + x * y].z += vertex_Z[mesh_width + x *y];
}


void launch_kernel(float4 *pos, unsigned int mesh_width,
                   unsigned int mesh_height, float time)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    simple_vbo_kernel<<< grid, block>>>(pos, mesh_width, radius, radius2, mesh_height, time, circle_X, circle_Y, circle_X2, circle_Y2);
 
		float* vertex_X;
		float* vertex_Y;
		float* vertex_Z;

		int mesh_volume = mesh_height * mesh_width;

		//allocating memory for the vertices x, y and z
		vertex_X = (float*)malloc(mesh_volume * sizeof(float));
		vertex_Y = (float*)malloc(mesh_volume * sizeof(float));
		vertex_Z = (float*)malloc(mesh_volume * sizeof(float));


		//calculating for jitter effect
		for (int i = 0; i < mesh_volume; i++)
		{
			vertex_X[i] = ((rand() % 100) - 50) / 11000.0f;
			vertex_Y[i] = ((rand() % 100) - 50) / 11000.0f;
			vertex_Z[i] = ((rand() % 100) - 50) / 11000.0f;
		}
		//setting the vertices to device
		float* vertex_X_d;
		float* vertex_Y_d;
		float* vertex_Z_d;

		cudaMalloc((void**)& vertex_X_d, (mesh_volume * sizeof(float)));
		cudaMemcpy(vertex_X_d, vertex_X, (mesh_volume * sizeof(float)), cudaMemcpyHostToDevice);
		cudaMalloc((void**)& vertex_Y_d, (mesh_volume * sizeof(float)));
		cudaMemcpy(vertex_Y_d, vertex_Y, (mesh_volume * sizeof(float)), cudaMemcpyHostToDevice);
		cudaMalloc((void**)& vertex_Z_d, (mesh_volume * sizeof(float)));
		cudaMemcpy(vertex_Z_d, vertex_Z, (mesh_volume * sizeof(float)), cudaMemcpyHostToDevice);


		jitter_kernel << < grid, block >> > (pos, mesh_width, radius, radius2, mesh_height, time, vertex_X_d, vertex_Y_d, vertex_Z_d);

		//free the memory
		cudaFree(vertex_X_d);
		cudaFree(vertex_Y_d);
		cudaFree(vertex_Z_d);
		free(vertex_X);
		free(vertex_Y);
		free(vertex_Z);
	
}


bool checkHW(char *name, const char *gpuType, int dev)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    strcpy(name, deviceProp.name);

    if (!STRNCASECMP(deviceProp.name, gpuType, strlen(gpuType)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    char *ref_file = NULL;

    pArgc = &argc;
    pArgv = argv;

#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    printf("%s starting...\n", sSDKsample);

    if (argc > 1)
    {
        if (checkCmdLineFlag(argc, (const char **)argv, "file"))
        {
            // In this mode, we are running non-OpenGL and doing a compare of the VBO was generated correctly
            getCmdLineArgumentString(argc, (const char **)argv, "file", (char **)&ref_file);
        }
    }

    printf("\n");
	printf("Aim of the game - player 1: get in behind player 2 and use the move backwards button to drag player 2 circle to a point n the y axis, this will end the game\n");
	printf("Aim of the game - player 2: RUN\n");
	printf("Game controls\n");
	printf("W: Move player 1's Circle Forwards\n");
	printf("A: Move player 1's Left\n");
	printf("S: Move player 1's Backwards\n");
	printf("D: Move player 1's Right\n");
	printf("T: Move player 2's Circle Forwards\n");
	printf("G: Move player 2's Left\n");
	printf("F: Move player 2's Backwards\n");
	printf("H: Move player 2's Right\n");
	

    runTest(argc, argv, ref_file);

    printf("%s completed, returned %s\n", sSDKsample, (g_TotalErrors == 0) ? "OK" : "ERROR!");
    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);

        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", avgFPS);
    glutSetWindowTitle(fps);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Cuda GL Interop (VBO)");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);

    // initialize necessary OpenGL extensions
    if (! isGLVersionSupported(2,0))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

    SDK_CHECK_ERROR_GL();

    return true;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool runTest(int argc, char **argv, char *ref_file)
{
    // Create the CUTIL timer
    sdkCreateTimer(&timer);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);

    // command line mode only
    if (ref_file != NULL)
    {
        // create VBO
        checkCudaErrors(cudaMalloc((void **)&d_vbo_buffer, mesh_width*mesh_height*4*sizeof(float)));

        // run the cuda part
        runAutoTest(devID, argv, ref_file);

        // check result of Cuda step
        checkResultCuda(argc, argv, vbo);

        cudaFree(d_vbo_buffer);
        d_vbo_buffer = NULL;
    }
    else
    {
        // First initialize OpenGL context, so we can properly set the GL for CUDA.
        // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
        if (false == initGL(&argc, argv))
        {
            return false;
        }

        // register callbacks
        glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
#if defined (__APPLE__) || defined(MACOSX)
        atexit(cleanup);
#else
        glutCloseFunc(cleanup);
#endif

        // create VBO
        createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

        // run the cuda part
        runCuda(&cuda_vbo_resource);

        // start rendering mainloop
        glutMainLoop();
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource)
{
    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
                                                         *vbo_resource));
    //printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

    // execute the kernel
    //    dim3 block(8, 8, 1);
    //    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    //    kernel<<< grid, block>>>(dptr, mesh_width, mesh_height, g_fAnim);

    launch_kernel(dptr, mesh_width, mesh_height, g_fAnim);

    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

#ifdef _WIN32
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) fopen_s(&fHandle, filename, mode)
#endif
#else
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) (fHandle = fopen(filename, mode))
#endif
#endif

void sdkDumpBin2(void *data, unsigned int bytes, const char *filename)
{
    printf("sdkDumpBin: <%s>\n", filename);
    FILE *fp;
    FOPEN(fp, filename, "wb");
    fwrite(data, bytes, 1, fp);
    fflush(fp);
    fclose(fp);
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runAutoTest(int devID, char **argv, char *ref_file)
{
    char *reference_file = NULL;
    void *imageData = malloc(mesh_width*mesh_height*sizeof(float));

    // execute the kernel
    launch_kernel((float4 *)d_vbo_buffer, mesh_width, mesh_height, g_fAnim);

    cudaDeviceSynchronize();
    getLastCudaError("launch_kernel failed");

    checkCudaErrors(cudaMemcpy(imageData, d_vbo_buffer, mesh_width*mesh_height*sizeof(float), cudaMemcpyDeviceToHost));

    sdkDumpBin2(imageData, mesh_width*mesh_height*sizeof(float), "simpleGL.bin");
    reference_file = sdkFindFilePath(ref_file, argv[0]);

    if (reference_file &&
        !sdkCompareBin2BinFloat("simpleGL.bin", reference_file,
                                mesh_width*mesh_height*sizeof(float),
                                MAX_EPSILON_ERROR, THRESHOLD, pArgv[0]))
    {
        g_TotalErrors++;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags)
{
    assert(vbo);

    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

    SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

    // unregister this buffer object with CUDA
    checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    sdkStartTimer(&timer);

    // run CUDA kernel to generate vertex positions
    runCuda(&cuda_vbo_resource);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();

    g_fAnim += 0.01f;

    sdkStopTimer(&timer);
    computeFPS();
}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    }
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    if (vbo)
    {
        deleteVBO(&vbo, cuda_vbo_resource);
    }
}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key)
	{
	case (27):
#if defined(__APPLE__) || defined(MACOSX)
		exit(EXIT_SUCCESS);
#else
		glutDestroyWindow(glutGetWindow());
		return;
#endif

	case ('w'):
		if (circle_Y > -1) {
			circle_Y -= 0.015;
		}
		
 			break;
		case ('s'):
			if (circle_Y < 1) {
				circle_Y += 0.015;
				

			}
			if (circle_Y <circle_Y2) {
				circle_Y2 -= 0.015;

			}
			if (circle_Y2 <= -0.7f) {
				printf("player 1 wins");
				cin.get();
				glutDestroyWindow(glutGetWindow());
				
			}
			break;
		case ('a'):
			if (circle_X > -1) {
				circle_X -= 0.015;
			}
			break;
		case ('d'):
			if (circle_X < 1) {
				circle_X += 0.015;
			}
			break;
		case ('t'):
			if (circle_Y2 < 1) {
				circle_Y2 -= 0.015;
			}
			
			break;
		case ('g'):
			if (circle_Y2 > -1) {
				circle_Y2 += 0.015;
			}
			break;
		case ('f'):
			if (circle_X2 < 1) {
				circle_X2 -= 0.015;
			}

			break;
		case ('h'):
			if (circle_X2 > -1) {
				circle_X2 += 0.015;
			}
			break;
    }
	
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4)
    {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

////////////////////////////////////////////////////////////////////////////////
//! Check if the result is correct or write data to file for external
//! regression testing
////////////////////////////////////////////////////////////////////////////////
void checkResultCuda(int argc, char **argv, const GLuint &vbo)
{
    if (!d_vbo_buffer)
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));

        // map buffer object
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        float *data = (float *) glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);

        // check result
        if (checkCmdLineFlag(argc, (const char **) argv, "regression"))
        {
            // write file for regression test
            sdkWriteFile<float>("./data/regression.dat",
                                data, mesh_width * mesh_height * 3, 0.0, false);
        }

        // unmap GL buffer object
        if (!glUnmapBuffer(GL_ARRAY_BUFFER))
        {
            fprintf(stderr, "Unmap buffer failed.\n");
            fflush(stderr);
        }

        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo,
                                                     cudaGraphicsMapFlagsWriteDiscard));

        SDK_CHECK_ERROR_GL();
    }
}
