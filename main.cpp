#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <vector>
#include <chrono>

// OpenCL kernel which is run for every work item created.
const char *saxpy_kernel =
"__kernel                                   \n"
"void saxpy_kernel(float alpha,     \n"
"                  __global float *A,       \n"
"                  __global float *B,       \n"
"                  __global float *C)       \n"
"{                                          \n"
"    //Get the index of the work-item       \n"
"    int index = get_global_id(0);          \n"
"    C[index] = alpha* A[index] + B[index]; \n"
"}                                          \n";

int main(int argc, char* argv[])
{
  float alpha = 2.35f;
  const int vecsz = 1024 * 1024;
  std::vector<float> Abuf(vecsz);
  std::vector<float> Bbuf(vecsz);
  std::vector<float> Cbuf(vecsz);

  float *A = &Abuf[0];
  float *B = &Bbuf[0];
  float *C = &Cbuf[0];

  for(int i = 0; i < vecsz; ++ i)
  {
    A[i] = i;
    B[i] = vecsz - i;
    C[i] = 0;
  }

  // Get platform and device information

  // Set up the Platform
  cl_uint num_platforms;
  cl_int st = clGetPlatformIDs(0, NULL, &num_platforms);

  std::vector<cl_platform_id> platforms(num_platforms);
  st = clGetPlatformIDs(num_platforms, &platforms[0], NULL);

  // Get the devices list and choose the device you want to run on
  cl_uint num_devices;
  st = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0,NULL, &num_devices);

  std::vector<cl_device_id> device_list(num_devices);
  st = clGetDeviceIDs(platforms[0],CL_DEVICE_TYPE_GPU, num_devices, &device_list[0], NULL);

  // Create one OpenCL context for each device in the platform
  cl_context context;
  context = clCreateContext(NULL, num_devices, &device_list[0], NULL, NULL, &st);

  // Create a command queue
  cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_list[0], 0, &st);

  // Create memory buffers on the device for each vector
  cl_mem A_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, vecsz * sizeof(float), NULL, &st);
  cl_mem B_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, vecsz * sizeof(float), NULL, &st);
  cl_mem C_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, vecsz * sizeof(float), NULL, &st);

  // Copy the Buffer A and B to the device
  st = clEnqueueWriteBuffer(command_queue, A_clmem, CL_TRUE, 0, vecsz * sizeof(float), A, 0, NULL, NULL);
  st = clEnqueueWriteBuffer(command_queue, B_clmem, CL_TRUE, 0, vecsz * sizeof(float), B, 0, NULL, NULL);

  // Create a program from the kernel source
  cl_program program = clCreateProgramWithSource(context, 1, (const char**)&saxpy_kernel, NULL, &st);

  // Build the program
  st = clBuildProgram(program, 1, &device_list[0], NULL, NULL, NULL);

  // Create the OpenCL kernel
  cl_kernel kernel = clCreateKernel(program, "saxpy_kernel", &st);

  // Set the arguments of the kernel
  st = clSetKernelArg(kernel, 0, sizeof(float), (void *)&alpha);
  st = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&A_clmem);
  st = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&B_clmem);
  st = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&C_clmem);

  // Execute the OpenCL kernel on the list
  size_t global_size = vecsz; // Process the entire lists
  size_t local_size = 64;           // Process one item at a time
  st = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

  // Read the cl memory C_clmem on device to the host variable C
  st = clEnqueueReadBuffer(command_queue, C_clmem, CL_TRUE, 0, vecsz * sizeof(float), C, 0, NULL, NULL);

  // Clean up and wait for all the comands to complete.
  st = clFlush(command_queue);
  st = clFinish(command_queue);

  // Display the result to the screen
  for(int i = 0; i < vecsz; i++)
    printf("%f * %f + %f = %f\n", alpha, A[i], B[i], C[i]);

  // Finally release all OpenCL allocated objects and host buffers.
  st = clReleaseKernel(kernel);
  st = clReleaseProgram(program);
  st = clReleaseMemObject(A_clmem);
  st = clReleaseMemObject(B_clmem);
  st = clReleaseMemObject(C_clmem);
  st = clReleaseCommandQueue(command_queue);
  st = clReleaseContext(context);

  return 0;
}