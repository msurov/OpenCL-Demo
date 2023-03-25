#include <stdio.h>
#include <stdexcept>
#include <stdlib.h>
#include <CL/cl.h>
#include <vector>
#include "time.h"
#include <iostream>
#include "throws.h"
#include "../kernels/kernels.h"

class OCL
{
private:
  std::vector<cl_platform_id> _platforms;
  std::vector<cl_device_id> _devices;
  cl_context _context = NULL;

public:
  OCL(
     cl_device_type devtype = CL_DEVICE_TYPE_DEFAULT // CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR
    )
  {
    cl_int st;
    cl_uint nplatforms = 0;
    st = clGetPlatformIDs(0, NULL, &nplatforms);
    if (st != CL_SUCCESS) throw_runtime_error("clGetPlatformIDs failed");

    _platforms.resize(nplatforms);
    st = clGetPlatformIDs(nplatforms, &_platforms[0], NULL);
    if (st != CL_SUCCESS) throw_runtime_error("clGetPlatformIDs failed");

    cl_uint ndevices = 0;
    st = clGetDeviceIDs(_platforms[0], devtype, 0, NULL, &ndevices);
    if (st != CL_SUCCESS) throw_runtime_error("clGetDeviceIDs failed");

    _devices.resize(ndevices);
    st = clGetDeviceIDs(_platforms[0], CL_DEVICE_TYPE_GPU, ndevices, &_devices[0], NULL);
    if (st != CL_SUCCESS) throw_runtime_error("clGetDeviceIDs failed");

    _context = clCreateContext(NULL, ndevices, &_devices[0], NULL, NULL, &st);
    if (st != CL_SUCCESS) throw_runtime_error("clCreateContext failed");
  }

  OCL(OCL const&) = delete;
  OCL(OCL&&) = delete;

  ~OCL()
  {
    if (_context != NULL)
      clReleaseContext(_context);
  }

  inline cl_context context() const { return _context; }
  inline cl_device_id device() const { return _devices[0]; }
};


class OCLBuf
{
  cl_mem _mem;
  size_t _elemsz;
  size_t _nelems;

public:
  OCLBuf()
  {
    _mem = NULL;
    _elemsz = 0;
  }

  template <typename T>
  OCLBuf(
      OCL const& ocl,
      size_t nelems,
      T* buf = nullptr,
      cl_mem_flags rdwr = CL_MEM_READ_WRITE /// CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY | CL_MEM_READ_WRITE)
    )
  {
    _elemsz = sizeof(T);
    _nelems = nelems;
    cl_int st;
    cl_mem_flags flags = rdwr;
    if (buf) {
      flags |= CL_MEM_COPY_HOST_PTR;
    }
    _mem = clCreateBuffer(ocl.context(), flags, nelems * sizeof(T), buf, &st);
    if (st != CL_SUCCESS) throw_runtime_error("clCreateBuffer failed: ", st);
  }

  ~OCLBuf()
  {
    if (_mem != NULL) {
      cl_int st = clReleaseMemObject(_mem);
      if (st != CL_SUCCESS) throw_runtime_error("clReleaseMemObject failed: ", st);
    }
  }

  inline size_t elemsz() const { return _elemsz; }
  inline size_t nelems() const { return _nelems; }
  inline cl_mem const& clmem() const { return _mem; }
};

class OCLProgram
{
  cl_program _program;
  cl_kernel _kernel;
  cl_device_id _device;
  int _arg_index = -1;

public:
  OCLProgram(OCL const& ocl, std::string const& source_code, char const* name)
  {
    cl_int st = CL_SUCCESS;
    _device = ocl.device();
    size_t sizes[] = {source_code.size()};
    char const* ptrs[] = {&source_code[0]};
    _program = clCreateProgramWithSource(ocl.context(), 1, ptrs, sizes, &st);
    if (st != CL_SUCCESS) throw_runtime_error("clCreateProgramWithSource failed: ", st);
    st = clBuildProgram(_program, 1, &_device, NULL, NULL, NULL);
    if (st != CL_SUCCESS) {
      throw_runtime_error("clBuildProgram failed: ", get_build_status());
    }
    _kernel = clCreateKernel(_program, name, &st);
    if (st != CL_SUCCESS) {
      clReleaseProgram(_program);
      throw_runtime_error("clCreateProgramWithSource failed: ", st);
    }
  }

  OCLProgram(OCL const& ocl, char const* source_code, char const* name)
  {
    cl_int st = CL_SUCCESS;
    _device = ocl.device();
    _program = clCreateProgramWithSource(ocl.context(), 1, &source_code, NULL, &st);
    if (st != CL_SUCCESS) throw_runtime_error("clCreateProgramWithSource failed: ", st);
    st = clBuildProgram(_program, 1, &_device, NULL, NULL, NULL);
    if (st != CL_SUCCESS) {
      throw_runtime_error("clBuildProgram failed: ", get_build_status());
    }
    _kernel = clCreateKernel(_program, name, &st);
    if (st != CL_SUCCESS) {
      clReleaseProgram(_program);
      throw_runtime_error("clCreateProgramWithSource failed: ", st);
    }
  }

  OCLProgram(OCLProgram const&) = delete;
  OCLProgram(OCLProgram&&) = delete;

  ~OCLProgram()
  {
    cl_int st;
    if (_kernel)
      st = clReleaseKernel(_kernel);
    if (st != CL_SUCCESS) throw_runtime_error("clReleaseKernel failed: ", st);
    _kernel = NULL;

    if (_program)
      st = clReleaseProgram(_program);
    if (st != CL_SUCCESS) throw_runtime_error("clReleaseProgram failed: ", st);
    _program = NULL;
  }

  std::string get_build_status() const
  {
    std::string status(1024, '\0');
    size_t sz;
    cl_int st;
    st = clGetProgramBuildInfo(_program, _device, CL_PROGRAM_BUILD_STATUS, status.size(), &status[0], &sz);
    if (st != CL_SUCCESS)
      return "undefined";
    status.resize(sz);
    return status;
  }

  template <typename T>
  void add_arg(T& arg, int arg_index=-1)
  {
    if (arg_index == -1) {
      ++ _arg_index;
      arg_index = _arg_index;
    }
    cl_int st;
    if constexpr (std::is_same_v<T, OCLBuf>)
      st = clSetKernelArg(_kernel, arg_index, sizeof(cl_mem), &arg.clmem());
    else
      st = clSetKernelArg(_kernel, arg_index, sizeof(T), &arg);
    if (st != CL_SUCCESS) throw_runtime_error("clSetKernelArg failed: ", st);
  }

  template <typename ... Args>
  void set_args(Args& ... args)
  {
    _arg_index = -1;
    (add_arg(args), ...);
  }

  inline cl_kernel const& kernel() const { return _kernel; }

};

class OCLTaskQueue
{
private:
  cl_command_queue _queue;

public:
  OCLTaskQueue(OCL const& ocl)
  {
    cl_int st;
    _queue = clCreateCommandQueueWithProperties(ocl.context(), ocl.device(), 0, &st);
    if (st != CL_SUCCESS) throw_runtime_error("clCreateCommandQueueWithProperties failed: ", st);
  }

  OCLTaskQueue(OCLTaskQueue const&) = delete;
  OCLTaskQueue(OCLTaskQueue&&) = delete;

  ~OCLTaskQueue()
  {
    cl_int st = clReleaseCommandQueue(_queue);
    if (st != CL_SUCCESS) throw_runtime_error("clReleaseCommandQueue failed: ", st);
  }

  void write_buf(OCLBuf const& buf, void const* smem, size_t smemsz)
  {
    cl_int st = clEnqueueWriteBuffer(_queue, buf.clmem(), CL_TRUE, 0, smemsz, smem, 0, NULL, NULL);
    if (st != CL_SUCCESS) throw_runtime_error("clEnqueueWriteBuffer failed: ", st);
  }

  void read_buf(OCLBuf const& buf, void* dmem, size_t dmemsz)
  {
    cl_int st = clEnqueueReadBuffer(_queue, buf.clmem(), CL_FALSE, 0, dmemsz, dmem, 0, NULL, NULL);
    if (st != CL_SUCCESS) throw_runtime_error("clEnqueueReadBuffer failed: ", st);
  }

  void add_program(OCLProgram const& prog, size_t datasz)
  {
    /// @todo: set correct local size, gloabl size wth clGetDeviceInfo?
    // clGetDeviceInfo(_device, CL_DEVICE_MAX_WORK_ITEM_SIZES, )
    size_t global_size = datasz;
    size_t local_size = 64;
    cl_int st = clEnqueueNDRangeKernel(_queue, prog.kernel(), 1, NULL, &global_size, &local_size, 0, NULL, NULL);
  }

  void run()
  {
    cl_int st = clFlush(_queue);
    if (st != CL_SUCCESS) throw_runtime_error("clFlush failed: ", st);
    st = clFinish(_queue);
    if (st != CL_SUCCESS) throw_runtime_error("clFinish failed: ", st);
  }

};

int main(int argc, char* argv[])
{
  uint8_t alpha = 5;
  const int vecsz = 1024 * 1024 * 256 * 4;
  std::vector<uint8_t> a_buf(vecsz);
  std::vector<uint8_t> b_buf(vecsz);
  std::vector<uint8_t> c_buf(vecsz);

  for(int i = 0; i < vecsz; ++ i)
  {
    a_buf[i] = i;
    b_buf[i] = 10 + i;
    c_buf[i] = 0;
  }

  auto t1 = epoch_usec();

  OCL ocl;
  OCLBuf a_mem(ocl, vecsz, &a_buf[0], CL_MEM_READ_ONLY);
  OCLBuf b_mem(ocl, vecsz, &b_buf[0], CL_MEM_READ_ONLY);
  OCLBuf c_mem(ocl, vecsz, (uint8_t*)nullptr, CL_MEM_WRITE_ONLY);

  auto t2 = epoch_usec();

  OCLProgram prog(ocl, kernel_addmul, "kernel_addmul");

  auto t3 = epoch_usec();

  prog.set_args(alpha, a_mem, b_mem, c_mem);

  auto t4 = epoch_usec();

  OCLTaskQueue queue(ocl);

  auto t5 = epoch_usec();

  queue.add_program(prog, c_mem.nelems());

  auto t6 = epoch_usec();

  queue.read_buf(c_mem, &c_buf[0], c_buf.size());

  auto t7 = epoch_usec();

  queue.run();

  auto t8 = epoch_usec();

  std::cout << int(c_buf[1]) << std::endl;
  std::cout << "load mem:        " << (t2 - t1) * 1e-3 << std::endl
            << "compile:         " << (t3 - t2) * 1e-3 << std::endl 
            << "set args:        " << (t4 - t3) * 1e-3 << std::endl
            << "init queue:      " << (t5 - t4) * 1e-3 << std::endl
            << "queue push prog: " << (t6 - t5) * 1e-3 << std::endl
            << "queue push read: " << (t7 - t6) * 1e-3 << std::endl
            << "run queue:       " << (t8 - t7) * 1e-3 << std::endl;

  return 0;
}
