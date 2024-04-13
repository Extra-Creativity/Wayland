#include "Conversion.h"
#include "thrust/device_ptr.h"
#include "thrust/transform.h"
#include "thrust/device_vector.h"

std::vector<unsigned char> FromFloatToChar(float * src0, std::size_t size)
{
    thrust::device_vector<unsigned char> resultBuffer(size);
    std::vector<unsigned char> cpuBuffer(size);
    thrust::device_ptr<float> src{src0};

    thrust::transform(src, src + size, resultBuffer.begin(),
                      [] __host__ __device__ (float val) {
                          if (val > 1.0f)
                              val = 1.0f;
                          return (unsigned char)(val * 255);
                      });
    thrust::copy(resultBuffer.begin(), resultBuffer.end(), cpuBuffer.begin());
    return cpuBuffer;
}