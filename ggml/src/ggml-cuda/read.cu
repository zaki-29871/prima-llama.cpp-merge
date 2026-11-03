#include "read.cuh"

// read every element, the result is written only if the sum is -inf, which keeps the loads
static __global__ void read_f32(const float * __restrict__ x, float * __restrict__ dst, const int64_t n) {
    float sum = 0.0f;
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x*blockDim.x) {
        sum += x[i];
    }
    if (sum == -INFINITY) {
        dst[0] = sum;
    }
}

void ggml_cuda_read(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t n = ggml_nelements(src0);

    cudaStream_t stream = ctx.stream();

    CUDA_CHECK(cudaMemsetAsync(dst->data, 0, sizeof(float), stream));

    const int64_t num_blocks = std::min<int64_t>((n + CUDA_READ_BLOCK_SIZE - 1)/CUDA_READ_BLOCK_SIZE, 65535);
    if (num_blocks > 0) {
        read_f32<<<num_blocks, CUDA_READ_BLOCK_SIZE, 0, stream>>>((const float *) src0->data, (float *) dst->data, n);
    }
}
