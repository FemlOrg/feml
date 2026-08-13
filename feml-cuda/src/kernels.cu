//------------------------------------------------------------------------------
// feml-cuda kernel sources (compiled at runtime via NVRTC).
//
// Layout semantics are ggml-compatible: buffers are opaque byte ranges, each
// kernel receives the base device pointer plus a byte offset (scalar) and the
// byte strides of each operand. Sources may fully alias the destination
// (inplace ops); every thread reads its inputs strictly before writing its
// output element.
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// mul / add: strided elementwise binary op with 4D broadcasting (ggml rules).
// Grid = (ne1, ne2, ne3); each thread block covers one (i1, i2, i3) line.
//------------------------------------------------------------------------------
extern "C" __global__ void kernel_mul(
    const unsigned char *src0, unsigned long long offset0,
    const unsigned char *src1, unsigned long long offset1,
    unsigned char *dst, unsigned long long offsetd,
    int ne00, int ne01, int ne02, int ne03,
    unsigned long long nb00, unsigned long long nb01,
    unsigned long long nb02, unsigned long long nb03,
    int ne10, int ne11, int ne12, int ne13,
    unsigned long long nb10, unsigned long long nb11,
    unsigned long long nb12, unsigned long long nb13,
    int ne0, int ne1, int ne2, int ne3,
    unsigned long long nb0, unsigned long long nb1,
    unsigned long long nb2, unsigned long long nb3)
{
    const int i03 = blockIdx.z;
    const int i02 = blockIdx.y;
    const int i01 = blockIdx.x;

    // ggml-style broadcast: index each source with the block index modulo
    // that source's extent in each dim (dim == 1 degrades to 0).
    const int i03a = i03 % ne03;
    const int i02a = i02 % ne02;
    const int i01a = i01 % ne01;
    const int i13 = i03 % ne13;
    const int i12 = i02 % ne12;
    const int i11 = i01 % ne11;

    const unsigned char *src0_ptr = src0 + offset0 + i03a * nb03 + i02a * nb02 + i01a * nb01;
    const unsigned char *src1_ptr = src1 + offset1 + i13 * nb13 + i12 * nb12 + i11 * nb11;
    unsigned char *dst_ptr = dst + offsetd + i03 * nb3 + i02 * nb2 + i01 * nb1;

    for (int i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        const int i10 = i0 % ne10;
        const int i00 = i0 % ne00;
        *((float *)(dst_ptr + i0 * nb0)) =
            *((const float *)(src0_ptr + i00 * nb00)) *
            *((const float *)(src1_ptr + i10 * nb10));
    }
}

extern "C" __global__ void kernel_add(
    const unsigned char *src0, unsigned long long offset0,
    const unsigned char *src1, unsigned long long offset1,
    unsigned char *dst, unsigned long long offsetd,
    int ne00, int ne01, int ne02, int ne03,
    unsigned long long nb00, unsigned long long nb01,
    unsigned long long nb02, unsigned long long nb03,
    int ne10, int ne11, int ne12, int ne13,
    unsigned long long nb10, unsigned long long nb11,
    unsigned long long nb12, unsigned long long nb13,
    int ne0, int ne1, int ne2, int ne3,
    unsigned long long nb0, unsigned long long nb1,
    unsigned long long nb2, unsigned long long nb3)
{
    const int i03 = blockIdx.z;
    const int i02 = blockIdx.y;
    const int i01 = blockIdx.x;

    // ggml-style broadcast: index each source with the block index modulo
    // that source's extent in each dim (dim == 1 degrades to 0).
    const int i03a = i03 % ne03;
    const int i02a = i02 % ne02;
    const int i01a = i01 % ne01;
    const int i13 = i03 % ne13;
    const int i12 = i02 % ne12;
    const int i11 = i01 % ne11;

    const unsigned char *src0_ptr = src0 + offset0 + i03a * nb03 + i02a * nb02 + i01a * nb01;
    const unsigned char *src1_ptr = src1 + offset1 + i13 * nb13 + i12 * nb12 + i11 * nb11;
    unsigned char *dst_ptr = dst + offsetd + i03 * nb3 + i02 * nb2 + i01 * nb1;

    for (int i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        const int i10 = i0 % ne10;
        const int i00 = i0 % ne00;
        *((float *)(dst_ptr + i0 * nb0)) =
            *((const float *)(src0_ptr + i00 * nb00)) +
            *((const float *)(src1_ptr + i10 * nb10));
    }
}

//------------------------------------------------------------------------------
// mul_mat: naive f32 matrix multiply (ggml semantics).
// a: [K, M], b: [K, N] -> dst: [M, N]; dst[m, n] = sum_k a[k, m] * b[k, n]
//------------------------------------------------------------------------------
extern "C" __global__ void kernel_mul_mat(
    const unsigned char *a, unsigned long long oa, unsigned long long na1,
    const unsigned char *b, unsigned long long ob, unsigned long long nb1,
    unsigned char *c, unsigned long long oc, unsigned long long nc1,
    int K, int M, int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) {
        return;
    }
    const int m = idx / N;
    const int n = idx % N;

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        const float x = *((const float *)(a + oa + k * 4 + m * na1));
        const float y = *((const float *)(b + ob + k * 4 + n * nb1));
        acc += x * y;
    }
    *((float *)(c + oc + m * 4 + n * nc1)) = acc;
}

//------------------------------------------------------------------------------
// fill: byte pattern fill of a range within a buffer.
//------------------------------------------------------------------------------
extern "C" __global__ void kernel_fill(
    unsigned char *dst, unsigned long long offset,
    unsigned char value, unsigned long long len)
{
    const unsigned long long i =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        dst[offset + i] = value;
    }
}
