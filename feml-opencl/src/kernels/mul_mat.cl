//------------------------------------------------------------------------------
// mul_mat: naive f32 matrix multiply (ggml semantics)
// a: [K, M], b: [K, N] -> dst: [M, N]; dst[m, n] = sum_k a[k, m] * b[k, n]
//------------------------------------------------------------------------------
kernel void kernel_mul_mat(
    global const char *a,
    ulong oa,
    ulong na1,
    global const char *b,
    ulong ob,
    ulong nb1,
    global char *c,
    ulong oc,
    ulong nc1,
    int K,
    int M,
    int N)
{
    int idx = get_global_id(0);
    if (idx >= M * N) {
        return;
    }
    int m = idx / N;
    int n = idx % N;

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        float x = *((const global float *)(a + oa + k * 4 + m * na1));
        float y = *((const global float *)(b + ob + k * 4 + n * nb1));
        acc += x * y;
    }
    *((global float *)(c + oc + m * 4 + n * nc1)) = acc;
}
