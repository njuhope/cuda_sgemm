#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_X 16
#define BLOCK_Y 16

#define TILE_X 128
#define TILE_X_4 32
#define TILE_Y 128
#define TILE_Y_4 32

#define TILE_K 16

#define WPTN 8
#define WPTM 8
#define WPTN_4  2

__global__ void gemm_kernel_NN(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float4* __restrict__ C,
    float alpha, float beta,
    int M, int N, int K)
{
    __shared__ float4 smem_a[2][TILE_K * TILE_Y_4];
    __shared__ float4 smem_b[2][TILE_K * TILE_X_4];

    int tx = threadIdx.x % 16;
    int ty = threadIdx.x / 16;

    int tx4 = threadIdx.x % 4;
    int ty4 = threadIdx.x / 4;

    int tx32 = threadIdx.x % 32;
    int ty32 = threadIdx.x / 32;

    const float* pA = (A + K * TILE_Y * blockIdx.y + ty4 * K + tx4 * 4);
    const float* pB = (B + TILE_X * blockIdx.x + ty32 * N + tx32 * 4);
    float4* pC = C + TILE_Y * blockIdx.y * N / 4 + TILE_X_4 * blockIdx.x;

    int sts_a_offset = tx4 * 4 * TILE_Y + ty4;
    int sts_b_offset = ty32 * TILE_X_4 + tx32;

    float4 f4_zero = make_float4(0.f, 0.f, 0.f, 0.f);
    bool valid_ld_a_0 = ((blockIdx.y * TILE_Y + ty4) < M) && ((tx * 4) < K);
    bool valid_ld_a_1 = ((blockIdx.y * TILE_Y + ty4 + 64) < M) && ((tx * 4) < K); 
    bool valid_ld_b_0 = ((blockIdx.x * TILE_X + tx32 * 4) < N) && (ty32 < K);
    bool valid_ld_b_1 = ((blockIdx.x * TILE_X + tx32 * 4) < N) && ((ty32 + 8) < K);

    float4 ldg_a_reg[2];
    float4 ldg_b_reg[2];

    ldg_a_reg[0] = valid_ld_a_0 ? *(const float4*)pA : f4_zero;
    ldg_a_reg[1] = valid_ld_a_1 ? *(const float4*)(pA + 64 * K) : f4_zero;
    ldg_b_reg[0] = valid_ld_b_0 ? *(const float4*)(pB + 0 * N) : f4_zero;
    ldg_b_reg[1] = valid_ld_b_1 ? *(const float4*)(pB + 8 * N) : f4_zero;

    float4 c[WPTM][WPTN_4] = { { f4_zero } };

    *((float*)&smem_a[0][0] + sts_a_offset + 0 * TILE_Y + 0) = ldg_a_reg[0].x;
    *((float*)&smem_a[0][0] + sts_a_offset + 1 * TILE_Y + 0) = ldg_a_reg[0].y;
    *((float*)&smem_a[0][0] + sts_a_offset + 2 * TILE_Y + 0) = ldg_a_reg[0].z;
    *((float*)&smem_a[0][0] + sts_a_offset + 3 * TILE_Y + 0) = ldg_a_reg[0].w;
    *((float*)&smem_a[0][0] + sts_a_offset + 0 * TILE_Y + 64) = ldg_a_reg[1].x;
    *((float*)&smem_a[0][0] + sts_a_offset + 1 * TILE_Y + 64) = ldg_a_reg[1].y;
    *((float*)&smem_a[0][0] + sts_a_offset + 2 * TILE_Y + 64) = ldg_a_reg[1].z;
    *((float*)&smem_a[0][0] + sts_a_offset + 3 * TILE_Y + 64) = ldg_a_reg[1].w;

    smem_b[0][sts_b_offset + 0] = ldg_b_reg[0];
    smem_b[0][sts_b_offset + 8 * TILE_X_4] = ldg_b_reg[1];

    __syncthreads();

    int i = 0;
    int write_stage_idx = 1;

    float4 reg_a[2][2];
    float4 reg_b[2][2];

    reg_a[0][0] = smem_a[0][0 + ty];
    reg_a[0][1] = smem_a[0][16 + ty];
    reg_b[0][0] = smem_b[0][0 + tx];
    reg_b[0][1] = smem_b[0][16 + tx];

    do
    {
        i += 16;
        valid_ld_a_0 = (valid_ld_a_0 && (i < K));
        valid_ld_a_1 = (valid_ld_a_1 && (i < K));
        valid_ld_b_0 = (valid_ld_b_0 && (i < K));
        valid_ld_b_1 = (valid_ld_b_1 && (i < K));

        ldg_a_reg[0] = (valid_ld_a_0) ? *(const float4*)(pA + i + 0) : f4_zero;
        ldg_a_reg[1] = (valid_ld_a_1) ? *(const float4*)(pA + i + 64 * K) : f4_zero;
        ldg_b_reg[0] = (valid_ld_b_0) ? *(const float4*)(pB + (i + 0) * N) : f4_zero;
        ldg_b_reg[1] = (valid_ld_b_1) ? *(const float4*)(pB + (i + 8) * N) : f4_zero;

        int load_stage_idx = write_stage_idx ^ 1;

#pragma unroll
        for (int j = 0; j < TILE_K - 1; j++)
        {
            reg_a[(j + 1) % 2][0] = smem_a[load_stage_idx][(j + 1) *  TILE_Y_4 + 0 + ty];
            reg_a[(j + 1) % 2][1] = smem_a[load_stage_idx][(j + 1) *  TILE_Y_4 + 16 + ty];
            reg_b[(j + 1) % 2][0] = smem_b[load_stage_idx][(j + 1) *  TILE_X_4 + 0 + tx];
            reg_b[(j + 1) % 2][1] = smem_b[load_stage_idx][(j + 1) *  TILE_X_4 + 16 + tx];
            c[0][0].x += reg_a[j % 2][0].x * reg_b[j % 2][0].x;
            c[0][0].y += reg_a[j % 2][0].x * reg_b[j % 2][0].y;
            c[0][0].z += reg_a[j % 2][0].x * reg_b[j % 2][0].z;
            c[0][0].w += reg_a[j % 2][0].x * reg_b[j % 2][0].w;
            c[0][1].x += reg_a[j % 2][0].x * reg_b[j % 2][1].x;
            c[0][1].y += reg_a[j % 2][0].x * reg_b[j % 2][1].y;
            c[0][1].z += reg_a[j % 2][0].x * reg_b[j % 2][1].z;
            c[0][1].w += reg_a[j % 2][0].x * reg_b[j % 2][1].w;
            c[1][0].x += reg_a[j % 2][0].y * reg_b[j % 2][0].x;
            c[1][0].y += reg_a[j % 2][0].y * reg_b[j % 2][0].y;
            c[1][0].z += reg_a[j % 2][0].y * reg_b[j % 2][0].z;
            c[1][0].w += reg_a[j % 2][0].y * reg_b[j % 2][0].w;
            c[1][1].x += reg_a[j % 2][0].y * reg_b[j % 2][1].x;
            c[1][1].y += reg_a[j % 2][0].y * reg_b[j % 2][1].y;
            c[1][1].z += reg_a[j % 2][0].y * reg_b[j % 2][1].z;
            c[1][1].w += reg_a[j % 2][0].y * reg_b[j % 2][1].w;
            c[2][0].x += reg_a[j % 2][0].z * reg_b[j % 2][0].x;
            c[2][0].y += reg_a[j % 2][0].z * reg_b[j % 2][0].y;
            c[2][0].z += reg_a[j % 2][0].z * reg_b[j % 2][0].z;
            c[2][0].w += reg_a[j % 2][0].z * reg_b[j % 2][0].w;
            c[2][1].x += reg_a[j % 2][0].z * reg_b[j % 2][1].x;
            c[2][1].y += reg_a[j % 2][0].z * reg_b[j % 2][1].y;
            c[2][1].z += reg_a[j % 2][0].z * reg_b[j % 2][1].z;
            c[2][1].w += reg_a[j % 2][0].z * reg_b[j % 2][1].w;
            c[3][0].x += reg_a[j % 2][0].w * reg_b[j % 2][0].x;
            c[3][0].y += reg_a[j % 2][0].w * reg_b[j % 2][0].y;
            c[3][0].z += reg_a[j % 2][0].w * reg_b[j % 2][0].z;
            c[3][0].w += reg_a[j % 2][0].w * reg_b[j % 2][0].w;
            c[3][1].x += reg_a[j % 2][0].w * reg_b[j % 2][1].x;
            c[3][1].y += reg_a[j % 2][0].w * reg_b[j % 2][1].y;
            c[3][1].z += reg_a[j % 2][0].w * reg_b[j % 2][1].z;
            c[3][1].w += reg_a[j % 2][0].w * reg_b[j % 2][1].w;
            c[4][0].x += reg_a[j % 2][1].x * reg_b[j % 2][0].x;
            c[4][0].y += reg_a[j % 2][1].x * reg_b[j % 2][0].y;
            c[4][0].z += reg_a[j % 2][1].x * reg_b[j % 2][0].z;
            c[4][0].w += reg_a[j % 2][1].x * reg_b[j % 2][0].w;
            c[4][1].x += reg_a[j % 2][1].x * reg_b[j % 2][1].x;
            c[4][1].y += reg_a[j % 2][1].x * reg_b[j % 2][1].y;
            c[4][1].z += reg_a[j % 2][1].x * reg_b[j % 2][1].z;
            c[4][1].w += reg_a[j % 2][1].x * reg_b[j % 2][1].w;
            c[5][0].x += reg_a[j % 2][1].y * reg_b[j % 2][0].x;
            c[5][0].y += reg_a[j % 2][1].y * reg_b[j % 2][0].y;
            c[5][0].z += reg_a[j % 2][1].y * reg_b[j % 2][0].z;
            c[5][0].w += reg_a[j % 2][1].y * reg_b[j % 2][0].w;
            c[5][1].x += reg_a[j % 2][1].y * reg_b[j % 2][1].x;
            c[5][1].y += reg_a[j % 2][1].y * reg_b[j % 2][1].y;
            c[5][1].z += reg_a[j % 2][1].y * reg_b[j % 2][1].z;
            c[5][1].w += reg_a[j % 2][1].y * reg_b[j % 2][1].w;
            c[6][0].x += reg_a[j % 2][1].z * reg_b[j % 2][0].x;
            c[6][0].y += reg_a[j % 2][1].z * reg_b[j % 2][0].y;
            c[6][0].z += reg_a[j % 2][1].z * reg_b[j % 2][0].z;
            c[6][0].w += reg_a[j % 2][1].z * reg_b[j % 2][0].w;
            c[6][1].x += reg_a[j % 2][1].z * reg_b[j % 2][1].x;
            c[6][1].y += reg_a[j % 2][1].z * reg_b[j % 2][1].y;
            c[6][1].z += reg_a[j % 2][1].z * reg_b[j % 2][1].z;
            c[6][1].w += reg_a[j % 2][1].z * reg_b[j % 2][1].w;
            c[7][0].x += reg_a[j % 2][1].w * reg_b[j % 2][0].x;
            c[7][0].y += reg_a[j % 2][1].w * reg_b[j % 2][0].y;
            c[7][0].z += reg_a[j % 2][1].w * reg_b[j % 2][0].z;
            c[7][0].w += reg_a[j % 2][1].w * reg_b[j % 2][0].w;
            c[7][1].x += reg_a[j % 2][1].w * reg_b[j % 2][1].x;
            c[7][1].y += reg_a[j % 2][1].w * reg_b[j % 2][1].y;
            c[7][1].z += reg_a[j % 2][1].w * reg_b[j % 2][1].z;
            c[7][1].w += reg_a[j % 2][1].w * reg_b[j % 2][1].w;
        }
        
        if(i < K) {
            *((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 0 * TILE_Y + 0) = ldg_a_reg[0].x;
            *((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 1 * TILE_Y + 0) = ldg_a_reg[0].y;
            *((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 2 * TILE_Y + 0) = ldg_a_reg[0].z;
            *((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 3 * TILE_Y + 0) = ldg_a_reg[0].w;
            *((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 0 * TILE_Y + 64) = ldg_a_reg[1].x;
            *((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 1 * TILE_Y + 64) = ldg_a_reg[1].y;
            *((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 2 * TILE_Y + 64) = ldg_a_reg[1].z;
            *((float*)&smem_a[write_stage_idx][0] + sts_a_offset + 3 * TILE_Y + 64) = ldg_a_reg[1].w;

            smem_b[write_stage_idx][sts_b_offset + 0] = ldg_b_reg[0];
            smem_b[write_stage_idx][sts_b_offset + 8 * TILE_X_4] = ldg_b_reg[1];
            __syncthreads();
            write_stage_idx ^= 1;
        }

        reg_a[0][0] = smem_a[load_stage_idx ^ 1][0 + ty];
        reg_a[0][1] = smem_a[load_stage_idx ^ 1][16 + ty];
        reg_b[0][0] = smem_b[load_stage_idx ^ 1][0 + tx];
        reg_b[0][1] = smem_b[load_stage_idx ^ 1][16 + tx];

        c[0][0].x += reg_a[1][0].x * reg_b[1][0].x;
        c[0][0].y += reg_a[1][0].x * reg_b[1][0].y;
        c[0][0].z += reg_a[1][0].x * reg_b[1][0].z;
        c[0][0].w += reg_a[1][0].x * reg_b[1][0].w;
        c[0][1].x += reg_a[1][0].x * reg_b[1][1].x;
        c[0][1].y += reg_a[1][0].x * reg_b[1][1].y;
        c[0][1].z += reg_a[1][0].x * reg_b[1][1].z;
        c[0][1].w += reg_a[1][0].x * reg_b[1][1].w;
        c[1][0].x += reg_a[1][0].y * reg_b[1][0].x;
        c[1][0].y += reg_a[1][0].y * reg_b[1][0].y;
        c[1][0].z += reg_a[1][0].y * reg_b[1][0].z;
        c[1][0].w += reg_a[1][0].y * reg_b[1][0].w;
        c[1][1].x += reg_a[1][0].y * reg_b[1][1].x;
        c[1][1].y += reg_a[1][0].y * reg_b[1][1].y;
        c[1][1].z += reg_a[1][0].y * reg_b[1][1].z;
        c[1][1].w += reg_a[1][0].y * reg_b[1][1].w;
        c[2][0].x += reg_a[1][0].z * reg_b[1][0].x;
        c[2][0].y += reg_a[1][0].z * reg_b[1][0].y;
        c[2][0].z += reg_a[1][0].z * reg_b[1][0].z;
        c[2][0].w += reg_a[1][0].z * reg_b[1][0].w;
        c[2][1].x += reg_a[1][0].z * reg_b[1][1].x;
        c[2][1].y += reg_a[1][0].z * reg_b[1][1].y;
        c[2][1].z += reg_a[1][0].z * reg_b[1][1].z;
        c[2][1].w += reg_a[1][0].z * reg_b[1][1].w;
        c[3][0].x += reg_a[1][0].w * reg_b[1][0].x;
        c[3][0].y += reg_a[1][0].w * reg_b[1][0].y;
        c[3][0].z += reg_a[1][0].w * reg_b[1][0].z;
        c[3][0].w += reg_a[1][0].w * reg_b[1][0].w;
        c[3][1].x += reg_a[1][0].w * reg_b[1][1].x;
        c[3][1].y += reg_a[1][0].w * reg_b[1][1].y;
        c[3][1].z += reg_a[1][0].w * reg_b[1][1].z;
        c[3][1].w += reg_a[1][0].w * reg_b[1][1].w;
        c[4][0].x += reg_a[1][1].x * reg_b[1][0].x;
        c[4][0].y += reg_a[1][1].x * reg_b[1][0].y;
        c[4][0].z += reg_a[1][1].x * reg_b[1][0].z;
        c[4][0].w += reg_a[1][1].x * reg_b[1][0].w;
        c[4][1].x += reg_a[1][1].x * reg_b[1][1].x;
        c[4][1].y += reg_a[1][1].x * reg_b[1][1].y;
        c[4][1].z += reg_a[1][1].x * reg_b[1][1].z;
        c[4][1].w += reg_a[1][1].x * reg_b[1][1].w;
        c[5][0].x += reg_a[1][1].y * reg_b[1][0].x;
        c[5][0].y += reg_a[1][1].y * reg_b[1][0].y;
        c[5][0].z += reg_a[1][1].y * reg_b[1][0].z;
        c[5][0].w += reg_a[1][1].y * reg_b[1][0].w;
        c[5][1].x += reg_a[1][1].y * reg_b[1][1].x;
        c[5][1].y += reg_a[1][1].y * reg_b[1][1].y;
        c[5][1].z += reg_a[1][1].y * reg_b[1][1].z;
        c[5][1].w += reg_a[1][1].y * reg_b[1][1].w;
        c[6][0].x += reg_a[1][1].z * reg_b[1][0].x;
        c[6][0].y += reg_a[1][1].z * reg_b[1][0].y;
        c[6][0].z += reg_a[1][1].z * reg_b[1][0].z;
        c[6][0].w += reg_a[1][1].z * reg_b[1][0].w;
        c[6][1].x += reg_a[1][1].z * reg_b[1][1].x;
        c[6][1].y += reg_a[1][1].z * reg_b[1][1].y;
        c[6][1].z += reg_a[1][1].z * reg_b[1][1].z;
        c[6][1].w += reg_a[1][1].z * reg_b[1][1].w;
        c[7][0].x += reg_a[1][1].w * reg_b[1][0].x;
        c[7][0].y += reg_a[1][1].w * reg_b[1][0].y;
        c[7][0].z += reg_a[1][1].w * reg_b[1][0].z;
        c[7][0].w += reg_a[1][1].w * reg_b[1][0].w;
        c[7][1].x += reg_a[1][1].w * reg_b[1][1].x;
        c[7][1].y += reg_a[1][1].w * reg_b[1][1].y;
        c[7][1].z += reg_a[1][1].w * reg_b[1][1].z;
        c[7][1].w += reg_a[1][1].w * reg_b[1][1].w;
        
    } while (i < K);

#pragma unroll
    for (int wm = 0; wm < WPTM; wm++){
#pragma unroll
        for (int wn = 0; wn < WPTN_4; wn++){
            c[wm][wn].x *= alpha;
            c[wm][wn].y *= alpha;
            c[wm][wn].z *= alpha;
            c[wm][wn].w *= alpha;
        }
    }

#pragma unroll
    for (int wm = 0; wm < 4; wm++){
#pragma unroll
        for (int wn = 0; wn < WPTN_4; wn++){
            if (((blockIdx.y * TILE_Y + ty * 4 + wm) < M) 
                && ((blockIdx.x * TILE_X + wn * 64 + tx * 4) < N)) {
                if (beta != 0) {
                    float4 vec4c = *(pC + ((ty * 4 + wm) * N / 4 + wn * 16 + tx));
                    vec4c.x = vec4c.x * beta + c[wm][wn].x;
                    vec4c.y = vec4c.y * beta + c[wm][wn].y;
                    vec4c.z = vec4c.z * beta + c[wm][wn].z;
                    vec4c.w = vec4c.w * beta + c[wm][wn].w;
                    *(pC + (ty * 4 + wm) * N / 4 + wn * 16 + tx) = vec4c;
                } else {
                    *(pC + (ty * 4 + wm) * N / 4 + wn * 16 + tx) = c[wm][wn];
                }
            }
        }
    }

#pragma unroll
    for (int wm = 0; wm < 4; wm++){
#pragma unroll
        for (int wn = 0; wn < WPTN_4; wn++){
            if (((blockIdx.y * TILE_Y + 64 + ty * 4 + wm) < M) 
                && ((blockIdx.x * TILE_X + wn * 64 + tx * 4) < N)) {
                if (beta != 0) {
                    float4 vec4c = *(pC + ((64 + ty * 4 + wm) * N / 4 + wn * 16 + tx));
                    vec4c.x = vec4c.x * beta + c[wm + 4][wn].x;
                    vec4c.y = vec4c.y * beta + c[wm + 4][wn].y;
                    vec4c.z = vec4c.z * beta + c[wm + 4][wn].z;
                    vec4c.w = vec4c.w * beta + c[wm + 4][wn].w;
                    *(pC + (64 + ty * 4 + wm) * N / 4 + wn * 16 + tx) = vec4c;
                } else {
                    *(pC + (64 + ty * 4 + wm) * N / 4 + wn * 16 + tx) = c[wm + 4][wn];
                }
            }
        }
    }
}
