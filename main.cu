#include <cstdio>
#include <cublas_v2.h>
#include "helper.cuh"
// #define IDX2C(i,j,ld) (((j)*(ld))+(i))

const int M = 1024 * 8 ;
const int K = 1024 * 8;
const int N = 1024 * 8;
#define rM 32
#define rN 32
#define rK 32
__device__ unsigned randomSeed = 1919810;
__global__ void genRandomMatrix(float *A,int size){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    A[tid] = tid % 3  ;//(randomSeed * 114 + 514) % 3777;
    randomSeed = A[tid];
}

__global__ void cmp(float *A,float *B){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(A[tid] != B[tid]){
        printf("%d,%f,%f\n",tid,A[tid],B[tid]);
    }
}
__global__ void SGEMM1(int M,int K,int N,float *A,float *B,float *C,float alpha, float beta){
    int bx = blockIdx.x;int by = blockIdx.y;
    int tx = threadIdx.x;int ty = threadIdx.y;
    int col = blockDim.x * bx + tx;
    int row = blockDim.y * by + ty;
    float tmp = 0;
    for(int i = 0; i < K; i++){
        tmp += A[row * K + i] * B[i * N + col];

    }
    C[row * N + col] = tmp;
}
__global__ void SGEMM3(int M,int K,int N,float *A,float *B,float *C,float alpha, float beta){
    int bx = blockIdx.x;int by = blockIdx.y;
    int tx = threadIdx.x;int ty = threadIdx.y;
    int col = blockDim.x * bx + tx;
    int row = blockDim.y * by + ty;
    float tmp = 0;
    __shared__ float buffA[rM * rN];
    __shared__ float buffB[rM * rN];
    for(int i = 0; i < K; i+=rK){
        buffA[ty * blockDim.x + tx] = A[row * K + tx + i];
        buffB[ty * blockDim.x + tx] = B[ty * N + col + i * N];
        __syncthreads();
        for(int j = 0; j < rK;j++){
            tmp += buffA[ty * rN + j] * buffB[j * rN + tx ];
        }
        __syncthreads();
    }
    C[row * N + col] = tmp;
}
__global__ void SGEMM2(int M,int K,int N,float *A,float *B,float *C,float alpha, float beta){
    int bx = blockIdx.x;int by = blockIdx.y;
    int tx = threadIdx.x;int ty = threadIdx.y;
    int col = blockDim.x * bx + tx;
    int row = blockDim.y * by + ty;
    float tmp = 0;
    __shared__ float buffA[rM * rN];
    __shared__ float buffB[rM * rN];
    for(int i = 0; i < K; i+=rK){
        buffA[ty * blockDim.x + tx] = A[row * K + tx + i];
        buffB[tx * blockDim.x + ty] = B[ty * N + col + i * N];
        __syncthreads();
        for(int j = 0; j < rK;j++){
            tmp += buffA[ty * rN + j] * buffB[tx * rN + j ];
        }
        __syncthreads();
    }
    C[row * N + col] = tmp;
}
__global__ void SGEMM4(int M,int K,int N,float *A,float *B,float *C,float alpha, float beta){
    int bx = blockIdx.x;int by = blockIdx.y;
    int tx = threadIdx.x;int ty = threadIdx.y;
    int col = blockDim.x * bx + tx;
    int row = blockDim.y * by + ty;
    float tmp[4] = {0., 0., 0., 0.};
    __shared__ float buffA[rM * rN];
    __shared__ float buffB[rM * rN];
    for(int i = 0; i < K; i+=rK){
        buffA[ty * blockDim.x + tx] = A[row * K + tx + i];
        buffA[ty * blockDim.x + tx + 1] = A[row * K + tx + 1 + i];
        buffA[ty * blockDim.x + tx + 2] = A[row * K + tx + 2 + i];
        buffA[ty * blockDim.x + tx + 3] = A[row * K + tx + 3 + i];

        buffB[ty * blockDim.x + tx] = B[ty * N + col + i * N];
        buffB[ty * blockDim.x + tx + 1] = B[ty * N + col + 1 + i * N];
        buffB[ty * blockDim.x + tx + 2] = B[ty * N + col + 2 + i * N];
        buffB[ty * blockDim.x + tx + 3] = B[ty * N + col + 3 + i * N];

        __syncthreads();
        #pragma unroll
        for(int j = 0; j < rK;j++){
            tmp[0] += buffA[ty * rN + j] * buffB[j * rN + tx ];
            tmp[1] += buffA[ty * rN + j + 1] * buffB[j * rN + tx + 1];
            tmp[2] += buffA[ty * rN + j + 2] * buffB[j * rN + tx + 2];
            tmp[3] += buffA[ty * rN + j + 3] * buffB[j * rN + tx + 3];
        }
        __syncthreads();
    }
    C[row * N + col] = tmp[0];
    C[row * N + col + 1] = tmp[1];
    C[row * N + col + 2] = tmp[2];
    C[row * N + col + 3] = tmp[3];
}
__global__ void print(float *A){
    for(int i = 0; i < 32;i++){
        printf("%.2f ",A[i]);
    }printf("\n");
}
int main(){
    float *A,*B,*C,*D;
    float alpha = 1,beta = 0;
    cudaMalloc(&A,M * K * sizeof(float));
    cudaMalloc(&B,K * N * sizeof(float));
    cudaMalloc(&C,M * N * sizeof(float));
    cudaMalloc(&D,M * N * sizeof(float));
    genRandomMatrix<<<M*K,1>>>(A,M * K);genRandomMatrix<<<N*K,1>>>(B,N * K);
    cublasHandle_t handle;cublasCreate(&handle);
    double start = cpuSecond();
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N, M, K,&alpha,B,N,A,K,&beta,C,N);
    cudaDeviceSynchronize();
    printf("Times :%f\n",cpuSecond() - start);
    dim3 grid(N/rN,M/rM);
    dim3 block(rN/4,rM);
    start = cpuSecond();
    SGEMM4<<<grid,block>>>(M,K,N,A,B,D,alpha,beta);
    cudaDeviceSynchronize();
    printf("Times :%f\n",cpuSecond() - start);

    // print<<<1,1>>>(D);
    cmp<<<M*N,1>>>(C,D);
    cudaDeviceSynchronize();
}