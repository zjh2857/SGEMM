#include <cstdio>
#include <cublas_v2.h>
#include "helper.cuh"
// #define IDX2C(i,j,ld) (((j)*(ld))+(i))

const int M = 1024 * 8;
const int K = 1024 * 8;
const int N = 1024 * 8;
#define rM 32
#define rN 32
#define rK 32
#define lM 4
#define lN 4
__device__ unsigned randomSeed = 1919810;
__global__ void genRandomMatrix(float *A,int size){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    A[tid] = tid % 3;//(randomSeed * 114 + 514) % 3777;
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
    int row = 4 * blockDim.y * by + ty * 4;
    // printf("%d,%d,%d,%d\n",row,col,by,blockDim.y);
    float tmp[4] = {0., 0., 0., 0.};
    __shared__ float buffA[rM * rN];
    __shared__ float buffB[rM * rN];
    for(int i = 0; i < K; i+=rK){
        buffA[(ty * 4 + 0) * blockDim.x + tx] = A[(row + 0) * K + tx + i];
        buffA[(ty * 4 + 1) * blockDim.x + tx] = A[(row + 1) * K + tx + i];
        buffA[(ty * 4 + 2) * blockDim.x + tx] = A[(row + 2) * K + tx + i];
        buffA[(ty * 4 + 3) * blockDim.x + tx] = A[(row + 3) * K + tx + i];

        buffB[(ty * 4 + 0) * blockDim.x + tx] = B[(ty * 4 + 0) * N + col + i * N];
        buffB[(ty * 4 + 1) * blockDim.x + tx] = B[(ty * 4 + 1) * N + col + i * N];
        buffB[(ty * 4 + 2) * blockDim.x + tx] = B[(ty * 4 + 2) * N + col + i * N];
        buffB[(ty * 4 + 3) * blockDim.x + tx] = B[(ty * 4 + 3) * N + col + i * N];

        __syncthreads();
        #pragma unroll
        for(int j = 0; j < rK;j++){
            tmp[0] += buffA[(ty * 4 + 0) * rN + j] * buffB[j * rN + tx ];
            tmp[1] += buffA[(ty * 4 + 1) * rN + j ] * buffB[j * rN + tx];
            tmp[2] += buffA[(ty * 4 + 2) * rN + j] * buffB[j * rN + tx];
            tmp[3] += buffA[(ty * 4 + 3) * rN + j] * buffB[j * rN + tx];
        }
        __syncthreads();
    }
    // printf("%d,%d\n",row,col);
    C[row * N + col] = tmp[0];
    C[(row + 1) * N + col] = tmp[1];
    C[(row + 2) * N + col] = tmp[2];
    C[(row + 3) * N + col] = tmp[3];
}

__global__ void SGEMM5(int M,int K,int N,float *A,float *B,float *C,float alpha, float beta){
    int bx = blockIdx.x;int by = blockIdx.y;
    int tx = threadIdx.x;int ty = threadIdx.y;
    int col = 4 * blockDim.x * bx + 4 * tx;
    int row = blockDim.y * by + ty;
    // printf("%d,%d,%d,%d\n",row,col,by,blockDim.y);
    float tmp[4] = {0., 0., 0., 0.};
    __shared__ float buffA[rM * rN];
    __shared__ float buffB[rM * rN];
    float4 Av;
    float4 bv;
    for(int i = 0; i < K; i+=rK){
        // Av = *(float4*)(&A[row * K + tx * 4 + i]);
        *(float4*)(&buffA[ty * rN + tx * 4]) = *(float4*)(&A[row * K + tx * 4 + i]);      
        
        // buffA[ty * rN + tx * 4 + 0] = A[row * K + tx * 4 + i];
        // buffA[ty * rN + tx * 4 + 1] = A[row * K + tx * 4 + 1 + i];
        // buffA[ty * rN + tx * 4 + 2] = A[row * K + tx * 4 + 2 + i];
        // buffA[ty * rN + tx * 4 + 3] = A[row * K + tx * 4 + 3 + i];

        // buffA[ty * rN + tx * 4 + 0] = A[ty * N + col + 0 + i * N];
        // buffA[ty * rN + tx * 4 + 1] = A[ty * N + col + 1 + i * N]; 
        // buffA[ty * rN + tx * 4 + 2] = A[ty * N + col + 2 + i * N];
        // buffA[ty * rN + tx * 4 + 3] = A[ty * N + col + 3 + i * N];
        // if(buffA[0] == 4.0){
        //     printf("!%d,%d,%d,%d,%d\n",row,col,row * K + (col + 0) + i,i,tx);
        // }
        *(float4*)(&buffB[ty * rN + tx * 4]) = *(float4*)(&B[ty * N + col + 0 + i * N]);   
        // buffB[ty * rN + tx * 4 + 0] = B[ty * N + col + 0 + i * N];
        // buffB[ty * rN + tx * 4 + 1] = B[ty * N + col + 1 + i * N]; 
        // buffB[ty * rN + tx * 4 + 2] = B[ty * N + col + 2 + i * N];
        // buffB[ty * rN + tx * 4 + 3] = B[ty * N + col + 3 + i * N];

        __syncthreads();
        #pragma unroll
        for(int j = 0; j < rK;j++){
            tmp[0] += buffA[(ty) * rN + j] * buffB[j * rN + 4 * tx + 0];
            tmp[1] += buffA[(ty) * rN + j] * buffB[j * rN + 4 * tx + 1];
            tmp[2] += buffA[(ty) * rN + j] * buffB[j * rN + 4 * tx + 2];
            tmp[3] += buffA[(ty) * rN + j] * buffB[j * rN + 4 * tx + 3];
        }
        __syncthreads();
    }

    // printf("%d,%d\n",row,col);
    C[row * N + col + 0] = tmp[0];
    C[(row) * N + col + 1] = tmp[1];
    C[(row) * N + col + 2] = tmp[2];
    C[(row) * N + col + 3] = tmp[3];
}
__global__ void SGEMM6(int M,int K,int N,float *A,float *B,float *C,float alpha, float beta){
    int bx = blockIdx.x;int by = blockIdx.y;
    int tx = threadIdx.x;int ty = threadIdx.y;
    int x = bx * blockDim.x + tx;
    int y = by * blockDim.y + ty;
    int col = x * lN;int row = y * lM;
    int bcol = tx * lN;int brow = ty * lM;
    float tmp[16] = {0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
    __shared__ float buffA[rM * rK];
    __shared__ float buffB[rK * rN];
    for(int i = 0; i < K; i += rK){
        #pragma unroll
        for(int r = 0; r < lM; r++){
            #pragma unroll
            for(int c = 0; c < lN; c++){
                buffA[(brow + r) * rK + bcol + c] = A[(row + r) * K + col + c];
            }
        }
        #pragma unroll
        for(int r = 0; r < lM; r++){
            #pragma unroll
            for(int c = 0; c < lN; c++){
                buffB[(brow + r) * rK + bcol + c] = B[(row + r) * N + col + c];
            }
        }
        #pragma unroll
        for(int j = 0; j < rK; j++){
            #pragma unroll
            for(int r = 0; r < (lM * lN); r++){
                tmp[r] += buffA[(brow + r / lM) * rK + j] * buffB[(j) * rN  + bcol + (r % lN)];
            }
        }
    }
    #pragma unroll
    for(int i = 0; i < lM * lN; i++){
        C[(row + lM) * N + col + (i%lN)] = tmp[i];
    }
}
__global__ void print(float *A){
    for(int i = 0; i < 64;i++){
        printf("%.2f ",A[i]);
    }printf("\n");
}
int main(int argc,char **argv){
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
    if(atoi(argv[1]) == 3){
        dim3 grid(N/rN,M/rM);
        dim3 block(rN,rM);
        start = cpuSecond();
        SGEMM3<<<grid,block>>>(M,K,N,A,B,D,alpha,beta);
        cudaDeviceSynchronize();
        printf("Times :%f\n",cpuSecond() - start);
    }
    else if(atoi(argv[1]) == 4){
        dim3 grid(N/rN,M/rM);
        dim3 block(rN,rM/4);
        start = cpuSecond();
        SGEMM4<<<grid,block>>>(M,K,N,A,B,D,alpha,beta);
        cudaDeviceSynchronize();
        printf("Times :%f\n",cpuSecond() - start);
    }
    else if(atoi(argv[1]) == 5){
        dim3 grid(N/rN,M/rM);
        dim3 block(rN/4,rM);
        start = cpuSecond();
        SGEMM5<<<grid,block>>>(M,K,N,A,B,D,alpha,beta);
        cudaDeviceSynchronize();
        printf("Times :%f\n",cpuSecond() - start);
    }
    else if(atoi(argv[1]) == 6){
        dim3 grid(N/rN,M/rM);
        dim3 block(rN/lN,rM/lM);
        start = cpuSecond();
        SGEMM5<<<grid,block>>>(M,K,N,A,B,D,alpha,beta);
        cudaDeviceSynchronize();
        printf("Times :%f\n",cpuSecond() - start);
    }
    else{
        exit(0);
    }
    print<<<1,1>>>(C);
    print<<<1,1>>>(D);
    cmp<<<M*N,1>>>(C,D);
    cudaDeviceSynchronize();
}