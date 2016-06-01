#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

__global__ void vectorAddKer(int *d_A, int *d_B, int *d_C, int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if(index >= size) {
        return;
    }
    d_C[index] = d_A[index] + d_B[index];
}



int main(int argc, char const *argv[]) {
    int size = 0;
    if(argc != 2) {
        size = 1024;
    } else {
        size = atoi(argv[1]);
    }

    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;

    h_A = new int[size];
    h_B = new int[size];
    h_C = new int[size];

    for(unsigned i = 0; i < size; ++i) {
        h_A[i] = i;
        h_B[i] = size - i;
    }

    cudaEvent_t start, stop;
    float elapsedTime = 0.0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMalloc(&d_A, size * sizeof(int));
    cudaMalloc(&d_B, size * sizeof(int));
    cudaMalloc(&d_C, size * sizeof(int));

    cudaMemcpy(d_A, h_A, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 gridsize, blocksize;
    blocksize.x = 256;
    gridsize.x = (size + blocksize.x - 1) / blocksize.x;

    vectorAddKer<<<gridsize, blocksize>>>(d_A, d_B, d_C, size);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);

    cout << "time=" << elapsedTime << endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_C, d_C, size * sizeof(int), cudaMemcpyDeviceToHost);

    for(unsigned i = 0; i < size; ++i) {
        printf("%d + %d = %d\n", h_A[i], h_B[i], h_C[i]);
    }

    return 0;
}