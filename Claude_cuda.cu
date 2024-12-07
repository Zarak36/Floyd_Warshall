#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <ctime>
#include <iostream>
#include <cstdlib>

#define BLOCK_SIZE 32
#define INF 999999

// Phase 1: Update the k-th row and column
__global__ void phase1Kernel(int* dist, int n, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && idx != k) {
        int ik = k * n + idx;  // k-th row
        int ki = idx * n + k;  // k-th column
        dist[ik] = dist[ik];
        dist[ki] = dist[ki];
        __syncthreads();
    }
}

// Phase 2: Update all remaining elements using the k-th row and column
__global__ void phase2Kernel(int* dist, int n, int k) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
   
    // Shared memory for k-th row and column
    __shared__ int row_k[BLOCK_SIZE];
    __shared__ int col_k[BLOCK_SIZE];
   
    if (i < n && j < n && i != k && j != k) {
        // Load k-th row and column elements into shared memory
        if (threadIdx.x == 0) {
            row_k[threadIdx.y] = dist[k * n + j];
            col_k[threadIdx.y] = dist[i * n + k];
        }
        __syncthreads();
       
        int idx = i * n + j;
        int new_dist = col_k[threadIdx.y] + row_k[threadIdx.x];
       
        if (col_k[threadIdx.y] != INF && row_k[threadIdx.x] != INF &&
            new_dist < dist[idx]) {
            dist[idx] = new_dist;
        }
    }
}
// Stop measuring time
    
    
    float milliseconds = 0;

// Host function to run Floyd-Warshall algorithm
void floydWarshallGPU(int* graph, int n) {
    int size = n * n * sizeof(int);
    int* d_graph;
   
    // Allocate device memory
    cudaMalloc((void**)&d_graph, size);
    cudaMemcpy(d_graph, graph, size, cudaMemcpyHostToDevice);
   
    // Calculate grid and block dimensions
    dim3 block1D(BLOCK_SIZE);
    dim3 grid1D((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
   
    dim3 block2D(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid2D((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
   
    // Main loop
    for (int k = 0; k < n; k++) {
        // Phase 1: Update k-th row and column
        phase1Kernel<<<grid1D, block1D>>>(d_graph, n, k);
        cudaDeviceSynchronize();
       
        // Phase 2: Update remaining elements
        phase2Kernel<<<grid2D, block2D>>>(d_graph, n, k);
        cudaDeviceSynchronize();
        
    }
   
    // Copy result back to host
    cudaMemcpy(graph, d_graph, size, cudaMemcpyDeviceToHost);
    cudaFree(d_graph);
    std::cout << "Time taken: " << milliseconds / 1000.0 << " seconds\n";
}

// Example usage
int main() {
    const int n = 1024;  // Graph size
    int* graph = new int[n * n];
   
    // Initialize graph with distances
    // ... (initialization code here)
   
    // Run Floyd-Warshall algorithm on GPU
    floydWarshallGPU(graph, n);
   
    delete[] graph;
    return 0;
}