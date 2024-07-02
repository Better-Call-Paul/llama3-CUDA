#include "utils.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

namespace llama {

long time_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int divUp(int a, int b) {
    return (a - 1) / b + 1;
}

void* safeMalloc(size_t size) {
    void* ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void safeFree(void* ptr) {
    free(ptr);
}

void* mapFile(const char* filename, size_t* filesize) {
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    
    *filesize = lseek(fd, 0, SEEK_END);
    void* data = mmap(NULL, *filesize, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    
    if (data == MAP_FAILED) {
        fprintf(stderr, "Error mapping file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    
    return data;
}

void unmapFile(void* data, size_t filesize) {
    munmap(data, filesize);
}

void cudaCheck(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", 
                code, cudaGetErrorString(code), file, line);
        exit(code);
    }
}

}