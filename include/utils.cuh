#ifndef UTILS_CUH
#define UTILS_CUH

namespace llama {

long time_in_ms();

int sample_argmax(float *probabilities, int n);

int divUp(int a, int b);

void* safeMalloc(size_t size);

void* mapFile(const char *file, size_t *filesize);

void unMapFile(void *data, size_t filesize);

void cudaCheck(cudaError_t error, const char *file, int line);

}

#endif