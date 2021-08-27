#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <pthread.h>

static inline float _mm_hSum_ps(__m128* v)
{
    __m128 _shift = _mm_movehdup_ps(*v);
    __m128 _evenSums = _mm_add_ps(*v, _shift);
    _shift = _mm_movehl_ps(_shift, _evenSums);
    _evenSums = _mm_add_ss(_evenSums, _shift);
    return _mm_cvtss_f32(_evenSums);
}

static inline float _mm256_hSum_ps(__m256* v)
{
    __m128 _l = _mm256_extractf128_ps(*v, 0);
    __m128 _h = _mm256_extractf128_ps(*v, 1);
    _l = _mm_add_ps(_l, _h);
    return _mm_hSum_ps(&_l);
}

float f32VectorReduce_simd(const float* vec, const unsigned int size)
{
    float sum = 0.0;
    const unsigned int simdWork = size - (size % 8);
    __m256 _tmp;
    for (size_t i = 0; i < simdWork; i += 8)
    {
        _tmp = _mm256_load_ps(vec + i);
        sum += _mm256_hSum_ps(&_tmp);
    }
    for (size_t i = simdWork; i < size; i++)
    {   sum += vec[i];  }

    return sum;
}

typedef struct threadData
{
    pthread_mutex_t* lockAddress;
    float* sumAddress;
    const float* vecAddress;
    unsigned int startIndex;
    unsigned int stopIndex;
} threadData_t;

void* f32VectorReduce_thread(void* arg)
{
    threadData_t* d = (threadData_t*)arg;
    const float* threadStart = d->vecAddress + d->startIndex;
    const unsigned int threadSize = d->stopIndex - d->startIndex;
    float partialSum = f32VectorReduce_simd(threadStart, threadSize);
    pthread_mutex_lock(d->lockAddress);
    *(d->sumAddress) += partialSum;
    pthread_mutex_unlock(d->lockAddress);
}

float f32VectorReduce_mt_simd(const unsigned int NUM_THREADS, const float* vecAddress, const unsigned int size)
{   
    pthread_mutex_t LOCK;
    int error = pthread_mutex_init(&LOCK, NULL);
    if (error != 0)
    {
        printf("Lock initialization failed.");
        exit(1);
    }
    float sum = 0.0;
    pthread_t threads[NUM_THREADS];
    threadData_t data[NUM_THREADS]; 
    const unsigned int elemPerThread = size / (NUM_THREADS * 8) * 8;
    
    for (size_t i = 0; i < NUM_THREADS; i++)
    {   
        data[i].lockAddress = &LOCK; data[i].sumAddress = &sum; data[i].vecAddress = vecAddress;
        data[i].startIndex = i * elemPerThread;
        data[i].stopIndex = (i + 1) * elemPerThread;
        pthread_create(&threads[i], NULL, &f32VectorReduce_thread, (void*)&data[i]);
    }
    pthread_mutex_lock(&LOCK);
    for (size_t i = elemPerThread * NUM_THREADS; i < size; i++)
    {   sum += vecAddress[i];   }
    pthread_mutex_unlock(&LOCK);
    
    for (size_t i = 0; i < NUM_THREADS; i++)
    {   pthread_join(threads[i], NULL);   }
    
    pthread_mutex_destroy(&LOCK);
    
    return sum;
}

int main(void)
{
    const unsigned int vecSize = 1001;
    float* vecAddress = (float*)malloc(sizeof(float) * vecSize);
    for (size_t i = 0; i < vecSize; i++)
    {   vecAddress[i] = 3.0;    }
    
    float sum = f32VectorReduce_mt_simd(3, vecAddress, vecSize);
    free(vecAddress);
    
    printf("Sum: %f \n", sum);

    return 0;
}
