#include <stdio.h>
#include <immintrin.h>

int main(void)
{
    const unsigned int size = 100000;
    const unsigned int alignment = sizeof(float) * 8;
    float* arr = (float*)_mm_malloc(sizeof(float) * size, alignment);

    for (size_t i = 0; i < size; i++)
    {
        *(arr + i) = 1.0 * i;
    }

    __m256 _v1, _v2, _two;
    _two = _mm256_set1_ps(2.0);

    for (size_t i = 0; i < size; i += 8)
    {
        _v1 = _mm256_load_ps(arr + i);
        _v2 = _mm256_mul_ps(_v1, _two);
        _mm256_store_ps(arr + i, _v2);
    }

    size_t j = 0;
    while (*(arr + j) < 25)
    {
        printf("%f ", *(arr + j));
        j++;
    }
    printf("\n");
    _mm_free(arr);
}
