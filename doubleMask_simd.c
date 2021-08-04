//Using Ryzen 5800x CPU, so AVX512 is not available.
//Need to add -march=zn2 gcc compile flag if want to use masks.

#include <stdio.h>
#include <immintrin.h>

//For debugging.
void _mm256_print_ps(__m256* vec)
{
    float* elem = (float*)aligned_alloc(sizeof(float)*8, sizeof(float)*8);
    _mm256_store_ps(elem, *vec);
    for (size_t i = 0; i < 8; i++)
    {
        printf("%f ", *(elem + i));
    }
    printf("\n");
    free(elem);
}

//For debugging.
void _mm256_print_si256(__m256i* vec)
{
    int* elem = (int*)aligned_alloc(sizeof(int)*8, sizeof(int)*8);
    _mm256_store_si256((__m256i*)elem, *vec);
    for (size_t i = 0; i < 8; i++)
    {
        printf("%d ", *(elem + i));
    }
    printf("\n");
    free(elem);
}

int main(void)
{
    const unsigned int size = 23;
    float* restrict arr = (float*)aligned_alloc(sizeof(float)*8, sizeof(float)*size);
    for (size_t i = 0; i < size; i++)
    {
        *(arr + i) = 1.0 * i;
    }

    //Determine how much work can be done with SIMD intrinsics.
    const unsigned int simdLength = size - (size % 8);
    //Allocate 256-bit registers (8 single precision stores)
    __m256 _v, _mask1_ps, _mask2_ps, _ones_ps;
    _ones_ps = _mm256_set1_ps(1.0);

    for (size_t i = 0; i < simdLength; i += 8)
    {   
        //Load array into 256-bit register.
        _v = _mm256_load_ps(arr + i);

        //Create mask on first condition.
        _mask1_ps = _mm256_set1_ps(i + 2.0);
        _mask1_ps = _mm256_cmp_ps(_v, _mask1_ps, _CMP_GE_OQ);

        //Create mask on second condition.
        _mask2_ps = _mm256_set1_ps(i + 5.0);
        _mask2_ps = _mm256_cmp_ps(_v, _mask2_ps, _CMP_LE_OQ);
        
        //Combine masks and add one.
        _mask1_ps = _mm256_and_ps(_mask1_ps, _mask2_ps);
        _mask1_ps = _mm256_and_ps(_mask1_ps, _ones_ps);
        _mask1_ps = _mm256_add_ps(_mask1_ps, _ones_ps);
        
        //Multiply the contents in the register by the mask.
        _v = _mm256_mul_ps(_v, _mask1_ps);

        //Store results back into arr.
        _mm256_store_ps(arr + i, _v);
    }
    //Do the leftover work with scalar intructions.
    for (size_t i = simdLength; i < size; i++)
    {
        if ((i >= simdLength + 2) && (i <= simdLength + 5))
        {
            *(arr + i) *= 2;
        }
    }
    
    for (size_t i = 0; i < size; i++)
    {
        printf("%f ", *(arr + i));
    }
    printf("\n");

    free(arr);
}
