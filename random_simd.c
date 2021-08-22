#include<stdio.h>
#include<math.h>
#include<immintrin.h>

//IEEE-754 float representation:
//https://www.h-schmidt.net/FloatConverter/IEEE754.html

void _mm256_print_ps(__m256* vec)
{
    float* elem = (float*)aligned_alloc(sizeof(float)*8, sizeof(float)*8);
    _mm256_store_ps(elem, *vec);
    for (size_t i = 0; i < 8; i++)
    {   printf("%f ", elem[i]); }
    printf("\n");
    free(elem);
}

//This is a sub-optimal implementation.
//When have more time, add SIMD taylor expansion implementation.
__m256 _mm256_log_ps(__m256* vec)
{
    float* restrict elem = (float*)aligned_alloc(sizeof(float)*8, sizeof(float)*8);
    _mm256_store_ps(elem, *vec);
    for (size_t i = 0; i < 8; i++)
    {   elem[i] = logf(elem[i]);    }
    __m256 _elem = _mm256_load_ps(elem); free(elem);
    return _elem;
}

__m256 _mm256_uniformRV_ps(const float a, const float b)
{
    float* restrict storeRand = (float*)aligned_alloc(sizeof(float) * 8, sizeof(float) * 8);
    for (size_t i = 0; i < 8; i++)
    {   storeRand[i] = (float)rand();    }
    
    static __m256 _mm256_RAND_MAX, _rand, _a, _b, _dif;
    _mm256_RAND_MAX = _mm256_set1_ps((float)RAND_MAX);
    _rand = _mm256_load_ps(storeRand); free(storeRand);
    _rand = _mm256_div_ps(_rand, _mm256_RAND_MAX);
    _a = _mm256_set1_ps(a); _b = _mm256_set1_ps(b); 
    _dif = _mm256_sub_ps(_b, _a);
    return _mm256_fmadd_ps(_rand, _dif, _a);
}

__m256 _mm256_normalRV_ps(const float mu, const float sigma)
{
    static __m256 _n2;
    static unsigned short generate = 1;

    if (!generate)
    {   generate = 1; return _n2;   }

    static __m256 _mu, _sigma;
    _mu = _mm256_set1_ps(mu); _sigma = _mm256_set1_ps(sigma);

    static __m256 _ones, _fullBits, _neg2s;
    _ones = _mm256_set1_ps(1.0F);
    _fullBits = _mm256_castsi256_ps(_mm256_set1_epi32(0xffffffff));
    _neg2s = _mm256_set1_ps(-2.0F);

    static __m256 _u1, _u1_tmp, _u2, _u2_tmp, _u22, _flag, _logFlag, _mask, _mask_flip, _n1, _tmp;
    _u1 = _mm256_uniformRV_ps(-1, 1); _u2 = _mm256_uniformRV_ps(-1, 1);
    _mask = _fullBits;

    do
    {
        _mask_flip = _mm256_xor_ps(_mask, _fullBits);
        _mask = _mm256_and_ps(_mask, _ones);
        _mask_flip = _mm256_and_ps(_mask_flip, _ones);

        _u1 = _mm256_mul_ps(_u1, _mask); _u2 = _mm256_mul_ps(_u2, _mask);

        _u1_tmp = _mm256_uniformRV_ps(-1, 1); _u2_tmp = _mm256_uniformRV_ps(-1, 1); 
        _u1_tmp = _mm256_mul_ps(_u1_tmp, _mask_flip); _u2_tmp = _mm256_mul_ps(_u2_tmp, _mask_flip);

        _u1 = _mm256_add_ps(_u1, _u1_tmp); _u2 = _mm256_add_ps(_u2, _u2_tmp);

        _u22 = _mm256_mul_ps(_u2, _u2);
        _flag = _mm256_fmadd_ps(_u1, _u1, _u22);
        _mask = _mm256_cmp_ps(_flag, _ones, _CMP_LT_OQ);
    } 
    while (_mm256_movemask_ps(_mask) < 0xFF); 
    
    _logFlag = _mm256_log_ps(&_flag);
    _tmp = _mm256_sqrt_ps(_mm256_mul_ps(_mm256_div_ps(_neg2s, _flag), _logFlag));
    _n1 = _mm256_mul_ps(_u1, _tmp);
    _n1 = _mm256_fmadd_ps(_n1, _sigma, _mu);
    _n2 = _mm256_mul_ps(_u2, _tmp);
    _n2 = _mm256_fmadd_ps(_n2, _sigma, _mu);
    generate = 0;

    return _n1;
}

int main()
{
    srand(9999);

    __m256 uniformVector = _mm256_uniformRV_ps(0, 10);
    _mm256_print_ps(&uniformVector);

    __m256 normalVector = _mm256_normalRV_ps(20, 1);
    _mm256_normalRV_ps(20, 1); //flush the RV generator (generates pairs of 2 vecs).
    __m256 normalVector2 = _mm256_normalRV_ps(10, 1);
    _mm256_print_ps(&normalVector);
    _mm256_print_ps(&normalVector2);

    return 0;
}
