#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <math.h>
#include <time.h>

size_t vecSize;
float* vecAddress1;
float* vecAddress2;

float f32Rho_scalar(const float* vecX, const float* vecY, const size_t size)
{
    float sumX = 0, sumY = 0, sumXY = 0, sumXX = 0, sumYY = 0, numerator, denominator;

    for (size_t i = 0; i < size; i++)
    {   
        sumX += vecX[i] ; sumY += vecY[i];  
        sumXY += vecX[i] * vecY[i];
        sumXX += vecX[i] * vecX[i];
        sumYY += vecY[i] * vecY[i];
    }

    numerator = (size*sumXY) - (sumX*sumY);
    denominator = sqrtf( (size*sumXX - sumX*sumX) * (size*sumYY - sumY*sumY) );
    
    return numerator / denominator;
}

static inline float _mm256_hSum_ps(const __m256* v)
{
    static __m128 _l, _h, _shift, _evenSums;
    _l = _mm256_extractf128_ps(*v, 0);
    _h = _mm256_extractf128_ps(*v, 1);
    _l = _mm_add_ps(_l, _h);
    _shift = _mm_movehdup_ps(_l);
    _evenSums = _mm_add_ps(_l, _shift);
    _shift = _mm_movehl_ps(_shift, _evenSums);
    _evenSums = _mm_add_ss(_evenSums, _shift);
    return _mm_cvtss_f32(_evenSums);
}

float f32Rho_simd(const float* vecX, const float* vecY, const size_t size)
{
    const unsigned int simdWork = size - (size % 8);
    static __m256 _tmpX, _tmpY, _tmp1, _tmp2, _tmp3;
    static float sumX, sumY, sumXY, sumXX, sumYY, numerator, denominator;
    sumX = 0 ; sumY = 0 ; sumXY = 0 ; sumXX = 0 ; sumYY = 0;
    for (size_t i = 0; i < simdWork; i += 8)
    {
        _tmpX = _mm256_loadu_ps(vecX + i);
        _tmpY = _mm256_loadu_ps(vecY + i);
        sumX += _mm256_hSum_ps(&_tmpX);
        sumY += _mm256_hSum_ps(&_tmpY);
        _tmp1 = _mm256_mul_ps(_tmpX, _tmpY);
        _tmp2 = _mm256_mul_ps(_tmpX, _tmpX);
        _tmp3 = _mm256_mul_ps(_tmpY, _tmpY);
        sumXY += _mm256_hSum_ps(&_tmp1);
        sumXX += _mm256_hSum_ps(&_tmp2);
        sumYY += _mm256_hSum_ps(&_tmp3);
    }
    for (size_t i = simdWork; i < size; i++)
    {   
        sumX += vecX[i] ; sumY += vecY[i];  
        sumXY += vecX[i] * vecY[i];
        sumXX += vecX[i] * vecX[i];
        sumYY += vecY[i] * vecY[i];
    }

    numerator = (size*sumXY) - (sumX*sumY);
    denominator = sqrtf( (size*sumXX - sumX*sumX) * (size*sumYY - sumY*sumY) );
    
    return numerator / denominator;
}

int main(void)
{   
    const size_t len = 48701;
    vecAddress1 = (float*)malloc(sizeof(float) * len);
    vecAddress2 = (float*)malloc(sizeof(float) * len);

    for (size_t i = 0; i < len; i++)
    {
        vecAddress1[i] = i * 2.0f;
        vecAddress2[i] = i* -1.0f / 3.0f;
    }

    clock_t start = clock();
    float rho = f32Rho_simd(vecAddress1, vecAddress2, len);
    clock_t end = clock();
    double exeTime = (double)(end - start) / CLOCKS_PER_SEC;
    
    start = clock();
    float rho_s = f32Rho_scalar(vecAddress1, vecAddress2, len);
    end = clock();
    double exeTime_s = (double)(end - start) / CLOCKS_PER_SEC;

    free(vecAddress1) ; free(vecAddress2);

    printf("Rho_simd: %f. Time: %lf \n", rho, exeTime);
    printf("Rho_scalar: %f. Time: %lf \n", rho_s, exeTime_s);
    
    return 0;
}
