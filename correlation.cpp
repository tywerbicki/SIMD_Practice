#include <iostream>
#include <immintrin.h>
#include <cmath>
#include <chrono>

float pearsons_rho(const float * __restrict__ vecX, const float * __restrict__ vecY, const size_t n) {
    
    float sumX = 0.0, sumY = 0.0, sumXY = 0.0, sumXX = 0.0, sumYY = 0.0, numerator, denominator;

    for (size_t i = 0; i < n; i++) {   
        sumX += vecX[i] ; sumY += vecY[i];  
        sumXY += vecX[i] * vecY[i];
        sumXX += vecX[i] * vecX[i];
        sumYY += vecY[i] * vecY[i];
    }

    numerator = (n*sumXY) - (sumX*sumY);
    denominator = std::sqrt( (n*sumXX - sumX*sumX) * (n*sumYY - sumY*sumY) );
    
    return numerator / denominator;
}


float _mm256_sum_reduction_ps(const __m256 vec) {
    
    __m128 low = _mm256_extractf128_ps(vec, 0);
    __m128 high = _mm256_extractf128_ps(vec, 1);
    low = _mm_add_ps(low, high);
    high = _mm_permute_ps(low, 0b11101110);
    low = _mm_add_ps(low, high);
    high = _mm_permute_ps(low, 0b11100101);
    low = _mm_add_ps(low, high);
    return *((float*)&low);
}

float pearsons_rho_simd(const float * __restrict__ vecX, const float * __restrict__ vecY, const size_t n)
{
    const size_t stride = sizeof(__m256) / sizeof(float);
    const size_t simd_work = n - (n % stride);

    __m256 _sumX, _sumY, _sumXY, _sumXX, _sumYY; 
    _sumX = _sumY = _sumXY = _sumXX = _sumYY = _mm256_set1_ps(0.0F);
    __m256 _tmpX, _tmpY;
    
    
    for (size_t i = 0; i < simd_work; i += stride) {
        
        _tmpX = _mm256_loadu_ps(vecX + i);
        _tmpY = _mm256_loadu_ps(vecY + i);
        _sumX += _tmpX;
        _sumY += _tmpY;
        _sumXY = _mm256_fmadd_ps(_tmpX, _tmpY, _sumXY);
        _sumXX = _mm256_fmadd_ps(_tmpX, _tmpX, _sumXX);
        _sumYY = _mm256_fmadd_ps(_tmpY, _tmpY, _sumYY);
    }

    float sumX = _mm256_sum_reduction_ps(_sumX);
    float sumY = _mm256_sum_reduction_ps(_sumY);
    float sumXY = _mm256_sum_reduction_ps(_sumXY);
    float sumXX = _mm256_sum_reduction_ps(_sumXX);
    float sumYY = _mm256_sum_reduction_ps(_sumYY);
    
    for (size_t i = simd_work; i < n; i++) {

        sumX += vecX[i]; 
        sumY += vecY[i];  
        sumXY += vecX[i] * vecY[i];
        sumXX += vecX[i] * vecX[i];
        sumYY += vecY[i] * vecY[i];
    }

    float numerator = (n*sumXY) - (sumX*sumY);
    float denominator = std::sqrt( (n*sumXX - sumX*sumX) * (n*sumYY - sumY*sumY) );
    
    return numerator / denominator;
}



int main() {

    const size_t SIZE = 2500;
    float *vecX = new float[SIZE];
    float *vecY = new float[SIZE];

    for( size_t i = 0; i < SIZE; i++ ) {
        vecX[i] = (float)i; vecY[i] = (float)i + 2.0F;
    } 

    auto start = std::chrono::steady_clock::now();
    float rho = pearsons_rho(vecX, vecY, SIZE);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end - start;

    start = std::chrono::steady_clock::now();
    float rho_simd = pearsons_rho_simd(vecX, vecY, SIZE);
    end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration_simd = end - start;

    std::cout << "Rho: " << rho << "  Duration: " << duration.count() << "\n";
    std::cout << "Rho_simd: " << rho_simd << "  Duration: " << duration_simd.count() << std::endl;

    delete[] vecX; delete[] vecY;
}
