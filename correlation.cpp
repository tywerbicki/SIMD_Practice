#include <iostream>
#include <array>
#include <immintrin.h>
#include <cmath>
#include <chrono>

template<size_t n>
float pearsons_rho(const std::array<float,n>& arrX, const std::array<float,n>& arrY) {
    
    float sumX = 0.0, sumY = 0.0, sumXY = 0.0, sumXX = 0.0, sumYY = 0.0, tmpX, tmpY, numerator, denominator;

    for (size_t i = 0; i < n; i++) {   
        tmpX = arrX[i] ; tmpY = arrY[i];
        sumX += tmpX ; sumY += tmpY;  
        sumXY += tmpX * tmpY;
        sumXX += tmpX * tmpX;
        sumYY += tmpY * tmpY;
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

template<size_t n>
float pearsons_rho_simd(const std::array<float,n>& arrX, const std::array<float,n>& arrY)
{   
    const float *arrX_r = arrX.data();
    const float *arrY_r = arrY.data();
    const size_t stride = sizeof(__m256) / sizeof(float);
    const size_t simd_work = n - (n % stride);

    __m256 _sumX, _sumY, _sumXY, _sumXX, _sumYY; 
    _sumX = _sumY = _sumXY = _sumXX = _sumYY = _mm256_set1_ps(0.0F);
    __m256 _tmpX, _tmpY;
    
    
    for (size_t i = 0; i < simd_work; i += stride) {
        
        _tmpX = _mm256_loadu_ps(arrX_r + i);
        _tmpY = _mm256_loadu_ps(arrY_r + i);
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

        sumX += arrX[i]; 
        sumY += arrY[i];  
        sumXY += arrX[i] * arrY[i];
        sumXX += arrX[i] * arrX[i];
        sumYY += arrY[i] * arrY[i];
    }

    float numerator = (n*sumXY) - (sumX*sumY);
    float denominator = std::sqrt( (n*sumXX - sumX*sumX) * (n*sumYY - sumY*sumY) );
    
    return numerator / denominator;
}



int main() {

    const size_t SIZE = 2500UL;
    std::array<float,SIZE> arrX;
    std::array<float,SIZE> arrY;

    for( size_t i = 0; i < SIZE; i++ ) {
        arrX[i] = static_cast<float>(i); arrY[i] = static_cast<float>(i) + 2.0F;
    }
    
    auto start = std::chrono::steady_clock::now();
    float rho = pearsons_rho<SIZE>(arrX, arrY);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end - start;

    start = std::chrono::steady_clock::now();
    float rho_simd = pearsons_rho_simd<SIZE>(arrX, arrY);
    end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration_simd = end - start;

    std::cout << "Rho: " << rho << "  Duration: " << duration.count() << "\n";
    std::cout << "Rho_simd: " << rho_simd << "  Duration: " << duration_simd.count() << std::endl;
}
