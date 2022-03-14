#include <iostream>
#include <array>
#include <immintrin.h>
#include <chrono>

template <typename T, size_t n>
T sum(const std::array<T,n>& X) {

    T sum = 0;
    for (size_t i = 0; i < n; i++) sum += X[i];
    
    return sum;
}


static inline float _mm256_sum_reduction_ps(const __m256 vec) {
    
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
float sum_simd(const std::array<float,n>& X) {

    const float *X_r = X.data();
    const size_t stride = sizeof(__m256) / sizeof(float);
    const size_t border = n - (n % stride);

    __m256 _sum = _mm256_set1_ps(0.0F);
    for (size_t i = 0; i < border; i += stride) _sum += _mm256_loadu_ps(X_r + i);
    float sum = _mm256_sum_reduction_ps(_sum);
    for (size_t i = border; i < n; i++) sum += X_r[i];
    
    return sum;
}


int main() {

    const size_t n = (2 << 10) + 7;
    std::array<float,n> vec;
    
    for (size_t i = 0; i < n; i++) vec[i] = 1.0F;

    auto start = std::chrono::steady_clock::now();
    float result = sum<float,n>(vec);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end - start;

    start = std::chrono::steady_clock::now();
    float result_simd = sum_simd<n>(vec);
    end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration_simd = end - start;

    std::cout << "Sum: " << result << "  Duration: " << duration.count() << "\n";
    std::cout << "Sum_simd: " << result_simd << "  Duration: " << duration_simd.count() << std::endl;
}
