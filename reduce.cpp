#include <iostream>
#include <immintrin.h>
#include <thread>
#include <mutex>
#include <chrono>

template <typename T>
T sum(const T * __restrict__ X, const size_t n) {

    T sum = 0;
    for (size_t i = 0; i < n; i++) sum += X[i];
    
    return sum;
}

float sum_simd(const float * __restrict__ X, const size_t n) {

    const size_t stride = sizeof(__m256) / sizeof(float);
    const size_t border = n - (n % stride);
    float tmp[stride];
    float sum = 0.0F;

    __m256 _sum = _mm256_set1_ps(0.0F);
    for (size_t i = 0; i < border; i += stride) _sum += _mm256_loadu_ps(X + i);
    _mm256_storeu_ps(tmp, _sum);
    for (size_t i = 0; i < stride; i++) sum += tmp[i];
    for (size_t i = border; i < n; i++) sum += X[i];
    
    return sum;
}


int main() {

    const size_t n = (2 << 8) + 7;
    float* vec = new float[n];
    
    for (size_t i = 0; i < n; i++) vec[i] = 2.0F;

    float result, result_simd;

    auto start = std::chrono::steady_clock::now();
    result = sum<float>(vec, n);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end - start;

    start = std::chrono::steady_clock::now();
    result_simd = sum_simd(vec, n);
    end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration_simd = end - start;

    delete[] vec;

    std::cout << "Sum: " << result << "  Duration: " << duration.count() << "\n";
    std::cout << "Sum_simd: " << result_simd << "  Duration: " << duration_simd.count() << std::endl;
}
