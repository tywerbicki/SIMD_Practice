# SIMD Practice Cases

## Compilation

Must include the `-mavx` compiler flag when compiling with gcc/g++. <br>
As an example: 
> `g++ -O3 <program.cpp> -o <program.exe> -mavx` <br>

Additionally, you can also specify the architecture of the target microprocessor by defining the `-march` compiler flag. <br/>
As an example:
> `g++ -O3 <program.cpp> -o <program.exe> -march=znver3 -mavx` <br>

## Programs

### doubleMask_simd.c

This program is a practice case of modifying an array based on two unique conditions using simd intrinsics. 

### reduce.cpp

This program uses simd intrinsics to perform a summation reduction over a vector. This implementation generally executes 6-7X faster than scalar code using 32-bit registers.

### random_simd.c

This program uses simd intrinsics to generate uniform and normal random vectors very quickly. <br>
Because this program uses some AVX2 intrinsics, the CPU architecture must be specified to the compiler via the `-march` compiler flag. Additionally, the `math.h` header file must be linked via the compiler flag `-lm`. <br>
As an example: `gcc <random_simd.c> -o <test.exe> -mavx -march=znver2 -lm`

### correlation.cpp

This program uses simd intrinsics to calculate the Pearson correlation coefficient between 2 vectors. This implementation generally executes 6-7X faster than scalar code using 32-bit registers.
