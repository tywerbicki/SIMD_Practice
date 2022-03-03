# Compilation

Must include the `-mavx2` compiler flag when compiling with gcc/g++. <br>
As an example: 
> `g++ -O3 <program.cpp> -o <program.exe> -mavx2` <br>

Additionally, you can also specify the architecture of the target microprocessor by defining the `-march` compiler flag. <br/>
As an example:
> `g++ -O3 <program.cpp> -o <program.exe> -march=znver3 -mavx2` <br>

---

# Programs

### correlation.cpp

This program uses simd intrinsics to calculate the Pearson correlation coefficient between 2 vectors. This implementation generally executes 6-8X faster than single-precision scalar code.

### reduce.cpp

This program uses simd intrinsics to perform a summation reduction over a vector. This implementation generally executes 6-8X faster than single-precision scalar code.

### random_simd.c

This program uses simd intrinsics to generate uniform and normal random vectors quickly. 

Because this program uses some AVX2 intrinsics, the CPU architecture sometimes must be specified to the compiler via the `-march` compiler flag. Additionally, the `math.h` header file must be linked via the linker flag `-lm`. <br/>
As an example: `gcc -03 <random_simd.c> -o <test.exe> -march=znver3 -mavx2 -lm`

### doubleMask_simd.c

This program is a practice case of modifying an array based on two unique conditions using simd intrinsics. 
