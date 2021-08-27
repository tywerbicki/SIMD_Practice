# SIMD Practice Cases

## Compilation

Must include the `-mavx` compiler flag when compiling with GCC. <br>
As an example: `gcc <program.c> -o <program.exe> -mavx` <br>
Moreover, if pthreads is being used, the `-lpthread` compiler flag must be included. <br>
As an example: `gcc <program.c> -o <program.exe> -mavx -lpthread`

## Programs

### doubleMask_simd.c

This program is a practice case of modifying an array based on two unique conditions using simd intrinsics. 

### f32VectorReduce_mt_simd.c

This program is a practice case of using simd intrinsics with multiple threads to perform a reduction over a vector. The solution is somewhat undefined, as I currently have not figured out a method to communicate the alignment of the memory necessary for simd intrinsics to each thread. I am actively seeking a solution to this.

### random_simd.c

This program uses simd intrinsics to generate uniform and normal random vectors very quickly. <br>
Because this program uses some AVX2 intrinsics, the CPU architecture must be specified to the compiler via the `-march` compiler flag. Additionally, the `math.h` header file must be linked via the compiler flag `-lm`. <br>
As an example: `gcc <random_simd.c> -o <test.exe> -mavx -march=znver2 -lm`

### correlation_simd.c

This program uses simd intrinsics to calculate the Pearson correlation coefficient between 2 vectors. <br>
Compile with: `gcc <correlation_simd.c> -o <test.exe> -march=znver2 -lm` <br>
You'll notice that the simd implementation is about 2.5x slower than the scalar implementation. This is likely because of the frequent array loading and insignificant / uncomplex number crunching.
