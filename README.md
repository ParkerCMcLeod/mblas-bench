## Basic BLAS benchmark for GEMMs


Goals:
- Use the familiar interface of rocblas-bench
- Implement native support for cuBLAS, cuBLASLT, and Intel's GPU BLAS
- Expose implementation specific tuneables to the user

### Building for both CUDA and ROCm
```
cmake -S src -B build  
cmake --build build  
```

### Building for ROCm only
```
cmake -S src -B build -DWITH_CUDA=false  
cmake --build build  
```

### Building for CUDA only
```
cmake -S src -B build -DWITH_ROCM=false  
cmake --build build  
```

### Running on ROCm or CUDA
#### Run 4k gemms
Use the below commands to run "4k" gemms on ROCm or CUDA. You'll need to add the correct `--driver` flag to select a compatible backend.  See below for available options 

| Precision | Base Command                                                                                                                                                                                                                                                                            |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FP64      | build/mblas-bench -m 4096 -n 4096 -k 4096 --alpha 1 --beta 0 --transposeA N --transposeB T --initialization trig_float --iters 5000 --cold_iters 1500 --function matmul --precision d --rotating 512                                                                           |
| FP32      | build/mblas-bench -m 4096 -n 4096 -k 4096 --alpha 1 --beta 0 --transposeA N --transposeB T --initialization trig_float --iters 10000 --cold_iters 2500 --a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r --function matmul --rotating 512       |
| TF32      | build/mblas-bench -m 4096 -n 4096 -k 4096 --alpha 1 --beta 0 --transposeA T --transposeB N --initialization trig_float --iters 75000 --cold_iters 20000 --compute_type CUBLAS_COMPUTE_32F_FAST_TF32 --function gemm_ex --precision s --rotating 512                            |
| FP16      | build/mblas-bench -m 4096 -n 4096 -k 4096 --alpha 1 --beta 0 --transposeA N --transposeB T --initialization trig_float --iters 100000 --cold_iters 25000 --a_type f16_r --b_type f16_r --c_type f16_r --d_type f16_r --compute_type f32_r --function matmul --rotating 512     |
| BF16      | build/mblas-bench -m 4096 -n 4096 -k 4096 --alpha 1 --beta 0 --transposeA N --transposeB T --initialization trig_float --iters 100000 --cold_iters 25000 --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r --compute_type f32_r --function matmul --rotating 512 |
| FP8       | build/mblas-bench -m 4096 -n 4096 -k 4096 --alpha 1 --beta 0 --transposeA T --transposeB N --initialization trig_float --iters 250000 --cold_iters 50000 --a_type f8_r --b_type f8_r --c_type f16_r --d_type f16_r --compute_type f32_r --function matmul --rotating 512       |
| INT8      | build/mblas-bench -m 4096 -n 4096 -k 4096 --alpha 1 --beta 0 --transposeA T --transposeB N --initialization rand_int --iters 25000 --cold_iters 50000 --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r --compute_type i32_r --function matmul --rotating 512          |
#### ROCm backends
Use the hipblaslt backend with the flag `--driver hipblaslt`
#### CUDA backends
Use the cublaslt backend with the flag `--driver cublaslt`