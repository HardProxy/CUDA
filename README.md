# CUDA

Programa que resolver a Equação de Sine-Gordon por meio de técnicas resolução numérica utilizadno paralelismo em GPGPU.

## Pré-requisitos

- CUDA API
- CUDA Toolkit 10.1
- CUDA NVSIGHT-SYSTEM
- CUDA NVSIGHT-COMPUTE
- GNU compiler

## Compilação

nvcc -o EXECUTAVEL.x kernel.cu

## Execução 

./EXECUTAVEL.x

## Profile 

sudo nsys profile ./EXECUTAVEL.x

sudo nsys stats report1.qdrep > relatorio_profile.dat
