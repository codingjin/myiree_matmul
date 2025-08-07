#!/bin/bash
make

export OMP_PROC_BIND=true

./avx512_16_p 4096 128 4096

./avx512_16_p 128 8192 4096

./avx512_16_p 128 4096 8192

./avx512_16_p 4096 4096 4096
