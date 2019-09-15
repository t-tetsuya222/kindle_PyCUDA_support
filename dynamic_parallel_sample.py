import math
import os
import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import DynamicSourceModule

# GPUの初期化
import pycuda.autoinit

# コンパイル時に余計なメッセージを表示させないようにする
os.environ["CL"] = r'-Xcompiler "/wd 4819'


# CUDA Cカーネルファイルの参照先の絶対パスを得る
cuda_file_path = os.path.abspath("./cuda")
#
#  CUDAカーネルの定義
#
module = DynamicSourceModule("""
#include "kernel_functions_for_dynamic_parallel.cu"
""", include_dirs=[cuda_file_path])

# コンパイルしたコードからカーネルを得る
add_two_vector_dynamic = module.get_function("add_two_vector_dynamic")

# 入力データの用意
num_comp = np.int32(10)
arr1 = np.arange(num_comp, dtype=np.float32)
arr2 = np.arange(num_comp, dtype=np.float32)
np.random.shuffle(arr2)

res_gpu = gpuarray.zeros(num_comp, dtype=np.float32)

threads_per_block = (256, 1, 1)
blocks_per_grid = (math.ceil(num_comp / threads_per_block[0]), 1, 1)
block = np.array(threads_per_block, dtype=np.int32)
grid = np.array(blocks_per_grid, dtype=np.int32)

# データをGPU上に送る
arr1_gpu = gpuarray.to_gpu(arr1)
arr2_gpu = gpuarray.to_gpu(arr2)
block_gpu = gpuarray.to_gpu(block)
grid_gpu = gpuarray.to_gpu(grid)

# カーネルを実行
add_two_vector_dynamic(grid_gpu, block_gpu, num_comp, arr1_gpu, arr2_gpu, res_gpu, block=(1, 1, 1), grid=(1, 1, 1))
print("answer :", arr1 + arr2)
print("result : ", res_gpu.get())
