import math
import os
import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

# GPUの初期化
import pycuda.autoinit

# CUDA Cカーネルファイルの参照先の絶対パスを得る
cuda_file_path = os.path.abspath("./cuda")

# コンパイル時に余計なメッセージを表示させないようにする
os.environ["CL"] = r'-Xcompiler "/wd 4819'

#
#  CUDAカーネルの定義
#
module = SourceModule("""
#include "kernel_functions_for_math_2d.cu"
""", include_dirs=[cuda_file_path])

# コンパイルしたコードからカーネルを得る
add_two_array = module.get_function("add_two_array_kernel")

# 計算対象のnumpyアレーの作成
num_x, num_y = np.int32(5), np.int32(2)
num_components = num_x * num_y
x = np.arange(num_components, dtype=np.float32).reshape(5, 2)
y = 10 * np.random.rand(5, 2)
y = np.float32(y)
res = np.zeros([5, 2], dtype=np.float32)

# cpu to gpuへデータを送付
x_gpu = gpuarray.to_gpu(x)
y_gpu = gpuarray.to_gpu(y)
res_gpu = gpuarray.to_gpu(res)


# ブロック、グリッドの決定
threads_per_block = (16, 16, 1)
block_x = math.ceil(num_x / threads_per_block[0])
block_y = math.ceil(num_y / threads_per_block[1])
blocks_per_grid = (block_x, block_y, 1)

# CUDAカーネルの実行
add_two_array(num_x, num_y, res_gpu, x_gpu, y_gpu,
              block=threads_per_block, grid=blocks_per_grid)

# gpuからcpuへデータを送付
res = res_gpu.get()

print("result :", res)
