import math
import os
import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

# GPUの初期化
import pycuda.autoinit

# コンパイル時に余計なメッセージを表示させないようにする
os.environ["CL"] = r'-Xcompiler "/wd 4819'


# CUDA Cカーネルファイルの参照先の絶対パスを得る
cuda_file_path = os.path.abspath("./cuda")
#
#  CUDAカーネルの定義
#
module = SourceModule("""
#include "kernel_functions_for_math_1d.cu"
""", include_dirs=[cuda_file_path])

# コンパイルしたコードからカーネルを得る
plus_one_kernel = module.get_function("plus_one_kernel")

# 計算対象のnumpyアレーの作成
num_components = np.int32(10)
x = np.arange(num_components, dtype=np.int32)

# cpu to gpuへデータを送付
x_gpu = gpuarray.to_gpu(x)
y_gpu = gpuarray.zeros(num_components, dtype=np.int32)

# ブロック、グリッドの決定
threads_per_block = (256, 1, 1)
blocks_per_grid = (math.ceil(num_components / threads_per_block[0]), 1, 1)

# CUDAカーネルの実行
plus_one_kernel(num_components, y_gpu, x_gpu, block=threads_per_block, grid=blocks_per_grid)

# gpu to cpuへデータを送付
y = y_gpu.get()

print("x :", x)
print("y :", y)
