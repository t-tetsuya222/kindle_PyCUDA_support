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
#include "kernel_functions_for_template.cu"
""", include_dirs=[cuda_file_path], no_extern_c=True)

# コンパイルしたコードからカーネルを得る
add_two_vector = module.get_function("add_two_vector_kernel")

# 計算対象のnumpyアレーの作成
np.random.seed(123)
num_components = np.int32(10)
x = np.arange(num_components, dtype=np.int32)
y = np.random.randint(0, 10, num_components, dtype=np.int32)

# cpu to gpuへデータを送付
x_gpu = gpuarray.to_gpu(x)
y_gpu = gpuarray.to_gpu(y)
res_gpu = gpuarray.zeros(num_components, dtype=np.int32)

# ブロック、グリッドの決定
threads_per_block = (256, 1, 1)
blocks_per_grid = (math.ceil(num_components / threads_per_block[0]), 1, 1)

# CUDAカーネルの実行
add_two_vector(num_components, x_gpu, y_gpu, res_gpu, block=threads_per_block, grid=blocks_per_grid)

# gpu to cpuへデータを送付
res = res_gpu.get()

print("answer :", x + y)
print("result :", res_gpu.get())
