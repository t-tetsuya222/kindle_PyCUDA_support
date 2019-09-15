import math
import os
import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import pycuda.driver as drv
import cuda_utils as cu

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
#include "kernel_functions_for_atomic.cu"
""", include_dirs=[cuda_file_path])

# コンパイルしたコードからカーネルを得る
sum_kernel = module.get_function("sum_atomic")

nx = np.int32(10)
arr = np.arange(nx, dtype=np.int32)
sum_gpu = gpuarray.zeros(1, dtype=np.int32)
arr_gpu = gpuarray.to_gpu(arr)

# ブロック、グリッドの決定
threads_per_block = (256, 1, 1)
blocks_per_grid = (math.ceil(nx / threads_per_block[0]), 1, 1)

# アトミック演算で合計を求める
sum_kernel(nx, sum_gpu, arr_gpu, block=threads_per_block, grid=blocks_per_grid)

# 結果の出力
print("answer :", np.sum(arr))
print("atomic sum :", sum_gpu.get())
