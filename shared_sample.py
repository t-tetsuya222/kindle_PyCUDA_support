import math
import os
import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
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
#include "kernel_functions_for_shared.cu"
""", include_dirs=[cuda_file_path])

# コンパイルしたコードからカーネルを得る
grad_x_s = module.get_function("calc_grad_shared_3d")
grad_x_g = module.get_function("calc_grad_global_3d")

kernel_s = drv.Event()
kernel_e = drv.Event()


# meshgrid用のnumpyアレー作成
dx = np.float32(0.01)
x = np.arange(0, 4, dx, dtype=np.float32)
y = np.arange(0, 6, dx, dtype=np.float32)
z = np.arange(0, 8, dx, dtype=np.float32)

num_x = np.int32(len(x))
num_y = np.int32(len(y))
num_z = np.int32(len(z))
num_components = num_x * num_y * num_z

Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
# 計算対象のnumpyアレー作成
arr = X ** 2

# cpu to gpuへデータを送付
arr_gpu = gpuarray.to_gpu(arr)
num_direction = np.int32(3)
arr_grad_gpu = gpuarray.zeros([num_direction, num_z, num_y, num_x], dtype=np.float32)


# ブロック、グリッドの決定
threads_per_block = (6, 6, 6)
block_x = math.ceil(num_x / threads_per_block[0])
block_y = math.ceil(num_y / threads_per_block[1])
block_z = math.ceil(num_z / threads_per_block[2])
blocks_per_grid = (block_x, block_y, block_z)

kernel_s.record()
grad_x_g(num_x, num_y, num_z, dx, arr_grad_gpu, arr_gpu, block=threads_per_block, grid=blocks_per_grid)
kernel_e.record()
kernel_e.synchronize()
print("Global memory: {} [ms]".format(kernel_s.time_till(kernel_e)))
arr_global = arr_grad_gpu.get()


# CUDAカーネルの実行
kernel_s.record()
grad_x_s(num_x, num_y, num_z, dx, arr_grad_gpu, arr_gpu, block=threads_per_block, grid=blocks_per_grid)
kernel_e.record()
kernel_e.synchronize()
print("Shared memory: {} [ms]".format(kernel_s.time_till(kernel_e)))
arr_shared = arr_grad_gpu.get()

# gpuからcpuへデータを送付
arr_grad = arr_grad_gpu.get()

print("result :", arr_grad)
