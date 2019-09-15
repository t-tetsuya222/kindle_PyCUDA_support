import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.elementwise import ElementwiseKernel

# GPUの初期化
import pycuda.autoinit

#
#  CUDAカーネルの定義
#
plus_one_kernel = ElementwiseKernel(
    "int *y, int *x",
    "y[i] = x[i] + 1",
    "plus_one")

# 計算対象のnumpyアレーの作成
num_components = 10
x = np.arange(num_components, dtype=np.int32)

# cpu to gpuへデータを送付
x_gpu = gpuarray.to_gpu(x)
y_gpu = gpuarray.zeros(num_components, dtype=np.int32)

# CUDAカーネルの実行
plus_one_kernel(y_gpu, x_gpu)

# gpu to cpuへデータを送付
y = y_gpu.get()

print("x :", x)
print("y :", y)
