import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
# cublasのimport
from skcuda import cublas

a = np.float32(2)
x = np.array([1, 2, 3], dtype=np.float32)
y = np.array([0.5, 0.5, 0.5], dtype=np.float32)

x_gpu = gpuarray.to_gpu(x)
y_gpu = gpuarray.to_gpu(y)

# 正解データ
print("answer :", a * x + y)

h = cublas.cublasCreate()
# y <- a * x + yを行う
cublas.cublasSaxpy(h, x_gpu.size, a, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
cublas.cublasDestroy(h)
print("cublas axpy :", y_gpu.get())
