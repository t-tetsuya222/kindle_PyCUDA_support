import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
# cublasを使う為にimport
from skcuda import cublas

# 最大値を求める用のnumpyアレー
np.random.seed(123)
x = np.random.randint(1, 100, 20, dtype=np.int32)
x_gpu = gpuarray.to_gpu(x)

# 正解を出力
print("target numpy array is : ", x)
print("answer : ", np.argmax(x))

# cublasで最大値を求める
h = cublas.cublasCreate()
max_id = cublas.cublasIsamax(h, x_gpu.size, x_gpu.gpudata, 1)
cublas.cublasDestroy(h)
print("max id based on cublas : ", max_id)