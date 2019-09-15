import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
# スキャンのための関数をimport
from pycuda.scan import InclusiveScanKernel

# 配列の最大値を求める為の2項演算子
scan_kernel =InclusiveScanKernel(np.int32, "a > b ? a: b")
x = np.arange(10, dtype=np.int32)
np.random.seed(123)
np.random.shuffle(x)
print("original data: ", x)
x_gpu = gpuarray.to_gpu(x)

#  !!注意点!! scanカーネルでは配列の中身が上書きされる
x_gpu2 = x_gpu.copy()
scan_kernel(x_gpu2)
print("result: ", x_gpu2.get())
print("max :", np.max(x))
