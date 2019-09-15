import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
# スキャンのための関数をimport
from pycuda.scan import InclusiveScanKernel

scan_kernel =InclusiveScanKernel(np.int32, "a+b")
x = np.arange(10, dtype=np.int32)
print("answer :", np.cumsum(x))
x_gpu = gpuarray.to_gpu(x)

#  !!注意点!! scanカーネルでは配列の中身が上書きされる
x_gpu2 = x_gpu.copy()
scan_kernel(x_gpu2)
print("scan result: ", x_gpu2.get())
