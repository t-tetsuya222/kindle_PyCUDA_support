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
#include "kernel_functions_for_thrust.cu"
""", include_dirs=[cuda_file_path], no_extern_c=True)

# コンパイルしたコードからカーネルを得る
sort_thrust = module.get_function("sort_thrust")

# 入力データの作成
nx = np.int32(10)
arr = np.arange(nx, dtype=np.int32)
np.random.shuffle(arr)
print("before sort : ", arr)

arr_gpu = gpuarray.to_gpu(arr)

# thrustでのソートを実行
sort_thrust(nx, arr_gpu, block=(1, 1, 1), grid=(1, 1, 1))

# 結果の出力
print("after sort :", arr_gpu.get())
