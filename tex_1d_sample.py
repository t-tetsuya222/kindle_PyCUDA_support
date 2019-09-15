import math
import os
import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import pycuda.driver as drv

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
#include "kernel_functions_for_tex_1d.cu"
""", include_dirs=[cuda_file_path])

# コンパイルしたコードからカーネルを得る
read_tex_1d = module.get_function("read_texture_1d")

# 計算対象のnumpyアレーの作成
num_components = np.int32(10)
x = np.arange(num_components, dtype=np.int32)
# 順番をシャッフル
np.random.shuffle(x)
# 正解データを出力
print(x)
x_gpu = gpuarray.to_gpu(x)

# textureメモリへバインド
tex_1d = module.get_texref("tex_1d")
x_gpu.bind_to_texref_ext(tex_1d)
# ブロック、グリッドの決定
threads_per_block = (256, 1, 1)
blocks_per_grid = (math.ceil(num_components / threads_per_block[0]), 1, 1)

# CUDAカーネルの実行
read_tex_1d(num_components, block=threads_per_block, grid=blocks_per_grid, texrefs=[tex_1d])
