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
#include "kernel_functions_for_tex_2d.cu"
""", include_dirs=[cuda_file_path])

# コンパイルしたコードからカーネルを得る
read_tex_2d = module.get_function("read_texture_2d")

# 計算対象のnumpyアレーの作成
num_components = np.int32(20)
x = np.arange(num_components, dtype=np.int32)
# 順番をシャッフル
np.random.shuffle(x)
num_x, num_y = np.int32(5), np.int32(4)
x = x.reshape(num_y, num_x)

# 正解データを出力
print(x)
x_gpu = gpuarray.to_gpu(x)

# textureメモリへバインド
tex_2d = module.get_texref("tex_2d")
drv.matrix_to_texref(x, tex_2d, order="C")

# ブロック、グリッドの決定
threads_per_block = (16, 16, 1)
block_x = math.ceil(num_x / threads_per_block[0])
block_y = math.ceil(num_y / threads_per_block[1])
blocks_per_grid = (block_x, block_y, 1)

# CUDAカーネルの実行
read_tex_2d(num_x, num_y, block=threads_per_block, grid=blocks_per_grid, texrefs=[tex_2d])
