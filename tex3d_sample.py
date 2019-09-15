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
#include "kernel_functions_for_tex_3d.cu"
""", include_dirs=[cuda_file_path])

# コンパイルしたコードからカーネルを得る
read_tex_3d = module.get_function("read_texture_3d")

# 計算対象のnumpyアレーの作成
num_components = np.int32(24)
x = np.arange(num_components, dtype=np.int32)
# 順番をシャッフル
np.random.shuffle(x)
num_x, num_y, num_z = np.int32(2), np.int32(3), np.int32(4)
x = x.reshape(num_z, num_y, num_x)

# 正解データを出力
print(x)
x_gpu = gpuarray.to_gpu(x)

# textureメモリへバインド
tex_3d = module.get_texref("tex_3d")
cu.bind_array_to_texture3d(x, tex_3d)

# ブロック、グリッドの決定
threads_per_block = (6, 6, 6)
block_x = math.ceil(num_x / threads_per_block[0])
block_y = math.ceil(num_y / threads_per_block[1])
block_z = math.ceil(num_z / threads_per_block[2])
blocks_per_grid = (block_x, block_y, block_z)

# CUDAカーネルの実行
read_tex_3d(num_x, num_y, num_z, block=threads_per_block, grid=blocks_per_grid, texrefs=[tex_3d])
