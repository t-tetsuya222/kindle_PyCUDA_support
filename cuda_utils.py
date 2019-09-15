import pycuda.driver as drv


def bind_array_to_texture3d(np_array, tex_ref):
    """
    generate array for texture3d based on numpy array
    :param np_array:
    :param tex_ref:
    :return: array for binding to texture reference
    """
    # get shape
    # caution: (d, h, w) C manner
    #        : (w, h, d) Fortran manner
    #
    d, h, w = np_array.shape
    # generate descriptor
    descr = drv.ArrayDescriptor3D()
    descr.width = w
    descr.height = h
    descr.depth = d
    descr.format = drv.dtype_to_array_format(np_array.dtype)
    descr.num_channels = 1
    descr.flags = 0
    device_array = drv.Array(descr)
    # method for copy
    copy = drv.Memcpy3D()
    copy.set_src_host(np_array)
    copy.set_dst_array(device_array)
    copy.width_in_bytes = copy.src_pitch = np_array.strides[1]
    copy.src_height = copy.height = h
    copy.depth = d
    copy()
    # bind array to texture 3d
    tex_ref.set_array(device_array)


if __name__ == "__main__":
    pass
