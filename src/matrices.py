import numpy
import numpy.lib.stride_tricks as numpy_stride


def integer_repr(array):
    representation = 0
    for bit in array:
        representation = (representation << 1) | bit
    return representation


def windows(matrix):
    grown = grow(matrix)
    windows = numpy_stride.as_strided(
        grown, shape=matrix.shape + (3, 3), strides=grown.strides + grown.strides
    )
    return windows.reshape(matrix.shape + (9,))


def grow(matrix):
    grown_v = numpy.concatenate((matrix[-1, :], matrix, matrix[0, :]), axis=0)
    grown = numpy.concatenate((grown_v[:, -1], grown_v, grown_v[:, 0]), axis=1)
    return grown
