import torch as th
import math

''' building complex number '''
def stack_complex(real_part, img_part):
    return th.stack((real_part, img_part), -1)

''' extracting complex information'''
def unstack_complex(stacked_array):
    return stacked_array[..., 0], stacked_array[..., 1]

''' conjugate operation '''
def conj(real_imag):
    real, imag = unstack_complex(real_imag)
    return stack_complex(real, -imag)

''' multiplying complex numbers '''
def mul_complex(field1, field2):
    real1, imag1 = unstack_complex(field1)
    real2, imag2 = unstack_complex(field2)

    real = real1 * real2 - imag1 * imag2
    imag = real1 * imag2 + imag1 * real2

    return stack_complex(real, imag)


''' real part function '''
def real_cart(real_imag):
    return real_imag[...,0]

''' imaginary part function '''
def imag_cart(real_imag):
    return real_imag[...,1]

''' from cartesian to polar coordinates '''
def cart_to_polar(com_image):
    real,imag = unstack_complex(com_image)
    mag = th.pow(real**2 + imag**2, 0.5)
    ang = th.atan2(imag, real)
    return mag, ang

''' absolute value '''
def get_abs(com_image):
    real, imag = unstack_complex(com_image)
    mag = th.pow(real ** 2 + imag ** 2, 0.5)
    return mag

''' obtaning phase '''
def get_phase(com_image):
    real, imag = unstack_complex(com_image)
    ang = th.atan2(imag, real)
    return ang

''' matrix multiplication '''
def mtprod(ima1,ima2):
    A,B = unstack_complex(ima1)
    C,D = unstack_complex(ima2)

    re = th.matmul(A,C) - th.matmul(B,D)
    im = th.matmul(A,D) + th.matmul(B,C)
    return stack_complex(re,im)

def mtprod_cr(ima1,im_real):
    A,B = unstack_complex(ima1)

    re = th.matmul(A,im_real)
    im = th.matmul(B,im_real)
    return stack_complex(re,im)

def mtprod_rc(im_real,ima1):
    A,B = unstack_complex(ima1)

    re = th.matmul(im_real,A)
    im = th.matmul(im_real,B)
    return stack_complex(re,im)

''' 2D fourier transform '''
def fft(real_imag, ndims=2, normalized=False, pad=None, padval=0):
    return th.fft(real_imag, ndims,normalized=normalized)

def rfft(input_real, ndims=2, normalized=False, pad=None, padval=0,onesided=True):
    return th.rfft(irfftshift(input_real, ndims), ndims,
                      normalized=normalized, onesided=onesided)

''' auxiliary functions '''
def ifftshift(array, ndims=2):
    return fftshift(array, ndims, invert=True)

def irfftshift(array, ndims=2):
    return rfftshift(array, ndims, invert=True)

def rfftshift(array, ndims=2, invert=False):
    shift_adjust = 0 if invert else 1

    if ndims >= 1:
        shift_len = (array.shape[-1] + shift_adjust) // 2
        array = th.cat((array[..., shift_len:],
                           array[..., :shift_len]), -1)
    if ndims >= 2:
        shift_len = (array.shape[-2] + shift_adjust) // 2
        array = th.cat((array[..., shift_len:, :],
                           array[..., :shift_len, :]), -2)
    if ndims == 3:
        shift_len = (array.shape[-3] + shift_adjust) // 2
        array = th.cat((array[..., shift_len:, :, :],
                           array[..., :shift_len, :, :]), -3)
    return array

def fftshift(array, ndims=2, invert=False):
    shift_adjust = 0 if invert else 1

    # skips the last dimension, assuming stacked fft output
    if ndims >= 1:
        shift_len = (array.shape[-2] + shift_adjust) // 2
        array = th.cat((array[..., shift_len:, :],
                           array[..., :shift_len, :]), -2)
    if ndims >= 2:
        shift_len = (array.shape[-3] + shift_adjust) // 2
        array = th.cat((array[..., shift_len:, :, :],
                           array[..., :shift_len, :, :]), -3)
    if ndims == 3:
        shift_len = (array.shape[-4] + shift_adjust) // 2
        array = th.cat((array[..., shift_len:, :, :, :],
                           array[..., :shift_len, :, :, :]), -4)
    return array

''' 2D convolution '''
def conv_fft(img_real_imag, kernel_real_imag, padval=0):
    img_pad, kernel_pad, output_pad = conv_pad_sizes(img_real_imag.shape,
                                                     kernel_real_imag.shape)

    # fft
    img_fft = fft(img_real_imag, pad=img_pad, padval=padval)
    kernel_fft = fft(kernel_real_imag, pad=kernel_pad, padval=0)

    # ifft, using img_pad to bring output to img input size
    return ifft(mul_complex(img_fft, kernel_fft), pad=output_pad)


def conv_rfft(img, kernel, padval=0):
    img_pad = kernel_pad = output_pad = None
    img_fft = rfft(img, pad=img_pad, padval=padval)
    kernel_fft = rfft(kernel, pad=kernel_pad, padval=0)

    return irfft(mul_complex(img_fft, kernel_fft), signal_sizes=[img.shape[-2], img.shape[-1]], pad=output_pad)


''' padding '''
def conv_pad_sizes(image_shape, kernel_shape):
    # skips the last dimension, assuming stacked fft output
    # minimum required padding is to img.shape + kernel.shape - 1
    # padding based on matching fftconvolve output

    # when kernels are even, padding the extra 1 before/after matters
    img_pad_end = (1 - ((kernel_shape[-2] % 2) | (image_shape[-2] % 2)),
                   1 - ((kernel_shape[-3] % 2) | (image_shape[-3] % 2)))

    image_pad = ((kernel_shape[-2] - img_pad_end[0]) // 2,
                 (kernel_shape[-2] - 1 + img_pad_end[0]) // 2,
                 (kernel_shape[-3] - img_pad_end[1]) // 2,
                 (kernel_shape[-3] - 1 + img_pad_end[1]) // 2)
    kernel_pad = (image_shape[-2] // 2, (image_shape[-2] - 1) // 2,
                  image_shape[-3] // 2, (image_shape[-3] - 1) // 2)
    output_pad = ((kernel_shape[-2] - 1) // 2, kernel_shape[-2] // 2,
                  (kernel_shape[-3] - 1) // 2, kernel_shape[-3] // 2)
    return image_pad, kernel_pad, output_pad

''' 2D inverse Fourier transform '''
def ifft(real_imag, ndims=2, normalized=False, pad=None):
    transformed = th.ifft(real_imag, ndims, normalized=normalized)
    return transformed

def irfft(input_complex, signal_sizes=None, ndims=2, normalized=False, pad=None):
    transformed = rfftshift(th.irfft(input_complex, ndims,
                                        signal_sizes=signal_sizes,
                                        normalized=normalized, onesided=True), ndims)
    return transformed

''' Complex division '''
def div_complex(field1, field2):
    real1, imag1 = unstack_complex(field1)
    real2, imag2 = unstack_complex(field2)

    mag_squared = (real2 ** 2) + (imag2 ** 2)

    real = th.true_divide(real1 * real2 + imag1 * imag2,mag_squared)
    imag = th.true_divide(-real1 * imag2 + imag1 * real2,mag_squared)

    return stack_complex(real, imag)

def div_com_real(field1, im_real):
    real1, imag1 = unstack_complex(field1)

    real = th.true_divide(real1 ,im_real)
    imag = th.true_divide(imag1 ,im_real)

    return stack_complex(real, imag)
