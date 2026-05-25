import math
import torch
import numpy as np
import torch.fft
import utils2

"""
This is the script that is used for the wave propagation using the angular spectrum method (ASM). Refer to 
Goodman, Joseph W. Introduction to Fourier optics. Roberts and Company Publishers, 2005, for principle details.

This file is borrowed from the implementation of the following paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop Training. ACM TOG (SIGGRAPH Asia), 2020.
"""

############## 彩色
def propagation_ASM(u_in, feature_size, wavelength, z, linear_conv=False,
                    padtype='zero', return_H=False, precomped_H=None,
                    dtype=torch.float32):
    """
    完整的彩色通用角谱传播函数
    支持动态尺寸适配与多波长并行计算
    """
    device = u_in.device

    if isinstance(wavelength, (list, tuple, np.ndarray)):
        wavelen = torch.tensor(wavelength, dtype=dtype, device=device).view(-1, 1, 1)
    else:
        wavelen = wavelength
      
    if linear_conv:
        input_resolution = u_in.size()[-2:] 
        conv_size = [i * 2 for i in input_resolution]
        if padtype == 'zero':
            padval = 0
        elif padtype == 'median':
            padval = torch.median(torch.abs(u_in)).item()
        u_in = utils2.pad_image(u_in, conv_size, padval=padval, stacked_complex=False)

    if precomped_H is None:
        field_resolution = u_in.size()
        num_y, num_x = field_resolution[2], field_resolution[3]
        dy, dx = feature_size
        y, x = (dy * float(num_y), dx * float(num_x))

        fy = np.linspace(-1 / (2 * dy) + 0.5 / (2 * y), 1 / (2 * dy) - 0.5 / (2 * y), num_y)
        fx = np.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), num_x)
        FX, FY = np.meshgrid(fx, fy)
        
        FX_t = torch.tensor(FX, dtype=dtype, device=device)
        FY_t = torch.tensor(FY, dtype=dtype, device=device)


        HH = 2 * math.pi * torch.sqrt(torch.clamp(1 / wavelen**2 - (FX_t**2 + FY_t**2), min=0))
        H_exp = HH * z

        fy_max = 1 / np.sqrt((2 * z * (1 / y))**2 + 1) / wavelen
        fx_max = 1 / np.sqrt((2 * z * (1 / x))**2 + 1) / wavelen
        H_filter = (torch.abs(FX_t) < fx_max) & (torch.abs(FY_t) < fy_max)

        H_real = torch.cos(H_exp) * H_filter
        H_imag = torch.sin(H_exp) * H_filter
        H = torch.complex(H_real, H_imag)
        
        H = utils2.ifftshift(H)
        
        if H.dim() == 3:
            H = H.unsqueeze(0)
    else:
        H = precomped_H

    if return_H:
        return H

    U1 = torch.fft.fftn(utils2.ifftshift(u_in), dim=(-2, -1), norm='ortho')
    U2 = H * U1
    u_out = utils2.fftshift(torch.fft.ifftn(U2, dim=(-2, -1), norm='ortho'))

    if linear_conv:
        return utils2.crop_image(u_out, input_resolution, pytorch=True, stacked_complex=False)
    else:
        return u_out


# ############### 单色
# def propagation_ASM(u_in, feature_size, wavelength, z, linear_conv=True,
#                     padtype='zero', return_H=False, precomped_H=None,
#                     return_H_exp=False, precomped_H_exp=None,
#                     dtype=torch.float32):


#     if linear_conv:
#         # preprocess with padding for linear conv.
#         input_resolution = u_in.size()[-2:]
#         conv_size = [i * 2 for i in input_resolution]
#         if padtype == 'zero':
#             padval = 0
#         elif padtype == 'median':
#             padval = torch.median(torch.pow((u_in**2).sum(-1), 0.5))
#         u_in = utils2.pad_image(u_in, conv_size, padval=padval, stacked_complex=False)

#     if precomped_H is None and precomped_H_exp is None:
#         # resolution of input field, should be: (num_images, num_channels, height, width, 2)
#         field_resolution = u_in.size()

#         # number of pixels
#         num_y, num_x = field_resolution[2], field_resolution[3]

#         # sampling inteval size
#         dy, dx = feature_size

#         # size of the field
#         y, x = (dy * float(num_y), dx * float(num_x))

#         # frequency coordinates sampling
#         fy = np.linspace(-1 / (2 * dy) + 0.5 / (2 * y), 1 / (2 * dy) - 0.5 / (2 * y), num_y)
#         fx = np.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), num_x)

#         # momentum/reciprocal space
#         FX, FY = np.meshgrid(fx, fy)

#         # transfer function in numpy (omit distance)
#         HH = 2 * math.pi * np.sqrt(1 / wavelength**2 - (FX**2 + FY**2))

#         # create tensor & upload to device (GPU)
#         H_exp = torch.tensor(HH, dtype=dtype).to(u_in.device)

#         ###
#         # here one may iterate over multiple distances, once H_exp is uploaded on GPU

#         # reshape tensor and multiply
#         H_exp = torch.reshape(H_exp, (1, 1, *H_exp.size()))

#     # handle loading the precomputed H_exp value, or saving it for later runs
#     elif precomped_H_exp is not None:
#         H_exp = precomped_H_exp

#     if precomped_H is None:
#         # multiply by distance
#         H_exp = torch.mul(H_exp, z)

#         # band-limited ASM - Matsushima et al. (2009)
#         fy_max = 1 / np.sqrt((2 * z * (1 / y))**2 + 1) / wavelength
#         fx_max = 1 / np.sqrt((2 * z * (1 / x))**2 + 1) / wavelength
#         H_filter = torch.tensor(((np.abs(FX) < fx_max) & (np.abs(FY) < fy_max)).astype(np.uint8), dtype=dtype)

#         # get real/img components
#         H_real, H_imag = utils2.polar_to_rect(H_filter.to(u_in.device), H_exp)

#         H = torch.stack((H_real, H_imag), 4)
#         H = utils2.ifftshift(H)
#         H = torch.view_as_complex(H)
#     else:
#         H = precomped_H

#     # return for use later as precomputed inputs
#     if return_H_exp:
#         return H_exp
#     if return_H:
#         return H

#     U1 = torch.fft.fftn(utils2.ifftshift(u_in), dim=(-2, -1), norm='ortho')

#     U2 = H * U1

#     u_out = utils2.fftshift(torch.fft.ifftn(U2, dim=(-2, -1), norm='ortho'))

#     if linear_conv:
#         # return utils.crop_image(u_out, input_resolution) # using stacked version
#         return utils2.crop_image(u_out, input_resolution, pytorch=True, stacked_complex=False)  # using complex tensor
#     else:
#         return u_out
    
