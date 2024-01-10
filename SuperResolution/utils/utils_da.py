from pyparsing import java_style_comment
import torch
import numpy as np

def extract_ampl_phase(fft_im):
    # fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    # fft_amp = torch.sqrt(fft_amp)
    # fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    # return fft_amp, fft_pha
    fft_amp = fft_im.real**2 + fft_im.imag**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im.imag, fft_im.real)
    return fft_amp, fft_pha

def low_freq_mutate( amp_src, amp_trg, L=0.1 ):
    _, _, h, w = amp_src.size()
    b = (np.floor(np.amin((h, w)) * L)).astype(int)     # get b
    amp_src[:, :, 0:b, 0:b] = amp_trg[:, :, 0:b, 0:b]      # top left
    amp_src[:, :, 0:b, w-b:w] = amp_trg[:, :, 0:b, w-b:w]    # top right
    amp_src[:, :, h-b:h, 0:b] = amp_trg[:, :, h-b:h, 0:b]    # bottom left
    amp_src[:, :, h-b:h, w-b:w] = amp_trg[:, :, h-b:h, w-b:w]  # bottom right
    return amp_src


def high_freq_mutate( amp_src, amp_trg, L=0.1 ):
    _, _, src_h, src_w = amp_src.size()
    _, _, trg_h, trg_w = amp_trg.size()
    src_c_h = np.floor(src_h/2.0).astype(int)
    src_c_w = np.floor(src_w/2.0).astype(int)
    b = (np.floor(np.amin((src_c_h, src_c_w)) * L)).astype(int)
    trg_c_h = np.floor(trg_h/2.0).astype(int)
    trg_c_w = np.floor(trg_w/2.0).astype(int)

    src_h1 = src_c_h - b
    src_h2 = src_c_h + b + 1
    src_w1 = src_c_w - b
    src_w2 = src_c_w + b + 1
    trg_h1 = trg_c_h - b
    trg_h2 = trg_c_h + b + 1
    trg_w1 = trg_c_w - b
    trg_w2 = trg_c_w + b + 1
    amp_src[:, :, src_h1:src_h2, src_w1:src_w2] = amp_trg[:, :, trg_h1:trg_h2, trg_w1:trg_w2]
    return amp_src


def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

    _, src_h, src_w = a_src.shape
    _, trg_h, trg_w = a_trg.shape
    src_c_h = np.floor(src_h/2.0).astype(int)
    src_c_w = np.floor(src_w/2.0).astype(int)
    b = (np.floor(np.amin((src_c_h, src_c_w)) * L)).astype(int)
    trg_c_h = np.floor(trg_h/2.0).astype(int)
    trg_c_w = np.floor(trg_w/2.0).astype(int)

    src_h1 = src_c_h - b
    src_h2 = src_c_h + b + 1
    src_w1 = src_c_w - b
    src_w2 = src_c_w + b + 1
    trg_h1 = trg_c_h - b
    trg_h2 = trg_c_h + b + 1
    trg_w1 = trg_c_w - b
    trg_w2 = trg_c_w + b + 1

    a_src[:, src_h1:src_h2, src_w1:src_w2] = a_trg[:, trg_h1:trg_h2, trg_w1:trg_w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src

def FDA_source_to_target_low(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    # print('in FDA')
    _, _, imgH, imgW = src_img.size()
    # get fft of both source and target
    fft_src = torch.fft.rfft2(src_img.clone(), s=[imgH, imgW], dim=(-2, -1))
    # ifft_src = torch.fft.irfft2(fft_src.clone())
    fft_trg = torch.fft.rfft2(trg_img.clone(), s=[imgH, imgW], dim=(-2, -1))
    # fft_src_rfft = torch.fft.rfft2(src_img.clone(), dim=(-2, -1))
    # fft_src_rfft = torch.stack((fft_src.real, fft_src.imag), -1)
    # fft_trg = torch.fft.rfft2(trg_img.clone(), dim=(-2, -1))
    # fft_trg_rfft = torch.stack((fft_trg.real, fft_trg.imag), -1)

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase(fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase(fft_trg.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate(amp_src.clone(), amp_trg.clone(), L=L)

    # recompose fft of source
    fft_src_ = torch.zeros(fft_src.size(), dtype=torch.float)
    fft_src_real = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_imag = torch.sin(pha_src.clone()) * amp_src_.clone()
    fft_src_ = torch.complex(fft_src_real, fft_src_imag)
    # fft_src_ = torch.cos(pha_src.clone()) * amp_src_.clone() + torch.sin(pha_src.clone()) * amp_src_.clone()
    # # fft_src_.real = torch.cos(pha_src.clone()) * amp_src_.clone()
    # fft_src_.imag = torch.sin(pha_src.clone()) * amp_src_.clone()

    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.fft.irfft2(fft_src_)
    # test = torch.fft.ifft2(fft_src_)
    # src_in_trg = test.real

    return src_in_trg


def FDA_source_to_target_high(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    # print('in FDA')
    _, _, imgH, imgW = src_img.size()
    # get fft of both source and target
    fft_src = torch.fft.rfft2(src_img.clone(), s=[imgH, imgW], dim=(-2, -1))
    # ifft_src = torch.fft.irfft2(fft_src.clone())
    fft_trg = torch.fft.rfft2(trg_img.clone(), s=[imgH, imgW], dim=(-2, -1))
    # fft_src_rfft = torch.fft.rfft2(src_img.clone(), dim=(-2, -1))
    # fft_src_rfft = torch.stack((fft_src.real, fft_src.imag), -1)
    # fft_trg = torch.fft.rfft2(trg_img.clone(), dim=(-2, -1))
    # fft_trg_rfft = torch.stack((fft_trg.real, fft_trg.imag), -1)

    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase(fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase(fft_trg.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = high_freq_mutate(amp_src.clone(), amp_trg.clone(), L=L)

    # recompose fft of source
    fft_src_ = torch.zeros(fft_src.size(), dtype=torch.float)
    fft_src_real = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_imag = torch.sin(pha_src.clone()) * amp_src_.clone()
    fft_src_ = torch.complex(fft_src_real, fft_src_imag)
    # fft_src_ = torch.cos(pha_src.clone()) * amp_src_.clone() + torch.sin(pha_src.clone()) * amp_src_.clone()
    # # fft_src_.real = torch.cos(pha_src.clone()) * amp_src_.clone()
    # fft_src_.imag = torch.sin(pha_src.clone()) * amp_src_.clone()

    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.fft.irfft2(fft_src_)
    # test = torch.fft.ifft2(fft_src_)
    # src_in_trg = test.real

    return src_in_trg


def FDA_source_to_target_np(src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img #.cpu().numpy()
    trg_img_np = trg_img #.cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2(src_img_np, axes=(-2, -1))
    fft_trg_np = np.fft.fft2(trg_img_np, axes=(-2, -1))

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np(amp_src, amp_trg, L=L)

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp(1j * pha_src)

    # get the mutated image
    src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    src_in_trg = np.real(src_in_trg)

    return src_in_trg

