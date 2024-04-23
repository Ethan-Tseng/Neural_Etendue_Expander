import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from Freq import freq_filt

def get_intensity(freq_target, expander_amp, expander_phase, slm_amp, slm_phase, H_torch, \
                  batch_size, R, C, upsample_factor, wvlpad_R, wvlpad_C):
    
    # Overlay SLM and Diffuser
    field_amp_out = slm_amp * expander_amp[None,None,:,:]
    field_phase_out = slm_phase + expander_phase[None,None,:,:]
    field_out = torch.polar(field_amp_out, field_phase_out)

    # Forward model
    img_out = torch.fft.fftshift(torch.fft.fft2(field_out))
    img_out = torch.square(torch.abs(img_out))
    img_out = torch.mean(img_out, axis=0, keepdims=True)
    img_out = freq_filt(img_out, H_torch)

    # Power scaling
    if wvlpad_R > 0:
        assert(wvlpad_R == wvlpad_C)
        img_out = img_out * torch.mean(freq_target[wvlpad_R:-wvlpad_R,wvlpad_C:-wvlpad_C]) / torch.mean(img_out[:,:,wvlpad_R:-wvlpad_R,wvlpad_C:-wvlpad_C])
    else:
        img_out = img_out * torch.mean(freq_target) / torch.mean(img_out)

    # Calculate loss value
    error = torch.square(torch.clip(img_out,0.0,1.0) - freq_target)
    if wvlpad_R > 0:
        error = error[:,:,wvlpad_R:-wvlpad_R,wvlpad_C:-wvlpad_C]
    loss = torch.mean(error)
    
    # Wavelength scaling
    if wvlpad_R > 0:
        img_out = img_out[:,:,wvlpad_R:-wvlpad_R,wvlpad_C:-wvlpad_C]
        img_out = F.interpolate(img_out, size=(R * upsample_factor, C * upsample_factor), mode='nearest')
    
    return loss, img_out
