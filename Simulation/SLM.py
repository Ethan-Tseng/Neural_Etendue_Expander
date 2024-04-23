import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from Freq import freq_filt

import os

def get_slm_phase(freq_target, expander_amp, expander_phase, slm_amp, H_torch, batch_size, iterations, lr, R, C, upsample_factor, wvlpad_R, wvlpad_C):

    slm_phase = Variable(2*np.pi*torch.rand(batch_size,1,R,C,device='cuda:0'), requires_grad=True)
    optimizer = torch.optim.Adam([slm_phase], lr=lr)

    for i in range(iterations):
        if i % 100 == 0:
            print(i)

        slm_phase_resize = F.interpolate(slm_phase, scale_factor=(upsample_factor,upsample_factor), mode='nearest')
        slm_amp_resize = torch.ones_like(slm_phase_resize)
        
        field_amp = slm_amp_resize * expander_amp[None,None,:,:]
        field_phase = slm_phase_resize + expander_phase[None,None,:,:]
        field = torch.polar(field_amp, field_phase)
        img_field = torch.fft.fftshift(torch.fft.fft2(field))
        
        intensity = torch.square(torch.abs(img_field))
        intensity = torch.mean(intensity, axis=(0,1))
        
        if wvlpad_R > 0:
            assert(wvlpad_R == wvlpad_C)
            intensity = intensity * torch.mean(freq_target[wvlpad_R:-wvlpad_R,wvlpad_C:-wvlpad_C]) / torch.mean(intensity[wvlpad_R:-wvlpad_R,wvlpad_C:-wvlpad_C])
        else:
            intensity = intensity * torch.mean(freq_target) / torch.mean(intensity)
    
        error = torch.square(freq_filt(intensity, H_torch) - freq_target)
    
        if wvlpad_R > 0:
            error = error[wvlpad_R:-wvlpad_R,wvlpad_C:-wvlpad_C]
        loss = torch.mean(error)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return slm_phase
