import numpy as np
import torch

def butterworth(Nx, Ny, sFac, order=5):
    M = Ny
    N = Nx
    ps_x = 1/Nx
    ps_y = 1/Ny
    [x,y]=np.meshgrid(np.linspace(-N/2+1,N/2,N), np.linspace(-M/2+1,M/2,M))
    fx=x/(ps_x*N)
    fy=y/(ps_y*M)
    r0 = np.sqrt((Ny/sFac) * (Nx/sFac) / np.pi)
    r2 = np.power(fy,2)+np.power(fx,2)
    H_butter = 1/(1 + np.power(r2/np.power(r0,2),order))
    return H_butter

def freq_filt(img, H):
    imgC = torch.polar(img,torch.zeros_like(img))
    imgC_FFT = torch.fft.fftshift(torch.fft.fft2(imgC))
    imgC_filtered = imgC_FFT * H
    imgC_ifft = torch.fft.ifft2(torch.fft.ifftshift(imgC_filtered))
    img_final = torch.abs(imgC_ifft)
    return img_final
