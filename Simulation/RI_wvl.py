import numpy as np

RI_660 = 1.5081
RI_517 = 1.5159
RI_450 = 1.5223
wvl_660 = 660e-9
wvl_517 = 517e-9
wvl_450 = 450e-9

def height2phase(height, wvl, RI, wrap=False):
    dRI = RI - 1
    wv_n = 2. * np.pi / wvl
    phi = wv_n * dRI * height
    if wrap:
        phi %= 2 * np.pi
    return phi

def phase2height(phase, wvl, RI):
    dRI = RI - 1
    return (wvl * phase / (2 * np.pi)) / dRI
