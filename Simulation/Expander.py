import torch
from RI_wvl import height2phase

def get_expander_phase(expander_type, wvl, RI):
    if expander_type == 'random_4x':
        print('loading random_4x')
        expander_height = torch.load('./Expanders/random_4x.pth')
    elif expander_type == 'random_16x':
        print('loading random_16x')
        expander_height = torch.load('./Expanders/random_16x.pth')
    elif expander_type == 'random_36x':
        print('loading random_36x')
        expander_height = torch.load('./Expanders/random_36x.pth')
    elif expander_type == 'random_64x':
        print('loading random_64x')
        expander_height = torch.load('./Expanders/random_64x.pth')
    elif expander_type == 'neural_mono_4x':
        print('loading neural_mono_4x')
        expander_height = torch.load('./Expanders/neural_mono_4x.pth')
    elif expander_type == 'neural_tri_4x':
        print('loading neural_tri_4x')
        expander_height = torch.load('./Expanders/neural_tri_4x.pth')
    elif expander_type == 'neural_mono_16x':
        print('loading neural_mono_16x')
        expander_height = torch.load('./Expanders/neural_mono_16x.pth')
    elif expander_type == 'neural_tri_16x':
        print('loading neural_tri_16x')
        expander_height = torch.load('./Expanders/neural_tri_16x.pth')
    elif expander_type == 'neural_mono_36x':
        print('loading neural_mono_36x')
        expander_height = torch.load('./Expanders/neural_mono_36x.pth')
    elif expander_type == 'neural_tri_36x':
        print('loading neural_tri_36x')
        expander_height = torch.load('./Expanders/neural_tri_36x.pth')
    elif expander_type == 'neural_mono_64x':
        print('loading neural_mono_64x')
        expander_height = torch.load('./Expanders/neural_mono_64x.pth')
    elif expander_type == 'neural_tri_64x':
        print('loading neural_tri_64x')
        expander_height = torch.load('./Expanders/neural_tri_64x.pth')
    else:
        assert('Undefined expander.')
    expander_phase = height2phase(expander_height, wvl, RI)
    return expander_phase
