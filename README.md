# Neural &Eacute;tendue Expander for Ultra-Wide-Angle High-Fidelity Holographic Display
### [Project Page](https://light.princeton.edu/neural-etendue-expander/) | [Paper](https://www.nature.com/articles/s41467-024-46915-3) | [Data](https://drive.google.com/drive/folders/1j_ImmMDodgRBSBG79e75j_WwkgAcY9JI?usp=drive_link)

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.10653321.svg)](https://doi.org/10.5281/zenodo.10653321)

[Ethan Tseng](https://ethan-tseng.github.io), [Grace Kuo](https://grace-kuo.com/), [Seung-Hwan Baek](https://sites.google.com/view/shbaek/), [Nathan Matsuda](https://www.nathanmatsuda.com/), [Andrew Maimone](https://scholar.google.com/citations?user=ZBkzYIwAAAAJ&hl=en), [Florian Schiffers](https://florianschiffers.com/), [Praneeth Chakravarthula](https://www.cs.unc.edu/~cpk/), [Qiang Fu](https://fuqiangx.github.io/), [Wolfgang Heidrich](https://vccimaging.org/People/heidriw/), [Douglas Lanman](https://scholar.google.com/citations?user=-qncsGYAAAAJ&hl=en), [Felix Heide](https://www.cs.princeton.edu/~fheide/)

This code implements a differentiable image formation and optimization model for neural &eacute;tendue expansion. The experimental and simulation results from the manuscript and the supplementary information are reproducible with this implementation. The proposed framework is implemented completely in PyTorch without dependency on third-party libraries.

Download the data from the [data repository](https://drive.google.com/drive/folders/1j_ImmMDodgRBSBG79e75j_WwkgAcY9JI?usp=drive_link) before running the code. The data is organized using the same directory structure as this repository. The size of the data repository is around 1 GB.

## Experimental Results
The experimental results shown in Figure 2 can be reproduced by running the notebooks in the 'Experimental' folder. The notebook reads the raw capture data and performs intensity scaling (equivalent to laser power scaling) on the data. The code reproduces the experimental results for both monochromatic and trichomatic holograms. Each color channel of the trichromatic holograms was captured separately. The code will combine the independent color channel data into a single RGB image. The comparisons against random expanders \[Kuo et al. 2020\] and conventional holograms \[Shi et al. 2021\] can also be reproduced from the notebooks. The reconstructed holograms will be displayed within the notebook. Please see the instructions in the notebooks for more details.

## Simulation Results
The simulation results shown in Figure 3 can be reproduced by running the notebooks in the 'Simulation' folder. The notebooks run the gradient-based optimization described in the manuscript that generates &eacute;tendue expanded holograms. These notebooks implement the inference step of the algorithm, that is, the neural &eacute;tendue expander has already been trained and only the SLM pattern is optimized for each test image. The comparisons against random expanders \[Kuo et al. 2020\] can also be reproduced from the notebooks. The simulated holograms will be displayed within the notebook. Please see the instructions in the notebooks for more details.

The expected runtime of the optimization increases for higher &eacute;tendue expansion ratios. For 4x &eacute;tendue expansion the entire optimization should be completed in less than a minute for a single target image on an Nvidia A100 GPU. For 64x &eacute;tendue expansion the entire optimization should be completed in a few minutes for a single target image on an Nvidia A100 GPU.

In addition, the virtual frequency analysis shown in Figure 3 can be reproduced by running the notebook 'Virtual_Freq.ipynb'. This notebook displays the virtual frequency of a dataset of natural images and the virtual frequency of the neural &eacute;tendue expanders.

## Requirements
This code has been tested with Python 3.10.10 using PyTorch 2.0.0 running on Linux with an Nvidia A100 GPU with 10 GB RAM. The tested CUDA version is 12.0. An environment.yml file is included that provides the conda environment that we used to run this code. The total installation time when installing with conda should be a few minutes.

We installed the following library packages to run this code:
```
PyTorch >= 2.0.0
Numpy
Scipy
matplotlib
jupyter-notebook
```

## Citation
If you find our work useful in your research, please cite:
```
@article{Tseng2024NeuralEtendueExpander,
  author={Ethan Tseng, Grace Kuo, Seung-Hwan Baek, Nathan Matsuda, Andrew Maimone, Florian Schiffers, Praneeth Chakravarthula, Qiang Fu, Wolfgang Heidrich, Douglas Lanman, Felix Heide},
  title={Neural \'{E}tendue Expander for Ultra-Wide-Angle High-Fidelity Holographic Display},
  journal={Nature Communications},
  year={2024},
  month={Apr},
  day={22},
  volume={15},
  number={1},
  pages={2907}
}
```

## Additional Information
For more information on holography and &eacute;tendue expansion check out [Holotorch](https://github.com/facebookresearch/holotorch) and our [SIGGRAPH 2022 course](https://sites.google.com/princeton.edu/neural-optics/).

## License
Our code is licensed under BSL-1. By downloading the software, you agree to the terms of this License.
