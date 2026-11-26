Installation
============

Native execution uses an NVIDIA CUDA GPU. The portable PyTorch backend also
supports CPU, PyTorch ROCm and Apple Silicon (MPS) devices; see
:doc:`execution` for precision and hardware-validation limits.

.. include:: install_steps.inc

Numba CUDA initializes on native use. The NVVM and NVJitLink packages must match
the CUDA libraries that PyTorch loads, so the extra must match
``torch.version.cuda``. A plain ``pip install diffct`` also works for native
execution if a matching CUDA Toolkit is already installed.

Troubleshooting
---------------

- **CUDA is unavailable:** check that the active Python environment has a CUDA
  build of PyTorch and that the GPU is visible to the process.
- **NumPy 2.5 compatibility:** Numba CUDA imports ``numpy.row_stack``, which
  NumPy 2.5 removes. Keep ``numpy<2.5`` until Numba CUDA no longer needs it.
- **Kernel compilation fails:** use Numba CUDA, NVVM and NVJitLink versions that
  match each other. Do not upgrade only one of them.
- **nvJitLink error:** ``nvJitLinkError`` with "may need newer version of
  nvJitLink library" means that the diffct extra does not match the CUDA
  version of PyTorch. Print
  ``torch.version.cuda`` and reinstall with ``diffct[cu12]`` for 12.x or
  ``diffct[cu13]`` for 13.x.

Tested CUDA combinations
------------------------

- CUDA 13:
  ``pip install "numpy<2.5" "numba-cuda[cu13]" "cuda-toolkit[cccl,cudart,nvrtc,nvvm]==13.0.2" "nvidia-nvjitlink<13.1"``
- CUDA 12: PyTorch 2.10 (cu126) with ``numba-cuda[cu12]`` 0.30.4 on NVIDIA
  driver 535.
