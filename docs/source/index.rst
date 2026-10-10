diffct: Differentiable CT Operators
===================================

diffct provides differentiable CUDA projectors and matched adjoints for computed
tomography (CT) in PyTorch. It supports 2D parallel and fan beams and 3D cone
beams. Each view can follow any trajectory, and autograd gives gradients for the
volume and for trajectory tensors. Detector pixels can follow parameterized
surfaces, with first-order gradients for their parameters. Views can be split
across several GPUs.

Code written for diffct 1.x must be updated; see :doc:`migration`.

Install
-------

.. code-block:: bash

   python -m pip install "diffct[cu12]"

Use ``diffct[cu13]`` for CUDA 13. The extra installs the CUDA compiler libraries
that Numba CUDA needs. A plain ``pip install diffct`` does not install them; use it
only if a CUDA Toolkit is already installed on the system.

Start here
----------

.. toctree::
   :maxdepth: 2

   getting_started
   trajectories
   multi_gpu
   examples
   api
   migration

Citation and license
--------------------

For software and technical-report citations, see the
`README <https://github.com/sypsyp97/diffct/blob/main/README.md#citation>`_.
The project uses the
`Apache 2.0 license <https://github.com/sypsyp97/diffct/blob/main/LICENSE>`_.
