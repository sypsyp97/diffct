diffct: Differentiable CT Operators
===================================

diffct provides differentiable CUDA projectors and matched adjoints for computed
tomography (CT) in PyTorch. It supports 2D parallel and fan beams and 3D cone
and parallel beams. Each view can follow any trajectory, and autograd gives gradients for the
volume and for trajectory tensors. Detector pixels can follow parameterized
surfaces, with first-order gradients for their parameters. Views can be split
across several GPUs.

Code written for diffct 1.x must be updated; see :doc:`migration`.

This site documents the source checkout, including
`unreleased APIs <https://github.com/sypsyp97/diffct/blob/main/CHANGELOG.md#unreleased>`_.
Install from source for features not yet in a tagged release.

Install
-------

.. code-block:: bash

   python -m pip install "diffct[cu12]"

Use ``diffct[cu13]`` for CUDA 13. The extra installs the CUDA compiler libraries
that Numba CUDA needs. A plain ``pip install diffct`` does not install them; use it
for the PyTorch backend, or when a CUDA Toolkit is already present for native
execution. Backend, precision and solver contracts are in :doc:`execution`.

Start here
----------

.. toctree::
   :maxdepth: 2

   getting_started
   execution
   trajectories
   chunking
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
