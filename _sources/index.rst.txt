diffct: Differentiable CT Operators
===================================

diffct provides CUDA forward projectors and matched adjoints for 2D parallel,
2D fan and 3D cone beams, with per-view acquisition geometry and PyTorch autograd.
Use the same ``Projector`` interface for circular scans, generated non-circular
trajectories and calibrated source/detector poses.

.. important::

   diffct 2.0 replaces the circular-orbit API of 1.x. See :doc:`migration`
   before you update code written for 1.x.

What the operator supports
--------------------------

- Cell-constant Siddon forward projection and its matched adjoint.
- Image and sinogram gradients, including second derivatives with fixed geometry.
- First-order gradients for trajectory tensors used in geometry calibration.
- View splitting across local GPUs or initialized distributed process groups.
  Each GPU still needs a full volume; distributed projections are rank-local.
- Separate analytical FBP/FDK helpers. The adjoint is not an inverse, and accepting
  a non-circular trajectory does not make FDK exact for that acquisition.

Start here
----------

.. toctree::
   :maxdepth: 2

   getting_started
   trajectories
   distributed
   migration
   examples
   api
   validation

Citation and license
--------------------

For software and technical-report citations, see the
`README <https://github.com/sypsyp97/diffct/blob/main/README.md#citation>`_.
The project uses the
`Apache 2.0 license <https://github.com/sypsyp97/diffct/blob/main/LICENSE>`_.
