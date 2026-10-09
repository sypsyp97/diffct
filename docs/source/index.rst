diffct: Differentiable CT Operators
===================================

diffct provides CUDA forward projectors and matched adjoints for 2D parallel,
2D fan and 3D cone beams, with per-view acquisition geometry and PyTorch autograd.
Use the same ``Projector`` interface for circular scans, generated non-circular
trajectories and calibrated source/detector poses.

.. important::

   These sources describe the candidate branch
   ``codex/arbitrary-trajectory-multigpu``. Install that checkout to use
   ``Projector``. The published PyPI release and documentation deployed from
   ``main`` are separate versions; a documentation badge is not a branch CI result.

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
`branch README <https://github.com/sypsyp97/diffct/blob/codex/arbitrary-trajectory-multigpu/README.md#citation>`_.
The project uses the
`Apache 2.0 license <https://github.com/sypsyp97/diffct/blob/codex/arbitrary-trajectory-multigpu/LICENSE>`_.
