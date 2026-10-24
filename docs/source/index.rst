:html_theme.sidebar_secondary.remove:

diffct
======

.. raw:: html

   <div class="landing-search">
     <form action="search.html" method="get" role="search">
       <input type="search" name="q" placeholder="Search the documentation" aria-label="Search the documentation" autocomplete="off" autofocus>
     </form>
     <div class="srch-mount"></div>
   </div>

diffct provides differentiable CUDA projectors and matched adjoints for computed
tomography (CT) in PyTorch. It supports 2D parallel and fan beams and 3D cone
and parallel beams. Each view can follow any trajectory, and autograd gives
gradients for the volume and for trajectory tensors. Detector pixels can follow
parameterized surfaces, with first-order gradients for their parameters. Views
can be split across several GPUs.

Code written for diffct 1.x must be updated; see :doc:`migration`.

Install
-------

.. include:: install_steps.inc

Backend, precision and solver contracts are in :doc:`execution`.

Quickstart
----------

Create the acquisition once. Then call ``project()`` and ``backproject()``.

.. tab-set::

   .. tab-item:: CUDA

      .. code-block:: python

         import torch
         from diffct import Projector, spiral_trajectory_3d

         trajectory = spiral_trajectory_3d(360, sid=320.0, sdd=512.0, z_range=40.0, n_turns=1.0,
                                           device="cpu")
         A = Projector(trajectory, volume_shape=(128, 128, 128), detector_shape=(384, 256),
                       detector_spacing=0.8)

         volume = torch.rand(A.volume_shape, device="cuda")
         sinogram = A.project(volume)        # (360, 384, 256) line integrals
         adjoint = A.backproject(sinogram)   # matched adjoint A^T

         x = torch.zeros_like(volume, requires_grad=True)
         loss = 0.5 * (A.project(x) - sinogram).square().sum()
         loss.backward()                     # x.grad = A^T (A x - y)

   .. tab-item:: CPU (PyTorch backend)

      .. code-block:: python

         import torch
         from diffct import Projector, spiral_trajectory_3d

         trajectory = spiral_trajectory_3d(60, sid=120.0, sdd=192.0, z_range=15.0, n_turns=1.0,
                                           device="cpu")
         A = Projector(trajectory, volume_shape=(48, 48, 48), detector_shape=(144, 96),
                       detector_spacing=0.8, backend="torch")

         volume = torch.rand(A.volume_shape)
         sinogram = A.project(volume)        # (60, 144, 96) line integrals
         adjoint = A.backproject(sinogram)   # matched adjoint A^T

         x = torch.zeros_like(volume, requires_grad=True)
         loss = 0.5 * (A.project(x) - sinogram).square().sum()
         loss.backward()                     # x.grad = A^T (A x - y)

See :doc:`quickstart` for shapes, devices and troubleshooting, :doc:`examples`
for complete reconstructions, and :doc:`performance` for multi-GPU timings.

Citation and license
--------------------

For software and technical-report citations, see the
`README <https://github.com/sypsyp97/diffct/blob/main/README.md#citation>`_.
The project uses the
`Apache 2.0 license <https://github.com/sypsyp97/diffct/blob/main/LICENSE>`_.
This site documents the source checkout, including
`unreleased APIs <https://github.com/sypsyp97/diffct/blob/main/CHANGELOG.md#unreleased>`_.

.. toctree::
   :hidden:

   getting_started
   user_guide
   examples
   api
   changelog
