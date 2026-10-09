Parallel Beam Iterative Reconstruction
======================================

This page shows a 2D parallel-beam reconstruction loop with ``diffct.Projector``.
The maintained iterative script is ``examples/iterative_reconstruction.py``, which
reconstructs 3D cone-beam scans. See :doc:`iterative_reco_cone_example`.

Objective
---------

The loop minimizes a least-squares data term over a nonnegative image ``x``,
where ``A`` is the projector and ``y`` is the measured sinogram:

.. math::
   \hat{x} = \arg\min_{x \ge 0} \tfrac{1}{2} \lVert A x - y \rVert_2^2

Autograd computes the gradient, which uses the adjoint ``A^T``. Adam updates the
image.

Code
----

.. code-block:: python

   import torch
   from diffct import Projector, circular_trajectory_2d_parallel

   n, n_views, n_det = 128, 360, 384
   trajectory = circular_trajectory_2d_parallel(n_views, device="cuda")
   A = Projector(trajectory, (n, n), n_det, beam="parallel", detector_spacing=0.5)

   truth = torch.zeros((n, n), device="cuda")
   truth[40:88, 40:88] = 1.0
   measured = A.project(truth).detach()

   image = torch.zeros((n, n), device="cuda", requires_grad=True)
   optimizer = torch.optim.Adam([image], lr=1e-2)
   for step in range(200):
       optimizer.zero_grad()
       loss = 0.5 * (A.project(image) - measured).square().sum()
       loss.backward()
       optimizer.step()
       with torch.no_grad():
           image.clamp_(min=0)

Notes
-----

- The learning rate and iteration count are starting values. Tune them for your
  data.
- For a real measurement, choose a stopping rule that does not use a ground-truth
  phantom.
- Use the same ``Projector`` interface for other 2D trajectories, such as
  ``custom_trajectory_2d_parallel``.
