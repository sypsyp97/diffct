User Guide
==========

These pages describe the contracts behind ``Projector`` and the reconstruction
APIs. Start with :doc:`getting_started` if you have not installed diffct yet.

.. grid:: 1 2 2 3
   :gutter: 3
   :class-container: feature-grid

   .. grid-item-card:: Trajectories and geometry
      :link: trajectories
      :link-type: doc

      Per-view trajectory tensors, built-in generators, custom scans, detector
      surfaces and geometry gradients.

   .. grid-item-card:: Execution and backends
      :link: execution
      :link-type: doc

      Backends, precision modes and solver contracts.

   .. grid-item-card:: Memory and chunking
      :link: chunking
      :link-type: doc

      Memory limits, spatial, view and pixel tiles, block stores and
      disk-backed volumes.

   .. grid-item-card:: Multi-GPU and multi-node
      :link: multi_gpu
      :link-type: doc

      Several GPUs in one process, or ``torchrun`` across nodes, with view,
      space or hybrid partitions.

   .. grid-item-card:: Differentiable FBP and FDK
      :link: analytical_learning
      :link-type: doc

      Analytical reconstruction with autograd, and learnable filters and
      weights.

   .. grid-item-card:: Migrating from 1.x
      :link: migration
      :link-type: doc

      Move code from the circular-orbit API of diffct 1.x to per-view
      trajectories.

.. toctree::
   :hidden:
   :maxdepth: 2

   trajectories
   execution
   chunking
   multi_gpu
   analytical_learning
   migration
