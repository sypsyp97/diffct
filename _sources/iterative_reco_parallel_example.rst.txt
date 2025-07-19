Parallel Beam Iterative Reconstruction
=====================================

This example demonstrates gradient-based iterative reconstruction for 2D parallel beam CT using the differentiable `ParallelProjectorFunction` from `diffct`.

Overview
--------

Iterative reconstruction methods solve the CT inverse problem through optimization, offering advantages over analytical methods like FBP. This example shows how to:

- Formulate CT reconstruction as an optimization problem
- Use automatic differentiation for gradient computation
- Apply advanced optimizers (Adam) for iterative reconstruction
- Monitor convergence and reconstruction quality

Mathematical Background
-----------------------

**Iterative Reconstruction Formulation**

CT reconstruction is formulated as an optimization problem:

.. math::
   \hat{f} = \arg\min_f \|A(f) - p\|_2^2 + \lambda R(f)

where:
- :math:`f` is the unknown 2D image
- :math:`A` is the forward projection operator (Radon transform)
- :math:`p` is the measured sinogram data
- :math:`R(f)` is an optional regularization term
- :math:`\lambda` is the regularization parameter

**Gradient-Based Optimization**

The gradient of the data fidelity term is computed using the adjoint operator:

.. math::
   \nabla_f \|A(f) - p\|_2^2 = 2A^T(A(f) - p)

where :math:`A^T` is the backprojection operator (adjoint of the forward projector).

**Automatic Differentiation**

PyTorch's automatic differentiation computes gradients through the differentiable operators:

.. math::
   \frac{\partial L}{\partial f} = \frac{\partial}{\partial f} \|A(f) - p_{\text{measured}}\|_2^2

This enables seamless integration with advanced optimizers like Adam.

**Adam Optimizer**

The Adam optimizer adapts learning rates using gradient statistics:

.. math::
   f^{(k+1)} = f^{(k)} - \alpha \cdot \frac{m^{(k)}}{1-\beta_1^k} \cdot \frac{1}{\sqrt{v^{(k)}/(1-\beta_2^k)} + \epsilon}

where :math:`m^{(k)}` and :math:`v^{(k)}` are biased first and second moment estimates.

**Implementation Steps**

1. **Problem Setup**: Define parameterized image as learnable tensor
2. **Forward Model**: Compute predicted sinogram using `ParallelProjectorFunction`
3. **Loss Computation**: Calculate L2 distance between predicted and measured data
4. **Gradient Computation**: Use automatic differentiation for gradient calculation
5. **Parameter Update**: Apply Adam optimizer for iterative improvement
6. **Convergence Monitoring**: Track loss and reconstruction quality

**Model Architecture**

The reconstruction model consists of:

- **Parameterized Image**: Learnable 2D tensor representing the unknown image
- **Forward Projection**: `ParallelProjectorFunction` for sinogram prediction
- **Loss Function**: Mean squared error between predicted and measured sinograms

**Advantages of Iterative Methods**

- **Noise Robustness**: Superior handling of noisy measurements
- **Regularization**: Natural incorporation of prior knowledge
- **Incomplete Data**: Effective with limited-angle or sparse-view acquisitions
- **Flexibility**: Easy modification of cost functions and constraints
- **Artifact Reduction**: Better control over reconstruction artifacts

**Convergence Characteristics**

Typical convergence behavior:

1. **Initial Phase** (0-100 iterations): Rapid loss decrease, basic structure emerges
2. **Refinement Phase** (100-500 iterations): Fine details develop, slower convergence  
3. **Convergence Phase** (500+ iterations): Minimal improvement, potential overfitting

.. literalinclude:: ../../examples/iterative_reco_parallel.py
   :language: python
   :linenos:
   :caption: 2D Parallel Beam Iterative Example
