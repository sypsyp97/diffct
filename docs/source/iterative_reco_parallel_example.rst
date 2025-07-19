2D Parallel Beam Iterative Reconstruction
=========================================

This example demonstrates how to use the differentiable `ParallelProjectorFunction` in an iterative reconstruction pipeline for 2D parallel-beam geometry using gradient-based optimization.

Mathematical Background
-----------------------

**Iterative Reconstruction Formulation**

Iterative reconstruction solves the inverse problem by minimizing a cost function. The basic formulation is:

.. math::
   \hat{f} = \arg\min_f \|Af - p\|_2^2 + R(f)

where:
- :math:`f` is the unknown image
- :math:`A` is the forward projection operator (Radon transform)
- :math:`p` is the measured sinogram data
- :math:`R(f)` is an optional regularization term
- :math:`\|\cdot\|_2^2` is the L2 norm (mean squared error)

**Gradient-Based Optimization**

The gradient of the data fidelity term with respect to the image is:

.. math::
   \nabla_f \|Af - p\|_2^2 = 2A^T(Af - p)

where :math:`A^T` is the backprojection operator (adjoint of the Radon transform).

**Automatic Differentiation**

Using PyTorch's automatic differentiation, the gradient is computed automatically:

.. math::
   \frac{\partial L}{\partial f} = \frac{\partial}{\partial f} \|A(f) - p_{\text{measured}}\|_2^2

This enables the use of advanced optimizers like Adam, which adapts the learning rate based on gradient statistics.

**Optimization Algorithm**

The example uses the Adam optimizer with the update rule:

.. math::
   f^{(k+1)} = f^{(k)} - \alpha \cdot \frac{m^{(k)}}{1-\beta_1^k} \cdot \frac{1}{\sqrt{v^{(k)}/(1-\beta_2^k)} + \epsilon}

where:
- :math:`m^{(k)}` is the first moment estimate (momentum)
- :math:`v^{(k)}` is the second moment estimate (RMSprop)
- :math:`\alpha` is the learning rate
- :math:`\beta_1, \beta_2` are exponential decay rates
- :math:`\epsilon` is a small constant for numerical stability

**Model Architecture**

The reconstruction model implements:

1. **Parameterized Image**: The unknown image :math:`f` as a learnable parameter
2. **Forward Model**: :math:`p_{\text{pred}} = A(f + f_{\text{initial}})`
3. **Loss Function**: :math:`L = \|p_{\text{pred}} - p_{\text{measured}}\|_2^2`

**Advantages of Iterative Methods**

- **Noise Robustness**: Better handling of noisy measurements
- **Regularization**: Can incorporate prior knowledge through regularization terms
- **Incomplete Data**: Works with limited-angle or sparse-view data
- **Flexibility**: Easy to modify cost function and constraints

**Expected Behavior**

The loss should decrease rapidly in the first 100-200 iterations, then converge more slowly. The reconstruction quality improves progressively, with fine details appearing as the optimization proceeds.

.. literalinclude:: ../../examples/iterative_reco_parallel.py
   :language: python
   :linenos:
   :caption: 2D Parallel Beam Iterative Example
