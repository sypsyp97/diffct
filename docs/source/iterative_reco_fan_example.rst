2D Fan Beam Iterative Reconstruction
====================================

This example demonstrates how to use the differentiable `FanProjectorFunction` in an iterative reconstruction pipeline for 2D fan-beam geometry using gradient-based optimization.

Mathematical Background
-----------------------

**Fan Beam Iterative Reconstruction**

The iterative reconstruction problem for fan beam geometry is formulated as:

.. math::
   \hat{f} = \arg\min_f \|A_{\text{fan}}(f) - p\|_2^2 + R(f)


where :math:`A_{\text{fan}}` is the fan beam forward projection operator.

**Fan Beam Forward Model**

The fan beam projection operator maps the 2D image :math:`f(x,y)` to sinogram :math:`p(\beta, u)`:

.. math::
   p(\beta, u) = \int_{\text{ray}} f(x,y) \, dl

where the integration is along the ray from source to detector element :math:`u` at angle :math:`\beta`.

**Gradient Computation**

The gradient of the loss function involves the adjoint (backprojection) operator:

.. math::
   \frac{\partial L}{\partial f} = 2A_{\text{fan}}^T(A_{\text{fan}}(f) - p_{\text{measured}})

where :math:`A_{\text{fan}}^T` is the fan beam backprojection operator.

**Geometry-Specific Considerations**

Fan beam geometry introduces additional complexity compared to parallel beam:

1. **Ray Divergence**: Rays diverge from a point source, affecting sampling density
2. **Magnification**: Objects closer to the source appear larger on the detector
3. **Geometric Distortion**: Non-uniform spatial resolution across the field of view

**Optimization Challenges**

Fan beam iterative reconstruction faces unique challenges:

- **Conditioning**: The system matrix has different conditioning than parallel beam
- **Convergence**: May require different learning rates due to geometry effects
- **Artifacts**: Geometric artifacts can appear if not properly handled

**Model Architecture**

The fan beam reconstruction model includes:

1. **Parameterized Image**: Learnable 2D image parameters
2. **Fan Beam Forward Model**: Uses `FanProjectorFunction` with geometry parameters
3. **Loss Computation**: L2 distance between predicted and measured sinograms

**Optimization Strategy**

- **Learning Rate**: 0.1 (same as parallel beam, but may need adjustment)
- **Optimizer**: AdamW for adaptive learning rate and weight decay
- **Iterations**: 1000 epochs with progress monitoring
- **Initialization**: Zero image (uniform background)

**Expected Convergence**

Fan beam reconstruction typically shows:

1. **Initial Phase** (0-50 iterations): Rapid decrease in loss, basic structure emerges
2. **Refinement Phase** (50-200 iterations): Fine details develop, slower convergence
3. **Convergence Phase** (200+ iterations): Minimal improvement, potential overfitting

**Comparison with Parallel Beam**

Fan beam iterative reconstruction differs from parallel beam in:

- **Computational Cost**: Slightly higher due to geometric calculations
- **Convergence Rate**: May be slower due to more complex geometry
- **Artifact Patterns**: Different artifact characteristics
- **Spatial Resolution**: Non-uniform resolution across the field of view

.. literalinclude:: ../../examples/iterative_reco_fan.py
   :language: python
   :linenos:
   :caption: 2D Fan Beam Iterative Example
