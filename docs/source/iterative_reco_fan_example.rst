Fan Beam Iterative Reconstruction
=================================

This example demonstrates gradient-based iterative reconstruction for 2D fan beam CT using the differentiable `FanProjectorFunction` from `diffct`.

Overview
--------

Fan beam iterative reconstruction extends the optimization approach to the more realistic fan beam geometry. This example shows how to:

- Formulate fan beam CT reconstruction as an optimization problem
- Handle geometric complexities of divergent ray geometry
- Apply gradient-based optimization with fan beam operators
- Compare convergence characteristics with parallel beam reconstruction

Mathematical Background
-----------------------

**Fan Beam Iterative Formulation**

The fan beam reconstruction problem is formulated as:

.. math::
   \hat{f} = \arg\min_f \|A_{\text{fan}}(f) - p\|_2^2 + \lambda R(f)

where :math:`A_{\text{fan}}` is the fan beam forward projection operator accounting for divergent ray geometry.

**Fan Beam Forward Model**

The fan beam projection operator maps 2D image :math:`f(x,y)` to sinogram :math:`p(\beta, u)`:

.. math::
   p(\beta, u) = \int_{\text{ray}} f(x,y) \, dl

where integration follows the ray from point source to detector element :math:`u` at source angle :math:`\beta`.

**Gradient Computation**

The gradient involves the fan beam backprojection operator (adjoint):

.. math::
   \frac{\partial L}{\partial f} = 2A_{\text{fan}}^T(A_{\text{fan}}(f) - p_{\text{measured}})

where :math:`A_{\text{fan}}^T` is the fan beam backprojection operator.

**Geometric Considerations**

Fan beam geometry introduces complexities compared to parallel beam:

- **Ray Divergence**: Non-parallel rays affect sampling density and conditioning
- **Magnification Effects**: Variable magnification across the field of view
- **Non-uniform Resolution**: Spatial resolution varies with distance from rotation center
- **Geometric Distortion**: Requires careful handling of coordinate transformations

**Implementation Steps**

1. **Geometry Setup**: Configure fan beam parameters (source distance, detector distance)
2. **Problem Formulation**: Define parameterized image and fan beam forward model
3. **Loss Computation**: Calculate L2 distance using `FanProjectorFunction`
4. **Gradient Computation**: Use automatic differentiation through fan beam operators
5. **Optimization**: Apply Adam optimizer with appropriate learning rate
6. **Convergence Monitoring**: Track reconstruction quality and loss evolution

**Model Architecture**

The fan beam reconstruction model consists of:

- **Parameterized Image**: Learnable 2D tensor representing the unknown image
- **Fan Beam Forward Model**: `FanProjectorFunction` with geometric parameters
- **Loss Function**: Mean squared error between predicted and measured sinograms

**Convergence Characteristics**

Fan beam reconstruction typically exhibits:

1. **Initial Convergence** (0-100 iterations): Rapid loss decrease, basic structure
2. **Detail Refinement** (100-500 iterations): Fine features develop, slower progress
3. **Final Convergence** (500+ iterations): Minimal improvement, convergence plateau

**Comparison with Parallel Beam**

Fan beam iterative reconstruction differs from parallel beam:

- **Computational Complexity**: Higher due to geometric calculations
- **Convergence Rate**: Potentially slower due to geometry effects
- **Artifact Characteristics**: Different artifact patterns from ray divergence
- **Spatial Resolution**: Non-uniform resolution requires careful interpretation

**Challenges and Solutions**

- **Conditioning**: Fan beam system matrix may have different conditioning properties
- **Geometric Artifacts**: Proper weighting and filtering help reduce artifacts
- **Parameter Tuning**: Learning rate may need adjustment for optimal convergence
- **Memory Usage**: Similar to parallel beam but with additional geometric computations

.. literalinclude:: ../../examples/iterative_reco_fan.py
   :language: python
   :linenos:
   :caption: 2D Fan Beam Iterative Example
