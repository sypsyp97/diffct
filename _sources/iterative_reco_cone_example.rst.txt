Cone Beam Iterative Reconstruction
==================================

This example demonstrates gradient-based iterative reconstruction for 3D cone beam CT using the differentiable `ConeProjectorFunction` from `diffct`.

Overview
--------

3D cone beam iterative reconstruction extends optimization methods to full volumetric reconstruction. This example shows how to:

- Formulate 3D cone beam CT reconstruction as a large-scale optimization problem
- Handle the computational complexity of 3D forward and backward projections
- Apply memory-efficient optimization strategies for volumetric data
- Monitor convergence in high-dimensional parameter space

Mathematical Background
-----------------------

**3D Cone Beam Iterative Formulation**

The 3D reconstruction problem is formulated as:

.. math::
   \hat{f} = \arg\min_f \|A_{\text{cone}}(f) - p\|_2^2 + \lambda R(f)

where:
- :math:`f(x,y,z)` is the unknown 3D volume
- :math:`A_{\text{cone}}` is the cone beam forward projection operator
- :math:`p(\phi, u, v)` is the measured 2D projection data
- :math:`R(f)` is an optional 3D regularization term

**3D Forward Projection**

The cone beam forward projection integrates along rays through the 3D volume:

.. math::
   p(\phi, u, v) = \int_0^{\infty} f\left(\vec{r}_s(\phi) + t \cdot \vec{d}(\phi, u, v)\right) dt

where :math:`\vec{r}_s(\phi)` is the source position and :math:`\vec{d}(\phi, u, v)` is the ray direction vector.

**3D Gradient Computation**

The gradient of the 3D loss function uses the cone beam backprojection operator:

.. math::
   \frac{\partial L}{\partial f} = 2A_{\text{cone}}^T(A_{\text{cone}}(f) - p_{\text{measured}})

where :math:`A_{\text{cone}}^T` is the 3D cone beam backprojection operator (adjoint).

**Computational Complexity**

3D reconstruction presents significant computational challenges:

- **Memory Requirements**: :math:`O(N^3)` for volume storage vs :math:`O(N^2)` for 2D images
- **Projection Data**: :math:`O(N_{\phi} \times N_u \times N_v)` 2D projections
- **Forward/Backward Operations**: :math:`O(N^3 \times N_{\phi})$ computational complexity
- **Gradient Storage**: Additional memory for automatic differentiation

**Implementation Steps**

1. **3D Problem Setup**: Define parameterized 3D volume as learnable tensor
2. **Cone Beam Forward Model**: Use `ConeProjectorFunction` for 2D projection prediction
3. **Loss Computation**: Calculate L2 distance between predicted and measured projections
4. **3D Gradient Computation**: Use automatic differentiation through cone beam operators
5. **Memory-Efficient Optimization**: Apply strategies to handle large 3D parameter space
6. **Convergence Monitoring**: Track loss and 3D reconstruction quality

**Model Architecture**

The 3D reconstruction model consists of:

- **Parameterized Volume**: Learnable 3D tensor representing the unknown volume
- **Cone Beam Forward Model**: `ConeProjectorFunction` with 3D geometry parameters
- **Loss Function**: Mean squared error between predicted and measured 2D projections

**3D Regularization Options**

Common 3D regularization terms:

1. **3D Total Variation**: :math:`R_{\text{TV}}(f) = \sum_{x,y,z} \|\nabla f(x,y,z)\|_2`
2. **3D Smoothness**: :math:`R_{\text{smooth}}(f) = \sum_{x,y,z} \|\nabla f(x,y,z)\|_2^2`
3. **L1 Sparsity**: :math:`R_{\text{L1}}(f) = \sum_{x,y,z} |f(x,y,z)|`

**Memory Management Strategies**

3D reconstruction requires careful memory management:

- **Gradient Checkpointing**: Trade computation for memory in backpropagation
- **Mixed Precision**: Use float16 when possible to reduce memory usage
- **Batch Processing**: Process volume slices when memory is extremely limited
- **Efficient Data Layout**: Optimize tensor storage and access patterns

**Convergence Characteristics**

3D cone beam reconstruction typically exhibits:

1. **Initial Convergence** (0-100 iterations): Rapid loss decrease, basic 3D structure emerges
2. **Detail Refinement** (100-500 iterations): Fine 3D features develop progressively
3. **Final Convergence** (500+ iterations): Slow improvement, potential overfitting risk

**Challenges in 3D Reconstruction**

- **Cone Beam Artifacts**: Increased artifacts for large cone angles in 3D
- **Incomplete Sampling**: Missing data in certain regions of 3D Fourier space
- **Computational Cost**: Orders of magnitude higher than 2D reconstruction
- **Memory Limitations**: Large volumes may exceed available GPU memory
- **Convergence Complexity**: Higher-dimensional optimization landscape

**Applications**

3D cone beam iterative reconstruction is essential for:

- **Medical CBCT**: Dental, orthopedic, and interventional imaging
- **Industrial CT**: Non-destructive testing and quality control
- **Micro-CT**: High-resolution imaging of small specimens and materials
- **Security Screening**: Advanced baggage and cargo inspection systems

.. literalinclude:: ../../examples/iterative_reco_cone.py
   :language: python
   :linenos:
   :caption: 3D Cone Beam Iterative Example
