3D Cone Beam Iterative Reconstruction
=====================================

This example demonstrates how to use the differentiable `ConeProjectorFunction` in an iterative reconstruction pipeline for 3D cone-beam geometry using gradient-based optimization.

Mathematical Background
-----------------------

**3D Cone Beam Iterative Reconstruction**

The 3D iterative reconstruction problem is formulated as:

.. math::
   \hat{f} = \arg\min_f \|A_{\text{cone}}(f) - p\|_2^2 + \lambda R(f)

where:
- :math:`f(x,y,z)` is the unknown 3D volume
- :math:`A_{\text{cone}}` is the cone beam forward projection operator
- :math:`p(\phi, u, v)` is the measured 2D projection data
- :math:`R(f)` is an optional 3D regularization term
- :math:`\lambda` is the regularization parameter

**3D Forward Projection**

The cone beam forward projection integrates along rays from the point source through the volume:

.. math::
   p(\phi, u, v) = \int_0^{\infty} f\left(\vec{r}_s(\phi) + t \cdot \vec{d}(\phi, u, v)\right) dt

where:
- :math:`\vec{r}_s(\phi) = D_s(\cos\phi, \sin\phi, 0)` is the source position
- :math:`\vec{d}(\phi, u, v)` is the normalized ray direction vector

**3D Gradient Computation**

The gradient of the 3D loss function is:

.. math::
   \frac{\partial L}{\partial f} = 2A_{\text{cone}}^T(A_{\text{cone}}(f) - p_{\text{measured}})

where :math:`A_{\text{cone}}^T` is the 3D cone beam backprojection operator.

**Computational Complexity**

3D cone beam reconstruction is significantly more computationally intensive than 2D:

- **Memory Requirements**: :math:`O(N^3)` for volume storage vs :math:`O(N^2)` for images
- **Projection Data**: :math:`O(N_{\phi} \times N_u \times N_v)` vs :math:`O(N_{\phi} \times N_u)`
- **Forward/Backward Operations**: :math:`O(N^3 \times N_{\phi})` complexity

**3D Regularization Options**

Common 3D regularization terms include:

1. **3D Total Variation**:
   
   .. math::
      R_{\text{TV}}(f) = \sum_{x,y,z} \sqrt{|\nabla_x f|^2 + |\nabla_y f|^2 + |\nabla_z f|^2}

2. **3D Smoothness**:
   
   .. math::
      R_{\text{smooth}}(f) = \sum_{x,y,z} (|\nabla_x f|^2 + |\nabla_y f|^2 + |\nabla_z f|^2)

3. **L1 Sparsity**:
   
   .. math::
      R_{\text{L1}}(f) = \sum_{x,y,z} |f(x,y,z)|

**Memory Management**

3D reconstruction requires careful memory management:

- **Batch Processing**: Process volume slices when memory is limited
- **Gradient Checkpointing**: Trade computation for memory in backpropagation
- **Mixed Precision**: Use float16 when possible to reduce memory usage

**Optimization Strategy**

- **Learning Rate**: 0.1 (may need adjustment based on volume size)
- **Optimizer**: AdamW with weight decay for implicit regularization
- **Iterations**: 1000 epochs (may need more for larger volumes)
- **Monitoring**: Track loss every 10 iterations to detect convergence

**Convergence Characteristics**

3D cone beam reconstruction typically shows:

1. **Initial Convergence** (0-100 iterations): Rapid loss decrease, basic 3D structure
2. **Detail Refinement** (100-500 iterations): Fine 3D features develop
3. **Final Convergence** (500+ iterations): Slow improvement, risk of overfitting

**Challenges in 3D Reconstruction**

- **Cone Beam Artifacts**: Increased artifacts for large cone angles
- **Incomplete Sampling**: Missing data in certain regions of 3D Fourier space
- **Computational Cost**: Significantly higher than 2D reconstruction
- **Memory Limitations**: Large volumes may not fit in GPU memory

**Applications**

3D cone beam iterative reconstruction is used in:

- **Medical Imaging**: CBCT for dental, orthopedic applications
- **Industrial CT**: Non-destructive testing, quality control
- **Micro-CT**: High-resolution imaging of small specimens
- **Security Screening**: Baggage inspection systems

.. literalinclude:: ../../examples/iterative_reco_cone.py
   :language: python
   :linenos:
   :caption: 3D Cone Beam Iterative Example
