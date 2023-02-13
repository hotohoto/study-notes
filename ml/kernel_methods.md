# Kernel methods

- kernel
  - $k(\mathbf{x}, \mathbf{x}^\prime) = \phi(\mathbf{x})^\mathsf{T}\cdot\phi(\mathbf{x})$
- kernel function
  - $\phi(x)$
    - feature space mapping
- kernel trick
  - kernel substitution
  - Can replace scalar product of inputs with kernel
- stationary kernel
  - $k(\mathbf{x}, \mathbf{x}^\prime) = k(\mathbf{x} - \mathbf{x}^\prime)$
- homogeneous kernel
  - $k(\mathbf{x}, \mathbf{x}^\prime) = k(\lVert \mathbf{x} - \mathbf{x}^\prime \rVert)$
- applications
  - kernel PCA
  - kernel SVM
  - Gaussian process
