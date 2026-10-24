diffct needs Python 3.10 or newer, and a Python version for which PyTorch
publishes wheels. Install PyTorch for your device first. Then print the CUDA
version of your PyTorch build:

.. code-block:: bash

   python -c "import torch; print(torch.version.cuda)"

Install diffct with the extra that matches this version. The extra installs the
CUDA compiler libraries that Numba CUDA needs.

.. tab-set::

   .. tab-item:: 13.x

      .. code-block:: bash

         pip install "diffct[cu13]"

   .. tab-item:: 12.x

      .. code-block:: bash

         pip install "diffct[cu12]"

   .. tab-item:: None (CPU or ROCm)

      .. code-block:: bash

         pip install diffct

      Use ``backend="torch"``. Importing diffct and using this backend require
      no CUDA runtime.

   .. tab-item:: From source

      .. code-block:: bash

         git clone https://github.com/sypsyp97/diffct.git
         cd diffct
         pip install -e ".[cu13]"    # or ".[cu12]", or "." without CUDA
         python examples/quickstart.py

      Most scripts in ``examples/`` need an NVIDIA GPU; ``plot_trajectory.py``
      and ``benchmark_trajectory_admission.py --backend torch`` run on CPU.

Check the installation:

.. code-block:: bash

   python -c "import torch, diffct; print(diffct.__version__); print(torch.cuda.is_available())"
