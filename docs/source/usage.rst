Usage
=====

This section provides instructions on how to obtain zebrafish tracking data and how to install and set up the Megabouts package for locomotion analysis.

Obtaining Zebrafish Tracking Data
---------------------------------
There are several tools and methods available for tracking zebrafish. Below are some popular options:

- **BonZeb**: `BonZeb Website <https://ncguilbeault.github.io/BonZeb/>`_ | `BonZeb Paper <https://www.nature.com/articles/s41598-021-85896-x>`_

- **Stytra**: `Stytra Website <https://portugueslab.com/stytra/>`_ | `Stytra Paper <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006699>`_

- **ZebraZoom**: `ZebraZoom Website <https://zebrazoom.org/>`_ | `ZebraZoom Paper <https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2013.00107/full>`_

If you are working with video recordings of zebrafish larvae, you can also track them using deep learning-based methods:

- **DeepLabCut**: `DeepLabCut Website <https://www.mackenziemathislab.org/deeplabcut>`_ | `DeepLabCut Paper <https://www.nature.com/articles/s41593-018-0209-y>`_

- **SLEAP**: `SLEAP Website <https://sleap.ai/>`_ | `SLEAP Paper <https://www.nature.com/articles/s41592-022-01426-1>`_

Once you have your zebrafish tracking data, you're ready to analyze it with Megabouts!"

Installing Megabouts
--------------------
Megabouts is an open-source Python toolbox designed for zebrafish locomotion analysis. To install Megabouts, follow the steps below:

1. **Create a Virtual Environment (using conda)**:
   First, create a virtual environment with Python 3.11 using `conda`:

   .. code-block:: bash

      conda create --name megabouts python=3.11

   Then activate the environment:

   .. code-block:: bash

      conda activate megabouts

2. **Install PyTorch [for GPU setup]**:
   Megabouts depends on PyTorch. To enable GPU support, follow the instructions on the `PyTorch website <https://pytorch.org/get-started/locally/>`_ to install the appropriate version for your system.

   Example installation command:

   .. code-block:: bash

      conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

   *Make sure to adjust the command based on your system's GPU or CPU configuration.*

3. **Once PyTorch is installed, you can install Megabouts**:
   Choose one of the following installation methods:

   From PyPI (stable version):

   .. code-block:: bash

      pip install megabouts

   From GitHub (latest development version):

   .. code-block:: bash

      pip install git+https://github.com/orger-lab/megabouts.git

4. **Verify the Installation**:
   After installation, you can verify that Megabouts is properly installed by checking its version:

   .. code-block:: bash

      python -c "import megabouts; print(megabouts.__version__)"

If you see the version number printed without errors, Megabouts has been successfully installed.

Usage Guide
-----------
Now that you've installed Megabouts, you can begin analyzing zebrafish locomotion data. Please refer to the :doc:`Tutorials </tutorials>` for detailed instructions and examples.
