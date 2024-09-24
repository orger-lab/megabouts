Zebrafish Tracking and Installation Guide
=========================================

This section provides instructions on how to obtain zebrafish tracking data and how to install and set up the Megabouts package for locomotion analysis.

Obtaining Zebrafish Tracking Data
---------------------------------
There are several tools and methods available for zebrafish tracking. Below are some popular options:

1. **BonZeb**:
   - **GitHub**: `BonZeb GitHub <https://github.com/BonZeb/BonZeb>`_
   - **Article**: You can read more about BonZeb in the following publication: `BonZeb Paper <https://doi.org/10.1038/s41598-020-72821-4>`_

2. **Stytra**:
   - **GitHub**: `Stytra GitHub <https://github.com/portugueslab/stytra>`_
   - **Article**: Learn more about Stytra in its accompanying article: `Stytra Paper <https://doi.org/10.1038/s41467-019-12201-2>`_

3. **ZebraZoom**:
   - **GitHub**: `ZebraZoom GitHub <https://github.com/oliviermirat/ZebraZoom>`_
   - **Article**: Detailed description of ZebraZoom can be found in its article: `ZebraZoom Paper <https://doi.org/10.1371/journal.pbio.2006950>`_

4. **Deep Learning-Based Methods**:
   - **DeepLabCut**:
     - **GitHub**: `DeepLabCut GitHub <https://github.com/DeepLabCut/DeepLabCut>`_
     - **Article**: Learn about DeepLabCut in its primary publication: `DeepLabCut Paper <https://doi.org/10.1038/s41592-018-0185-0>`_

   - **SLEAP**:
     - **GitHub**: `SLEAP GitHub <https://github.com/murthylab/sleap>`_
     - **Article**: Find the research behind SLEAP: `SLEAP Paper <https://doi.org/10.1038/s41592-019-0471-7>`_

Installing Megabouts
--------------------
Megabouts is an open-source Python toolbox designed for zebrafish locomotion analysis. To install Megabouts, follow the steps below:

1. **Create a Virtual Environment (using conda)**:
   First, create a virtual environment with Python 3.11 using `conda`:

   .. code-block:: bash

      conda create --name megabouts python=3.11

   Activate the environment:

   .. code-block:: bash

      conda activate megabouts

2. **Install PyTorch**:
   Since Megabouts depends on PyTorch, you need to install PyTorch first. Follow the instructions on the `PyTorch website <https://pytorch.org/get-started/locally/>`_ to install the appropriate version of PyTorch for your system.

   Example installation command:

   .. code-block:: bash

      conda install pytorch torchvision torchaudio cpuonly -c pytorch

   *(Make sure to adjust the command based on your system and GPU/CPU configuration.)*

3. **Install Megabouts**:
   After PyTorch is installed, you can install Megabouts from its GitHub repository:

   .. code-block:: bash

      pip install git+https://github.com/orger-lab/megabouts.git

4. **Verify the Installation**:
   After installation, you can verify that Megabouts is properly installed by running:

   .. code-block:: bash

      python -c "import megabouts"

If no errors are raised, Megabouts has been successfully installed.

Usage Guide
-----------
Now that you've installed Megabouts, you can begin analyzing zebrafish locomotion data. Please refer to the `User Guide <index.html>`_ for detailed tutorials and examples.
