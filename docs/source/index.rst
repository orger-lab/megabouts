.. image:: _static/images/logo_color_dark.png
   :align: center
   :width: 640px
   :class: only-dark

.. image:: _static/images/logo_color_white.png
   :align: center
   :width: 640px
   :class: only-light



.. rst-class:: heading-center

Welcome to Megabouts' documentation!
====================================
Accurate measurement of animal behavior is essential for neuroscience, as well as genetic and pharmacological screening. 
Megabouts is a Python toolbox designed to precisely quantify zebrafish larval locomotion. 
It supports locomotion analysis in both freely swimming and head-restrained conditions and is designed to facilitate the standardized quantification of behavior.

Please support the development of Megabouts by starring and/or watching the project on Github_!

Megabouts is designed for flexibility, allowing it to manage a variety of tracking configurations and ensuring consistent analysis across diverse experimental setups. 
Below are two examples demonstrating Megabouts in action, using the same algorithm to classify movements recorded under different tracking configuration.

.. tabs::

   .. tab:: Analysis of 'tail + trajectory' tracking at 700 fps
  
      .. image:: _static/images/VideoPreyCapture.gif
         :align: center
         :width: 800px
         :alt: High-resolution ethogram analysis

   .. tab:: Analysis of 'trajectory only tracking' at 25 fps
      .. image:: _static/images/VideoZebrabox.gif
         :align: center
         :width: 800px
         :alt: Low-resolution ethogram analysis

Installation and Setup
----------------------
The software is available on `GitHub <https://github.com/orger-lab/megabouts>`_. Please head over to the :doc:`Usage </usage>` tab to find step-by-step installation instructions and example use cases in the :doc:`Tutorials </tutorials>` tab.

.. note::
   Megabouts is under active development and might include breaking changes
   between versions. 


Publication
------------
The details for the method can be found in
`our paper on bioRxiv <https://www.biorxiv.org/content/10.1101/2024.09.14.613078v2>`_:

.. code-block:: bibtex

   @article {Jouary2024.09.14.613078,
      author = {Jouary, Adrien and Silva, Pedro T.M. and Laborde, Alexandre and Mata, J. Miguel and Marques, Joao C. and Collins, Elena and Peterson, Randall T. and Machens, Christian K. and Orger, Michael B.},
      title = {Megabouts: a flexible pipeline for zebrafish locomotion analysis},
      elocation-id = {2024.09.14.613078},
      year = {2024},
      doi = {10.1101/2024.09.14.613078},
      publisher = {Cold Spring Harbor Laboratory},
      abstract = {Accurate quantification of animal behavior is crucial for advancing neuroscience and for defining reliable physiological markers. We introduce Megabouts (megabouts.ai), a software package standardizing zebrafish larvae locomotion analysis across experimental setups. Its flexibility, achieved with a Transformer neural network, allows the classification of actions regardless of tracking methods or frame rates. We demonstrate Megabouts{\textquoteright} ability to quantify sensorimotor transformations and enhance sensitivity to drug-induced phenotypes through high-throughput, high-resolution behavioral analysis.Competing Interest StatementThe authors have declared no competing interest.},
      URL = {https://www.biorxiv.org/content/early/2024/11/28/2024.09.14.613078},
      eprint = {https://www.biorxiv.org/content/early/2024/11/28/2024.09.14.613078.full.pdf},
      journal = {bioRxiv}
   }

.. toctree::
   :hidden:

   Home <self>

Getting Started
------------
.. toctree::
   :maxdepth: 1
   
   usage
   tutorials
   api/index