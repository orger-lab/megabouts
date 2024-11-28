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
Accurate measurements of animal behavior are crucial for neuroscience, as well as genetic and pharmacological screening. Megabouts is a Python toolbox designed to precisely quantify zebrafish larval locomotion.
Our software supports locomotion analysis in both freely swimming and head-restrained conditions, 
with the goal of promoting standardized zebrafish locomotion studies across neuroscience, pharmacology, and genetics.

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
`our paper on bioRxiv <https://www.biorxiv.org/content/10.1101/2024.09.14.613078v1>`_:

.. code-block:: bibtex

   @article{jouary2024megabouts,
   title={Megabouts: a flexible pipeline for zebrafish locomotion analysis},
   author={Jouary, Adrien and Silva, Pedro TM and Laborde, Alexandre and Mata, J Miguel 
           and Marques, Joao C and Collins, Elena MD and Peterson, Randall T 
           and Machens, Christian K and Orger, Michael B},
   journal={bioRxiv},
   pages={2024--09},
   year={2024},
   publisher={Cold Spring Harbor Laboratory}
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