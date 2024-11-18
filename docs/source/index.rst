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


.. figure:: _static/images/VideoPreyCapture.gif
   :align: center
   :width: 800px
   :alt: High resolution ethogram analysis


Installation and Setup
----------------------
The software is available on `GitHub <https://github.com/orger-lab/megabouts>`_. Please head over to the :doc:`Usage </usage>` tab to find step-by-step installation instructions and example use cases in the :doc:`Tutorials </tutorials>` tab.

.. note::
   Megabouts is under active development and might include breaking changes
   between versions. If you use Megabouts in your work, we recommend double-checking
   your current version.


Publication
------------
The details for the method can be found in
`our paper on bioRxiv <https://www.biorxiv.org/content/10.1101/2024.09.14.613078v1>`_:

.. code-block:: bibtex

   @article{jouary2024megabouts,
   title={Megabouts: a flexible pipeline for zebrafish locomotion analysis},
   author={Jouary, Adrien and Laborde, Alexandre and Silva, Pedro T and Mata, J Miguel 
           and Marques, Joao C and Collins, Elena and Peterson, Randall T 
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