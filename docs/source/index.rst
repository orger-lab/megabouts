.. image:: logo_color_dark.png
   :align: center
   :width: 640px
   :class: only-dark

.. image:: logo_color_white.png
   :align: center
   :width: 640px
   :class: only-light

.. rst-class:: heading-center

Welcome to Megabouts' documentation!
====================================

Accurate measurements of behavior are crucial to understanding animal physiology. 
Megabouts allows to quantify zebrafish larvae locomotion. 
This open-source Python toolbox handles locomotion in both freely swimming and head-restrained conditions and aims to promote standardized zebrafish locomotion analysis across neuroscience, pharmacology, and genetics.
For more details, check `our paper on bioRxiv <https://www.biorxiv.org/content/10.1101/2024.09.14.613078v1>`_

Please support the development of Megabouts by starring and/or watching the project on Github_!


.. note::
   Megabouts is under active development and might include breaking changes
   between versions. If you use Megabouts in your work, we recommend to double check
   your current version.



Installation and Setup
----------------------

Please see the dedicated :doc:`Installation Guide </installation>` for information on installation options using ``conda``, ``pip`` and ``docker``.

Have fun! 😁
---

Publication:
------------
The details for the underlying mathematics and methods can be found in our publication:
`our paper on bioRxiv <https://www.biorxiv.org/content/10.1101/2024.09.14.613078v1>`_:

   **Jouary, A et al.**  
   *Megabouts: a flexible pipeline for zebrafish locomotion analysis*,  
   **bioRxiv**, 2024.

---

Find the Software:
------------------
You can find the software and source code on `GitHub <https://github.com/orger-lab/megabouts>`_




Usage
-----

Please head over to the :doc:`Usage </usage>` tab to find step-by-step instructions to use CEBRA on your data. For example use cases, see the :doc:`Demos </demos>` tab.

**Installation**

PyPI install, presuming you have pytorch and all its requirements installed:


.. code:: bash

    pip install megabouts

References
----------
.. code::

   @article{jouary2024megabouts,
   title={Megabouts: a flexible pipeline for zebrafish locomotion analysis},
   author={Jouary, Adrien and Laborde, Alexandre and Silva, Pedro T and Mata, J Miguel and Marques, Joao C and Collins, Elena and Peterson, Randall T and Machens, Christian K and Orger, Michael B},
   journal={bioRxiv},
   pages={2024--09},
   year={2024},
   publisher={Cold Spring Harbor Laboratory}
   }


.. toctree::
   :hidden:

   Home <self>


.. toctree::
   :maxdepth: 1
   :caption: User Guide / Tutorial:

   notebooks/Loading_Data
   preprocessing
   segmentation
   notebooks/tutorial_Tail_Classification
   api/index
