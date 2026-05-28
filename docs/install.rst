Installation instructions
=========================

Prerequisites
^^^^^^^^^^^^^

First, create a minimal environment with only Python installed in it. We recommend using `Python virtual environments <https://docs.python.org/3.10/tutorial/venv.html>`__.

.. code:: sh

    python3 -m venv /path/to/virtual/environment

Activate your new environment.

.. code:: sh

    source /path/to/virtual/environment/bin/activate


.. _Option 1:

There are two primary ways to install the packages:

- :ref:`Option 1`
- :ref:`Option 2`

Option 1: Install from PyPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most straightforward way to install the ``qiskit-addon-pna`` package is via ``PyPI``.

.. code:: sh

    pip install 'qiskit-addon-pna'


.. _Option 2:

Option 2: Install from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you plan to develop in the repository or run the notebooks locally, you should install from source.

First, clone the ``qiskit-addon-pna`` repository.

.. code:: sh

    git clone git@github.com:Qiskit/qiskit-addon-pna.git

Next, upgrade pip and enter the repository.

.. code:: sh

    pip install --upgrade pip
    cd qiskit-addon-pna

The next step is to install ``qiskit-addon-pna`` to the virtual environment. If you plan to run the notebooks and their visualizations, install the
notebook dependencies.
If you plan on developing in the repository, install the ``dev`` dependencies.

Adjust the options below to suit your needs.

.. code:: sh

    pip install tox notebook -e '.[notebook-dependencies,dev]'

If you installed the notebook dependencies, you can get started by running the notebooks in the docs.

.. code::

    cd docs/
    jupyter lab
