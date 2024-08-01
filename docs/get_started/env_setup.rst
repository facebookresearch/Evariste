Setup
=====

Conda env setup
---------------

.. code-block:: console

    # create conda env
        
        conda deactivate
        conda create -n evariste_20220415 python=3.8.8
        conda activate evariste_20220415


    # Update .bashrc (or equivalent)

        module purge
        module load cuda/11.3
        module load cudnn/v8.1.1.33-cuda.11.0
        module load NCCL/2.10.3-cuda.11.0


    # Install PyTorch 
    
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch


    # Clone Evariste and install requirements
    
    cd ..
    git clone git@github.com:facebookresearch/Evariste.git
    cd Evariste/formal

    pip install -r requirements.txt




Developing
----------
We currently use black with version 19.10b0
``pip install black==19.10b0``

Run unit tests
--------------

.. code-block::  console

    cd formal
    python -m pytest -m "not slow" tests/fast
    python -m pytest -m "not slow" tests/mcts
    python -m pytest -m "not slow" evariste/backward
    python -m pytest -m "not slow" evariste/forward/
    python -m pytest -m "not slow" params/
    python -m pytest -m "not slow" evariste/comms
    python -m pytest evariste/model/test_reload_model.py
    python -m pytest evariste/model/data/envs/batch_iterator_tests.py
    python -m pytest -m "not slow" evariste/async_workers
    pytest evariste/envs/eq/test

Type checking with Mypy
-----------------------

.. code-block:: console

    cd formal
    mypy --install-types --non-interactive ./

