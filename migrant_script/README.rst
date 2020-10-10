=================================================
Migrant script: Emigration form Mlflow to Trains
=================================================
Migrant all your tracking experiments, and models.

Installing
----------
install the requirements by run::

    pip install -r /path/to/requirements.txt

Using Migrant
_____________
To run the migrant script please provide <path> and <branch> parameters:

- <branch> stand for running environment (e.g Local or Remote).
- <path> stand for path/address to mlruns directory.

Run::

    python trains_immigration.py local /Users/.../mlruns

