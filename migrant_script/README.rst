=================================================
Migrant script: Emigration form Mlflow to Trains
=================================================
Migrant all your tracking experiments, and models.

Installing
----------
install the requirements by run::

    conda install --file requirements.txt

Using Migrant
_____________
To run the migrant script please provide <path> and <branch> parameters:

- <branch> stand for running environment (e.g Local or Remote).
- <path> stand for path to mlruns directory or database URL (e.g ``postgresql://<IP>:<PORT>/<DB-NAME>`` ) .

Run::

    python trains_immigration.py <branch> <path>

