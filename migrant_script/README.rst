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
To run the migrant script please provide <url> parameter.

Supported url formats:

- Local Computer : ``file://path_to_store``
- SQLAlchemy database URI (e.g. ``<dialect>+<driver>://<username>:<password>@<host>:<port>/<database>`` )
- mlflow tracking UI host address (e.g. ``http://<IP>:<PORT>`` )

Run::

    python trains_immigration.py <url>

Options
_______
``-a`` or ``--analysis`` - Printing migration CPU usage for analysis

Important Note
--------------
- According to MLflow tracking UI migration -  artifacts migration from remote computer unsupported, only from storage server (e.g. S3).
- If the <url> parameter in illegal format the script will continue running with local computer configuration.
- MLflow tags (e.g., "estimator_class" etc.) will be recorded in Trains configuration tab under "MLflow Tags" attribute.
- The migrated experiments will be named by mlflow run-uuid number. If the experiments got new names before migration the names will be migrated only in local migration or database migration (exclude HTTP migration).