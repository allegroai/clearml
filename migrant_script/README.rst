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

Important Note
--------------
According to MLflow tracking UI migration -  artifacts migration from remote computer unsupported, only from storage server (e.g. S3)