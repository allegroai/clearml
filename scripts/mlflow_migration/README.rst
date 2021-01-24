==============================================================
Migration script: Migrate your projects form Mlflow to Trains!
==============================================================
Mirgrate all of the tracked experiments, artfiacts, configurations and metrics.

Installing
----------
install the requirements by running:

    pip install -r requirements.txt


Using Migrate Script
____________________
To run the migration script please provide a valid <path> for mlflow projects.

Supported paths:

- Local Computer : ``file://path_to_mlruns_folder``
- SQLAlchemy database URI (e.g. ``<dialect>+<driver>://<username>:<password>@<host>:<port>/<database>`` )
- mlflow tracking UI server address (e.g. ``http://<IP>:<PORT>`` )

Run::

    python trains_immigration.py <path


Important Notes
---------------
- Migrating artifacts from remote HTTP server is only supported if artifacts are stored in an external storage (e.g. S3) and not locally.
- MLflow tags (e.g., "estimator_class" etc.) will be recorded in Trains configuration tab under "MLflow Tags" section.
- Custom MLFlow project names (not run-uuid) won't be reocrded when migrating from a remote HTTP server (MLFlow API limitation)