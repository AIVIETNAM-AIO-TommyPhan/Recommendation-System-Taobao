FROM apache/airflow:2.10.5-python3.11

COPY requirements-dag.txt /requirements-dag.txt
# No --constraint here: the DAG needs pandas>=2.2 (for groupby.apply's
# include_groups kwarg) which is newer than the base image's bundled
# pandas 2.1.4, and Airflow's own constraints file pins that older
# version - applying it would make this pin unsatisfiable.
RUN pip install --no-cache-dir -r /requirements-dag.txt
