"""Airflow DAG wrapping the Bronze -> Silver -> Gold medallion pipeline.

Mirrors medallion/run_pipeline.py, but as three separate tasks instead of
one script, so each layer gets its own logs/retries/status in the Airflow
UI. Triggered manually (schedule=None): ingest() re-reads the full
input_data/ source files every run rather than a new day's slice, so
there's no natural cadence to schedule against.
"""
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from medallion.bronze.ingest import ingest
from medallion.gold.features import run as build_gold_features
from medallion.silver.clean import clean

with DAG(
    dag_id="medallion_pipeline",
    description="Bronze -> Silver -> Gold medallion pipeline for the Taobao ad click dataset",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["medallion"],
) as dag:
    # do_xcom_push=False: each layer function returns a pathlib.Path (its
    # output dir), only used for the standalone-script log line - Airflow's
    # XCom backend is JSON-only and can't serialize a Path, and nothing
    # here needs the value passed to the next task anyway.
    bronze_ingest = PythonOperator(task_id="bronze_ingest", python_callable=ingest, do_xcom_push=False)
    silver_clean = PythonOperator(task_id="silver_clean", python_callable=clean, do_xcom_push=False)
    gold_features = PythonOperator(task_id="gold_features", python_callable=build_gold_features, do_xcom_push=False)

    bronze_ingest >> silver_clean >> gold_features
