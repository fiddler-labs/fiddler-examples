from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.papermill.operators.papermill import PapermillOperator


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=1)
}

# User defined functions (placeholder)
def _model_output():
    return None

def _data_prep_etl():
    return None

# Define Dag
with DAG(
    dag_id='airflow_fiddler_event_publishing',
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2021, 1, 1),
    template_searchpath='/usr/local/airflow/include',
    catchup=False
) as dag_1:
    
    publish_events = PapermillOperator(
        task_id="fiddler_publish_events",
        input_nb="include/Fiddler_Churn_Event_Publishing.ipynb",
        output_nb="include/ep-out-{{ execution_date }}.ipynb",
        parameters={"execution_date": "{{ execution_date }}"},
    )
    
    model_output = PythonOperator(
            task_id="model_output_placeholder",
            python_callable=_model_output
    )
    
    data_preperation = PythonOperator(
        task_id="data_preparation_etl_placeholder",
        python_callable=_data_prep_etl
    )
    
    data_preperation >> model_output >> publish_events # Graph Flow
    