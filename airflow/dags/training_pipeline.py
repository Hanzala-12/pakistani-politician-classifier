"""
Airflow DAG for Pakistani Politician Classifier Training Pipeline
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


# Default arguments
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'politician_classifier_pipeline',
    default_args=default_args,
    description='End-to-end training pipeline for Pakistani Politician Classifier',
    schedule_interval='@weekly',  # Run weekly
    catchup=False,
    tags=['ml', 'computer-vision', 'politician-classifier'],
)


def collect_data_task():
    """Task to collect data"""
    from src.collect_data import main
    main()


def split_dataset_task():
    """Task to split dataset"""
    from src.split_dataset import main
    main()


def augment_data_task():
    """Task to augment training data"""
    from src.augment import main
    main()


def train_models_task():
    """Task to train models"""
    from src.train import main
    main()


def evaluate_models_task():
    """Task to evaluate models"""
    from src.evaluate import main
    main()


# Define tasks
t1_collect = PythonOperator(
    task_id='collect_data',
    python_callable=collect_data_task,
    dag=dag,
)

t2_split = PythonOperator(
    task_id='split_dataset',
    python_callable=split_dataset_task,
    dag=dag,
)

t3_augment = PythonOperator(
    task_id='augment_data',
    python_callable=augment_data_task,
    dag=dag,
)

t4_train = PythonOperator(
    task_id='train_models',
    python_callable=train_models_task,
    dag=dag,
)

t5_evaluate = PythonOperator(
    task_id='evaluate_models',
    python_callable=evaluate_models_task,
    dag=dag,
)

# Optional: DVC tracking
t6_dvc_push = BashOperator(
    task_id='dvc_push',
    bash_command='cd {{ params.project_root }} && dvc push',
    params={'project_root': str(project_root)},
    dag=dag,
)

# Set task dependencies
t1_collect >> t2_split >> t3_augment >> t4_train >> t5_evaluate >> t6_dvc_push
