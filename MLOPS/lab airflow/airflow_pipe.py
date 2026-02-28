import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer # т.н. преобразователь колонок
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import requests
from pathlib import Path
import os
from datetime import timedelta
from train_model import train

def download_data():
    df = pd.read_csv('/home/artem/Downloads/archive/US_Top_50_Universities_2026.csv', delimiter = ',')
    df.to_csv("universities.csv", index = False)
    print("df: ", df.shape)
    return df

def clear_data():
    df = pd.read_csv("universities.csv")

    # Категориальные и числовые признаки
    cat_columns = ['University_Name', 'Institution_Type', 'State']
    num_columns = ['Founded_Year',
                   'Research_Impact_Score',
                   'Intl_Student_Ratio',
                   'Employment_Rate']

    # Удаляем пропуски
    df = df.dropna()

    # Проверка года основания (разумные границы)
    df = df[(df['Founded_Year'] > 1600) & (df['Founded_Year'] <= 2026)]

    # Ограничим значения метрик от 0 до 100 (если это проценты / индексы)
    df = df[(df['Research_Impact_Score'] >= 0) & (df['Research_Impact_Score'] <= 100)]
    df = df[(df['Intl_Student_Ratio'] >= 0) & (df['Intl_Student_Ratio'] <= 100)]
    df = df[(df['Employment_Rate'] >= 0) & (df['Employment_Rate'] <= 100)]

    df = df.reset_index(drop=True)

    # Кодирование категориальных признаков
    ordinal = OrdinalEncoder()
    df[cat_columns] = ordinal.fit_transform(df[cat_columns])

    df.to_csv("df_clear.csv", index=False)

    print("Cleaned shape:", df.shape)
    return True

dag_universities = DAG(
    dag_id="train_pipe",
    start_date=datetime(2025, 2, 3),
    max_active_tasks=4,
    schedule=timedelta(minutes=5),
#   schedule="@hourly",
    max_active_runs=1,
    catchup=False,
)
download_task = PythonOperator(python_callable=download_data, task_id = "download_universities", dag = dag_universities)
clear_task = PythonOperator(python_callable=clear_data, task_id = "clear_universities", dag = dag_universities)
train_task = PythonOperator(python_callable=train, task_id = "train_universities", dag = dag_universities)
download_task >> clear_task >> train_task
