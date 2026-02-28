import mlflow
import os
from mlflow.models import infer_signature
import mlflow.sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,confusion_matrix,ConfusionMatrixDisplay

def setup_mlflow(experiment_name="Hand-Gesture-Classification",tracking_uri=None):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)    



def log_dataset(df):
    mlflow.log_param("dataset_rows",df.shape[0])
    mlflow.log_param("dataset_features",df.shape[1])
    mlflow.log_param("num_classes",df['label'].nunique())

    summary_path = "dataset_summary.csv"
    df.describe().to_csv(summary_path)
    mlflow.log_artifact(summary_path)
    os.remove(summary_path)

def log_model_params(model_name, params_dict):

    mlflow.log_param("model_name",model_name)
    for param_name , param_value in params_dict.items():
        try:
            mlflow.log_param(param_name,param_value)
        except:
            pass

def train_and_log_model(model, X_train, y_train):

    model.fit(X_train,y_train)

    signature = infer_signature(X_train , model.predict(X_train))            
    mlflow.sklearn.log_model(model, "model",signature=signature)
    mlflow.log_input(mlflow.data.from_pandas(X_train , name="train_data"),context= "training")

    return model


def log_metrics(y_test, y_pred):

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}


def log_classification_report(y_test, y_pred, label_encoder):
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    report_path = "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)
    os.remove(report_path)    

def log_confusion_matrix(y_test, y_pred, label_encoder):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    os.remove("confusion_matrix.png")
    plt.show()


def log_comparison_chart(all_results):
    models_names = list(all_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model_name in enumerate(models_names):
        values = [all_results[model_name][m] * 100 for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=model_name, edgecolor='black')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score (%)')
    ax.set_title('Model Comparison — All Metrics')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig("model_comparison_chart.png")
    mlflow.log_artifact("model_comparison_chart.png")
    os.remove("model_comparison_chart.png")
    plt.show()    