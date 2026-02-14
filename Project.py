import warnings
warnings.filterwarnings('ignore')

import io
import tkinter as tk
from tkinter import messagebox, scrolledtext

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

sns.set(style="darkgrid")
plt.rcParams['figure.dpi'] = 120


class HeartDiseaseGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Heart Disease Prediction - ML GUI")
        self.root.geometry("720x520")

        self.df = None
        self.model = None
        self.cm = None
        self.X_columns = None

        tk.Button(root, text="Load Dataset (with Cleaning)", width=30, command=self.load_data).pack(pady=4)
        tk.Button(root, text="Show Target Distribution", width=30, command=self.plot_target).pack(pady=4)
        tk.Button(root, text="Show Correlation Heatmap", width=30, command=self.plot_heatmap).pack(pady=4)
        tk.Button(root, text="Train Random Forest Model", width=30, command=self.train_model).pack(pady=4)
        tk.Button(root, text="Show Confusion Matrix", width=30, command=self.plot_confusion).pack(pady=4)
        tk.Button(root, text="Show Feature Importance", width=30, command=self.plot_importance).pack(pady=4)

        self.output = scrolledtext.ScrolledText(root, width=85, height=12)
        self.output.pack(pady=10)

    # LOAD + CLEAN DATA

    def load_data(self):
        try:
            data = pd.read_csv(r"A:\Apps\University\Level 2\SEM 1\AI\Project\heart.csv")

            # RAW DATA OUTPUT
            self.output.delete(1.0, tk.END)
            self.output.insert(tk.END, "Dataset Loaded Successfully ✔\n\n")
            self.output.insert(tk.END, f"Original Shape: {data.shape}\n\n")

            self.output.insert(tk.END, "HEAD\n")
            self.output.insert(tk.END, str(data.head()) + "\n\n")

            self.output.insert(tk.END, "TAIL\n")
            self.output.insert(tk.END, str(data.tail()) + "\n\n")

            self.output.insert(tk.END, "DESCRIBE\n")
            self.output.insert(tk.END, str(data.describe()) + "\n\n")

            buffer = io.StringIO()
            data.info(buf=buffer)
            self.output.insert(tk.END, "INFO\n")
            self.output.insert(tk.END, buffer.getvalue() + "\n")

            # CLEANING

            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

            for col in numeric_cols:
                data[col].fillna(data[col].median(), inplace=True)

            data.drop_duplicates(inplace=True)

            for col in numeric_cols:
                data[col] = data[col].abs()

            data = pd.get_dummies(
                data,
                columns=['cp', 'restecg', 'thal'],
                drop_first=True
            )

            for col in ['sex', 'fbs', 'exang', 'slope', 'ca', 'target']:
                data[col] = data[col].astype(int)

            self.df = data

            # CLEANED DATA OUTPUT
            self.output.insert(tk.END, "\n===== DATA AFTER CLEANING =====\n")
            self.output.insert(tk.END, f"Final Shape: {self.df.shape}\n\n")
            self.output.insert(tk.END, str(self.df.head()) + "\n")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # TARGET DISTRIBUTION

    def plot_target(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Load dataset first")
            return

        plt.figure(figsize=(4, 3))
        sns.countplot(x='target', data=self.df)
        plt.title("Target Distribution")
        plt.tight_layout()
        plt.show()

    # HEATMAP

    def plot_heatmap(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Load dataset first")
            return

        plt.figure(figsize=(7, 5))
        sns.heatmap(self.df.corr(), annot=False, cmap="Reds")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()

    # TRAIN MODEL

    def train_model(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Load dataset first")
            return

        X = self.df.drop('target', axis=1)
        y = self.df['target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        rf = RandomForestClassifier(random_state=42)

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6, None],
            'min_samples_split': [2, 4]
        }

        grid = GridSearchCV(
            rf,
            param_grid,
            cv=5,
            scoring='recall',
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        self.model = grid.best_estimator_
        self.X_columns = X_train.columns

        y_pred = self.model.predict(X_test)
        self.cm = confusion_matrix(y_test, y_pred)

        self.output.insert(tk.END, "\nModel Trained Successfully ✔\n")
        self.output.insert(tk.END, f"Best Parameters: {grid.best_params_}\n\n")
        self.output.insert(tk.END, classification_report(y_test, y_pred))

    # CONFUSION MATRIX

    def plot_confusion(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Train model first")
            return

        plt.figure(figsize=(4, 3))
        sns.heatmap(self.cm, annot=True, fmt='d', cmap='Reds')
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    # FEATURE IMPORTANCE

    def plot_importance(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Train model first")
            return

        importances = self.model.feature_importances_

        if len(importances) != len(self.X_columns):
            messagebox.showerror("Error", "Feature mismatch detected")
            return

        imp = pd.Series(importances, index=self.X_columns)
        imp = imp.sort_values().tail(8)

        plt.figure(figsize=(5, 4))
        imp.plot(kind='barh')
        plt.title("Top Feature Importances")
        plt.tight_layout()
        plt.show()


# RUN GUI

if __name__ == "__main__":
    root = tk.Tk()
    app = HeartDiseaseGUI(root)
    root.mainloop()
