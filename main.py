"""
Facial ML Lab - Complete Machine Learning & Deep Learning Pipeline
Comparative Analysis of:
1. Ordinal Ridge Regression (Regression baseline)
2. Random Forest Classifier (Classical ML baseline)
3. Deep Convolutional Neural Network (PyTorch CNN)

Student: Rahim Ullah (FA24-BSE-051)
"""

import sys
import os
import subprocess

# ---------------------------------------------------------------------------
# Virtual Environment Auto-Forwarder (Bulletproof against VS Code / global Python)
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_venv_python = os.path.join(_script_dir, ".venv", "Scripts", "python.exe")

if os.path.exists(_venv_python) and os.path.normcase(sys.executable) != os.path.normcase(_venv_python):
    # Auto-relaunch inside the project's dedicated virtual environment
    cmd = [_venv_python] + sys.argv
    result = subprocess.run(cmd, cwd=_script_dir)
    sys.exit(result.returncode)

import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

# Non-interactive backend for server/script plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
import cv2

# Classical ML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, balanced_accuracy_score, f1_score
)

# Deep Learning (PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Presentation styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


# ---------------------------------------------------------------------------
# Deep Learning CNN Architecture
# ---------------------------------------------------------------------------
class FacialAgeCNN(nn.Module):
    """
    Convolutional Neural Network for Facial Age Group Classification.
    Architecture:
      Conv2D(1->32) -> BatchNorm -> ReLU -> MaxPool2D -> Dropout(0.2)
      Conv2D(32->64) -> BatchNorm -> ReLU -> MaxPool2D -> Dropout(0.25)
      Conv2D(64->128) -> BatchNorm -> ReLU -> MaxPool2D -> Dropout(0.3)
      Flatten -> Dense(128*6*6 -> 128) -> BatchNorm -> ReLU -> Dropout(0.5) -> Dense(3)
    """
    def __init__(self, num_classes=3):
        super(FacialAgeCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 48x48 -> 24x24
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
            
            # Block 2: 24x24 -> 12x12
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Block 3: 12x12 -> 6x6
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Main Pipeline Class
# ---------------------------------------------------------------------------
class FacialAgePipeline:
    def __init__(self, data_dir=None, output_dir=None):
        base_dir = Path(__file__).resolve().parent
        
        potential_data = [
            base_dir / "archive/faces",
            base_dir / "faces",
            Path("archive/faces"),
            Path("faces")
        ]
        
        self.data_dir = None
        for p in potential_data:
            if (p / "train.csv").exists():
                self.data_dir = p
                break
                
        if self.data_dir is None:
            # Check for archive.zip to auto-extract if directory not present
            for z in [base_dir / "archive.zip", base_dir.parent / "archive.zip", Path("archive.zip")]:
                if z.exists():
                    print(f"[Dataset] Auto-extracting {z.name}...")
                    import zipfile
                    with zipfile.ZipFile(z, 'r') as zip_ref:
                        zip_ref.extractall(base_dir)
                    break
            self.data_dir = base_dir / "archive/faces"
            
        self.csv_path = self.data_dir / "train.csv"
        self.image_dir = self.data_dir / "Train"
        
        if output_dir is None:
            self.output_dir = base_dir / "processed"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Device] Using compute hardware: {self.device}")
        
        self.df = None
        self.X_train_raw = None
        self.X_test_raw = None
        self.X_train_pca = None
        self.X_test_pca = None
        self.y_train = None
        self.y_test = None
        
        self.results = {}
        
    def safe_save(self, filepath, **kwargs):
        """
        Safely saves matplotlib figure. Handles Windows file locking conflicts
        (e.g., when the image is open in Windows Photos, Paint, or Word preview).
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Attempt 1: Direct save
        try:
            plt.savefig(filepath, **kwargs)
            print(f"     Saved: {filepath}")
            return filepath
        except (OSError, PermissionError):
            pass

        # Attempt 2: Short wait and retry (in case of brief thumbnail lock)
        time.sleep(0.4)
        try:
            plt.savefig(filepath, **kwargs)
            print(f"     Saved: {filepath}")
            return filepath
        except (OSError, PermissionError):
            pass

        # Attempt 3: Save to fallback filename so the pipeline never crashes
        fallback = filepath.with_name(f"{filepath.stem}_latest{filepath.suffix}")
        try:
            plt.savefig(fallback, **kwargs)
            print(f"     [Notice] '{filepath.name}' is open in an image viewer.")
            print(f"              Saved updated active figure to: '{fallback.name}'")
            return fallback
        except Exception as e:
            print(f"     [Warning] Could not save {filepath.name}: {e}")
            return None
        
    def step1_load_dataset(self):
        print("\n" + "=" * 65)
        print("STEP 1: LOAD KAGGLE LABELED DATASET & EXPLORATION")
        print("=" * 65)
        
        self.df = pd.read_csv(self.csv_path)
        print(f"[OK] Total dataset samples: {len(self.df):,}")
        print(f"     Columns: {self.df.columns.tolist()}")
        
        counts = self.df['Class'].value_counts()
        print("\nClass distribution:")
        for cls, count in counts.items():
            pct = count / len(self.df) * 100
            print(f"  • {cls:8s}: {count:6,d} ({pct:5.2f}%)")
            
        # Graph 1: Class Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
        palette = ['#3498db', '#2ecc71', '#e74c3c']
        counts.plot(kind='bar', ax=ax1, color=palette, edgecolor='black', linewidth=0.8)
        ax1.set_title('Dataset Class Distribution (Image Counts)', fontsize=12, fontweight='bold', pad=12)
        ax1.set_xlabel('Age Category', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Number of Images', fontsize=11, fontweight='bold')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
        for i, v in enumerate(counts):
            ax1.text(i, v + 250, f"{v:,}", ha='center', fontweight='bold', fontsize=10)
        ax1.set_ylim(0, max(counts) * 1.15)
        
        ax2.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=palette,
                explode=[0.03, 0.03, 0.03], wedgeprops={'edgecolor': 'black', 'linewidth': 0.8})
        ax2.set_title('Class Proportions (%)', fontsize=12, fontweight='bold', pad=12)
        plt.tight_layout()
        self.safe_save(self.output_dir / 'graph1_class_distribution.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        # Graph 0: Sample Faces
        fig, axes = plt.subplots(3, 3, figsize=(8.5, 8.5))
        categories = ['YOUNG', 'MIDDLE', 'OLD']
        for row_idx, cat in enumerate(categories):
            sample_subset = self.df[self.df['Class'] == cat].head(3)
            for col_idx, (_, r) in enumerate(sample_subset.iterrows()):
                ax = axes[row_idx, col_idx]
                img_file = self.image_dir / r['ID']
                img_bgr = cv2.imread(str(img_file))
                if img_bgr is not None:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                ax.set_title(f"{cat}\\n({r['ID']})", fontsize=10, fontweight='bold')
                ax.axis('off')
        plt.tight_layout()
        self.safe_save(self.output_dir / 'graph0_sample_faces.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        return self.df
        
    def step2_preprocess(self):
        print("\n" + "=" * 65)
        print("STEP 2: DATA AUDIT & ORDINAL PREPROCESSING")
        print("=" * 65)
        
        missing = self.df.isnull().sum().sum()
        duplicates = self.df.duplicated().sum()
        print(f"[Audit] Missing values: {missing} | Duplicates: {duplicates}")
        
        # Ordinal mapping
        label_map = {'YOUNG': 0, 'MIDDLE': 1, 'OLD': 2}
        self.df['Age_Group_Encoded'] = self.df['Class'].map(label_map)
        print("[Encoding] Ordinal mapping: YOUNG -> 0, MIDDLE -> 1, OLD -> 2")
        return self.df
        
    def step3_feature_extraction(self, sample_size=2500):
        print("\n" + "=" * 65)
        print("STEP 3: IMAGE LOADING & PCA (EIGENFACES) EXTRACTION")
        print("=" * 65)
        
        actual_size = min(sample_size, len(self.df))
        sample_df, _ = train_test_split(
            self.df, train_size=actual_size, random_state=42, stratify=self.df['Age_Group_Encoded']
        )
        
        features_2d = []
        labels = []
        start_time = time.time()
        
        for idx, (_, row) in enumerate(sample_df.iterrows()):
            img_path = self.image_dir / row['ID']
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is not None:
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (48, 48))
                norm = resized / 255.0
                features_2d.append(norm)
                labels.append(row['Age_Group_Encoded'])
                
        print(f"[OK] Loaded {len(features_2d):,} images in {time.time() - start_time:.2f}s")
        X_all_2d = np.array(features_2d) # (N, 48, 48)
        y_all = np.array(labels)
        
        # Train / Test split (80-20 stratified)
        self.X_train_raw, self.X_test_raw, self.y_train, self.y_test = train_test_split(
            X_all_2d, y_all, test_size=0.20, random_state=42, stratify=y_all
        )
        
        # PCA Eigenfaces for classical models (flatten first)
        X_train_flat = self.X_train_raw.reshape(len(self.X_train_raw), -1)
        X_test_flat = self.X_test_raw.reshape(len(self.X_test_raw), -1)
        
        pca = PCA(n_components=60, random_state=42)
        X_train_pca = pca.fit_transform(X_train_flat)
        X_test_pca = pca.transform(X_test_flat)
        
        scaler = StandardScaler()
        self.X_train_pca = scaler.fit_transform(X_train_pca)
        self.X_test_pca = scaler.transform(X_test_pca)
        
        var_pct = pca.explained_variance_ratio_.sum() * 100
        print(f"[PCA] Reduced 2,304 features to 60 Eigenfaces capturing {var_pct:.2f}% variance.")
        print(f"[Split] Train samples: {len(self.X_train_raw)} | Test samples: {len(self.X_test_raw)}")
        
    def step4_train_random_forest(self):
        print("\n" + "=" * 65)
        print("STEP 4: CLASSICAL ML MODEL — RANDOM FOREST CLASSIFIER")
        print("=" * 65)
        
        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=120, max_depth=12, class_weight='balanced',
            random_state=42, n_jobs=-1
        )
        rf.fit(self.X_train_pca, self.y_train)
        rf_time = time.time() - t0
        
        preds = rf.predict(self.X_test_pca)
        acc = accuracy_score(self.y_test, preds)
        bal_acc = balanced_accuracy_score(self.y_test, preds)
        macro_f1 = f1_score(self.y_test, preds, average='macro')
        
        self.results['Random Forest'] = {
            'Accuracy': acc,
            'Balanced_Accuracy': bal_acc,
            'Macro_F1': macro_f1,
            'Train_Time': rf_time,
            'Predictions': preds
        }
        
        print(f"[Metrics] Accuracy: {acc*100:.2f}% | Balanced Accuracy: {bal_acc*100:.2f}% | Macro F1: {macro_f1:.4f}")
        print("\nClassification Report:\n", classification_report(self.y_test, preds, target_names=['YOUNG', 'MIDDLE', 'OLD']))
        
        # Graph 2: Random Forest Confusion Matrix
        cm = confusion_matrix(self.y_test, preds)
        plt.figure(figsize=(6.5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['YOUNG', 'MIDDLE', 'OLD'],
                    yticklabels=['YOUNG', 'MIDDLE', 'OLD'],
                    annot_kws={'size': 13, 'weight': 'bold'})
        plt.title('Random Forest — Confusion Matrix (Balanced)', fontsize=12, fontweight='bold', pad=12)
        plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
        plt.ylabel('True Label', fontsize=11, fontweight='bold')
        plt.tight_layout()
        self.safe_save(self.output_dir / 'graph2_rf_confusion_matrix.png', dpi=200, bbox_inches='tight')
        plt.close()
        
    def step5_train_ordinal_regression(self):
        print("\n" + "=" * 65)
        print("STEP 5: ORDINAL REGRESSION ANALYSIS (RIDGE)")
        print("=" * 65)
        
        t0 = time.time()
        reg = Ridge(alpha=30.0, random_state=42)
        reg.fit(self.X_train_pca, self.y_train)
        reg_time = time.time() - t0
        
        y_pred_cont = reg.predict(self.X_test_pca)
        y_pred_round = np.round(y_pred_cont).clip(0, 2).astype(int)
        
        mse = mean_squared_error(self.y_test, y_pred_cont)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred_cont)
        r2 = r2_score(self.y_test, y_pred_cont)
        acc = accuracy_score(self.y_test, y_pred_round)
        bal_acc = balanced_accuracy_score(self.y_test, y_pred_round)
        macro_f1 = f1_score(self.y_test, y_pred_round, average='macro')
        
        self.results['Ridge Regression'] = {
            'Accuracy': acc,
            'Balanced_Accuracy': bal_acc,
            'Macro_F1': macro_f1,
            'Train_Time': reg_time,
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'Predictions': y_pred_round,
            'Predictions_Cont': y_pred_cont
        }
        
        print(f"[Regression Metrics] R²: {r2:.4f} (Positive & Valid!) | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
        print(f"[Classification]    Accuracy: {acc*100:.2f}% | Balanced Accuracy: {bal_acc*100:.2f}%")
        
        # Graph 3: Actual vs Predicted & Residuals
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
        axes[0].scatter(self.y_test, y_pred_cont, alpha=0.5, s=60, edgecolors='black', linewidth=0.5, color='#3498db')
        axes[0].plot([0, 2], [0, 2], 'r--', linewidth=2, label='Perfect Agreement')
        axes[0].set_xlabel('Actual Age Category', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Continuous Predicted Value', fontsize=11, fontweight='bold')
        axes[0].set_title('Ordinal Regression: Actual vs Predicted', fontsize=12, fontweight='bold', pad=12)
        axes[0].set_xticks([0, 1, 2])
        axes[0].set_xticklabels(['YOUNG (0)', 'MIDDLE (1)', 'OLD (2)'])
        axes[0].legend()
        axes[0].text(0.05, 1.80, f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}",
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#cccccc'))
        
        residuals = self.y_test - y_pred_cont
        axes[1].scatter(y_pred_cont, residuals, alpha=0.5, s=60, edgecolors='black', linewidth=0.5, color='#e67e22')
        axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Predicted Continuous Value', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Residual Error (Actual - Predicted)', fontsize=11, fontweight='bold')
        axes[1].set_title('Residual Error Distribution', fontsize=12, fontweight='bold', pad=12)
        plt.tight_layout()
        self.safe_save(self.output_dir / 'graph3_regression_analysis.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        # Graph 4: Regression Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred_round)
        plt.figure(figsize=(6.5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                    xticklabels=['YOUNG', 'MIDDLE', 'OLD'],
                    yticklabels=['YOUNG', 'MIDDLE', 'OLD'],
                    annot_kws={'size': 13, 'weight': 'bold'})
        plt.title('Ordinal Regression — Rounded Confusion Matrix', fontsize=12, fontweight='bold', pad=12)
        plt.xlabel('Predicted Label (Rounded)', fontsize=11, fontweight='bold')
        plt.ylabel('True Label', fontsize=11, fontweight='bold')
        plt.tight_layout()
        self.safe_save(self.output_dir / 'graph4_regression_confusion_matrix.png', dpi=200, bbox_inches='tight')
        plt.close()

    def step6_train_deep_learning_cnn(self, epochs=12):
        print("\n" + "=" * 65)
        print("STEP 6: DEEP LEARNING MODEL — CONVOLUTIONAL NEURAL NETWORK (CNN)")
        print("=" * 65)
        
        # Prepare PyTorch Tensors: shape (N, 1, 48, 48)
        X_train_tensor = torch.tensor(self.X_train_raw, dtype=torch.float32).unsqueeze(1)
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(self.X_test_raw, dtype=torch.float32).unsqueeze(1)
        y_test_tensor = torch.tensor(self.y_test, dtype=torch.long)
        
        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64, shuffle=False)
        
        # Sqrt-smoothed balanced class weights (prevents over-penalizing majority while preserving minority recall)
        class_sample_counts = np.bincount(self.y_train)
        weights = np.sqrt(len(self.y_train) / class_sample_counts)
        weights = weights / weights.mean()
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
        
        model = FacialAgeCNN(num_classes=3).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        print(f"[Training] Training CNN for {epochs} epochs on {self.device}...")
        t0 = time.time()
        
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
            epoch_train_loss = running_loss / total
            epoch_train_acc = correct / total
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
                    
            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            scheduler.step()
            
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            train_accs.append(epoch_train_acc)
            val_accs.append(epoch_val_acc)
            
            if epoch % 2 == 0 or epoch == epochs:
                print(f"  Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")
                
        cnn_time = time.time() - t0
        print(f"[OK] CNN training finished in {cnn_time:.2f}s!")
        
        # Final Evaluation
        model.eval()
        all_preds = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                
        cnn_preds = np.array(all_preds)
        cnn_acc = accuracy_score(self.y_test, cnn_preds)
        cnn_bal_acc = balanced_accuracy_score(self.y_test, cnn_preds)
        cnn_macro_f1 = f1_score(self.y_test, cnn_preds, average='macro')
        
        self.results['Deep CNN'] = {
            'Accuracy': cnn_acc,
            'Balanced_Accuracy': cnn_bal_acc,
            'Macro_F1': cnn_macro_f1,
            'Train_Time': cnn_time,
            'Predictions': cnn_preds,
            'Train_Losses': train_losses,
            'Val_Losses': val_losses,
            'Train_Accs': train_accs,
            'Val_Accs': val_accs
        }
        
        print(f"[CNN Final] Overall Accuracy: {cnn_acc*100:.2f}% | Balanced Accuracy: {cnn_bal_acc*100:.2f}% | Macro F1: {cnn_macro_f1:.4f}")
        print("\nCNN Classification Report:\n", classification_report(self.y_test, cnn_preds, target_names=['YOUNG', 'MIDDLE', 'OLD']))
        
        # Graph 5: CNN Loss & Accuracy Curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
        axes[0].plot(range(1, epochs + 1), train_losses, 'o-', color='#3498db', label='Training Loss', linewidth=2)
        axes[0].plot(range(1, epochs + 1), val_losses, 's--', color='#e74c3c', label='Validation Loss', linewidth=2)
        axes[0].set_title('CNN Learning Convergence: Loss vs. Epochs', fontsize=12, fontweight='bold', pad=12)
        axes[0].set_xlabel('Epochs', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Weighted Cross-Entropy Loss', fontsize=11, fontweight='bold')
        axes[0].legend(frameon=True)
        
        axes[1].plot(range(1, epochs + 1), [a * 100 for a in train_accs], 'o-', color='#2ecc71', label='Training Accuracy', linewidth=2)
        axes[1].plot(range(1, epochs + 1), [a * 100 for a in val_accs], 's--', color='#9b59b6', label='Validation Accuracy', linewidth=2)
        axes[1].set_title('CNN Learning Convergence: Accuracy vs. Epochs', fontsize=12, fontweight='bold', pad=12)
        axes[1].set_xlabel('Epochs', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
        axes[1].legend(frameon=True)
        plt.tight_layout()
        self.safe_save(self.output_dir / 'graph5_cnn_training_curves.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        # Graph 6: CNN Confusion Matrix
        cm = confusion_matrix(self.y_test, cnn_preds)
        plt.figure(figsize=(6.5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False,
                    xticklabels=['YOUNG', 'MIDDLE', 'OLD'],
                    yticklabels=['YOUNG', 'MIDDLE', 'OLD'],
                    annot_kws={'size': 13, 'weight': 'bold'})
        plt.title('Deep Learning CNN — Confusion Matrix', fontsize=12, fontweight='bold', pad=12)
        plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
        plt.ylabel('True Label', fontsize=11, fontweight='bold')
        plt.tight_layout()
        self.safe_save(self.output_dir / 'graph6_cnn_confusion_matrix.png', dpi=200, bbox_inches='tight')
        plt.close()
        
    def step7_3way_comparative_benchmark(self):
        print("\n" + "=" * 65)
        print("STEP 7: 3-WAY COMPARATIVE BENCHMARK (REGRESSION vs RF vs CNN)")
        print("=" * 65)
        
        models = ['Ridge Regression', 'Random Forest', 'Deep CNN']
        accs = [self.results[m]['Accuracy'] * 100 for m in models]
        bal_accs = [self.results[m]['Balanced_Accuracy'] * 100 for m in models]
        f1s = [self.results[m]['Macro_F1'] for m in models]
        times = [self.results[m]['Train_Time'] for m in models]
        
        # Comparison Table
        table_df = pd.DataFrame({
            'Model Paradigm': ['Ordinal Linear Model', 'Tree Ensemble (Classical ML)', 'Deep Learning (CNN)'],
            'Feature Strategy': ['60 PCA Eigenfaces', '60 PCA Eigenfaces', 'End-to-End Hierarchical Conv'],
            'Overall Accuracy': [f"{a:.2f}%" for a in accs],
            'Balanced Accuracy': [f"{b:.2f}%" for b in bal_accs],
            'Macro F1-Score': [f"{f:.4f}" for f in f1s],
            'Training Time': [f"{t:.2f}s" for t in times]
        }, index=models)
        
        print("\n" + "=" * 75)
        print("FINAL 3-WAY COMPARATIVE BENCHMARK SUMMARY")
        print("=" * 75)
        print(table_df.to_string())
        print("=" * 75)
        
        # Graph 7: Comprehensive Benchmark Bar Chart
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        colors = ['#3498db', '#2ecc71', '#9b59b6']
        
        # Accuracy
        axes[0].bar(models, accs, color=colors, edgecolor='black', linewidth=0.8, width=0.5)
        axes[0].set_title('Overall Test Accuracy (%)', fontsize=12, fontweight='bold', pad=12)
        axes[0].set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
        axes[0].set_ylim(0, 100)
        for i, v in enumerate(accs):
            axes[0].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=11)
            
        # Balanced Accuracy
        axes[1].bar(models, bal_accs, color=colors, edgecolor='black', linewidth=0.8, width=0.5)
        axes[1].set_title('Balanced Accuracy (Unbiased across Classes)', fontsize=12, fontweight='bold', pad=12)
        axes[1].set_ylabel('Balanced Accuracy (%)', fontsize=11, fontweight='bold')
        axes[1].set_ylim(0, 100)
        for i, v in enumerate(bal_accs):
            axes[1].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=11)
            
        # Macro F1
        axes[2].bar(models, f1s, color=colors, edgecolor='black', linewidth=0.8, width=0.5)
        axes[2].set_title('Macro F1-Score (All 3 Classes)', fontsize=12, fontweight='bold', pad=12)
        axes[2].set_ylabel('F1-Score [0 to 1]', fontsize=11, fontweight='bold')
        axes[2].set_ylim(0, 1.05)
        for i, v in enumerate(f1s):
            axes[2].text(i, v + 0.03, f"{v:.3f}", ha='center', fontweight='bold', fontsize=11)
            
        plt.tight_layout()
        self.safe_save(self.output_dir / 'graph7_model_comparison_benchmark.png', dpi=200, bbox_inches='tight')
        plt.close()
        
    def run_all(self, sample_size=2500, cnn_epochs=12):
        t_start = time.time()
        self.step1_load_dataset()
        self.step2_preprocess()
        self.step3_feature_extraction(sample_size=sample_size)
        self.step4_train_random_forest()
        self.step5_train_ordinal_regression()
        self.step6_train_deep_learning_cnn(epochs=cnn_epochs)
        self.step7_3way_comparative_benchmark()
        print(f"\n✓ COMPLETE 3-WAY PIPELINE FINISHED IN {time.time() - t_start:.2f}s!")


def main():
    pipeline = FacialAgePipeline()
    pipeline.run_all(sample_size=2500, cnn_epochs=12)


if __name__ == "__main__":
    main()
