#!/usr/bin/env python3

"""
generate_report_v2.py
Refactored script for multi-model comparison with stratified k-fold cross-validation.

This script:
1. Loads and combines train/test data (56 samples total)
2. Runs stratified 5-fold CV on four BERT models:
   - google-bert/bert-base-uncased
   - google-bert/bert-base-cased
   - monologg/biobert_v1.1_pubmed (BioBERT)
   - NoYo25/BiodivBERT
3. Records timing for each model
4. Saves fold-level predictions
5. Generates error analysis
6. Creates comparison visualizations and HTML report

Addresses reviewer comments: R1-2a, R1-2b, R1-2c, R1-5b, R2-4, R2-5
"""

import os
import sys
import json
import time
import shutil
import glob
import random
import warnings
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path
import base64

# ===== LOGGING SETUP =====
# Create logs directory
LOGS_DIR = Path(__file__).parent / "logs" if '__file__' in dir() else Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Create timestamped log file
log_filename = LOGS_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_filename}")
# ===== END LOGGING SETUP =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics as sk_metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    TrainerCallback,
    EarlyStoppingCallback,
    set_seed
)

warnings.filterwarnings('ignore')

# ===== COMPREHENSIVE REPRODUCIBILITY SETUP =====
SEED = 42  # Using 42 for k-fold consistency as discussed

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

logger.info(f"All random seeds set to {SEED} for reproducible results")

# ===== DEVICE INFO =====
if torch.cuda.is_available():
    logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    logger.warning("No GPU detected - running on CPU (will be slower)")
# ===== END REPRODUCIBILITY SETUP =====

# ===== MODEL CONFIGURATIONS =====
MODELS = {
    'bert-base-uncased': 'google-bert/bert-base-uncased',
    'bert-base-cased': 'google-bert/bert-base-cased',
    'biobert': 'monologg/biobert_v1.1_pubmed',
    'biodivbert': 'NoYo25/BiodivBERT'
}

# ===== HYPERPARAMETERS (Standardized across all models) =====
HYPERPARAMS = {
    'num_train_epochs': 20,
    'early_stopping_patience': 3,
    'learning_rate': 5e-5,
    'per_device_train_batch_size': 8,
    'weight_decay': 0.05,
    'hidden_dropout_prob': 0.2,
    'attention_probs_dropout_prob': 0.2,
    'metric_for_best_model': 'eval_loss',
    'lr_scheduler_type': 'linear',
}

N_FOLDS = 5

# ===== OUTPUT DIRECTORIES =====
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# ===== DATASET CLASS =====
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ===== CALLBACK FOR TRAINING CURVES =====
class TrainingLogCallback(TrainerCallback):
    def __init__(self):
        self.log_history = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.log_history.append(logs.copy())


# ===== COMPUTE METRICS FUNCTION =====
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    accuracy = sk_metrics.accuracy_score(labels, preds)
    return {'eval_accuracy': accuracy}


# ===== DATA LOADING =====
def load_all_data():
    """Load and combine train.csv and test.csv for k-fold CV"""
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    # Combine all data
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Filter out NaN labels
    all_df = all_df.dropna(subset=['label'])
    all_df['label'] = all_df['label'].astype(int)
    
    print(f"Total samples: {len(all_df)}")
    print(f"Class distribution: {all_df['label'].value_counts().to_dict()}")
    
    return all_df


def compute_token_length_stats(all_df, tokenizer_name='monologg/biobert_v1.1_pubmed'):
    """Compute token length statistics before and after truncation"""
    print("\nComputing token length statistics...")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    texts = all_df['abstract_text'].tolist()
    
    lengths_raw = []  # Before truncation
    lengths_truncated = []  # After truncation to 512
    truncated_count = 0
    
    for text in texts:
        # Get raw token count (no truncation)
        encoding_raw = tokenizer.encode_plus(
            text,
            truncation=False,
            add_special_tokens=True,
            return_tensors=None
        )
        raw_len = len(encoding_raw['input_ids'])
        lengths_raw.append(raw_len)
        
        # Get truncated token count
        encoding_trunc = tokenizer.encode_plus(
            text,
            truncation=True,
            max_length=512,
            add_special_tokens=True,
            return_tensors=None
        )
        trunc_len = len(encoding_trunc['input_ids'])
        lengths_truncated.append(trunc_len)
        
        if raw_len > 512:
            truncated_count += 1
    
    stats = {
        'raw_mean': np.mean(lengths_raw),
        'raw_std': np.std(lengths_raw),
        'raw_min': min(lengths_raw),
        'raw_max': max(lengths_raw),
        'truncated_mean': np.mean(lengths_truncated),
        'truncated_std': np.std(lengths_truncated),
        'truncated_min': min(lengths_truncated),
        'truncated_max': max(lengths_truncated),
        'n_truncated': truncated_count,
        'n_total': len(texts),
        'pct_truncated': (truncated_count / len(texts)) * 100
    }
    
    print(f"  Token lengths (before truncation): mean={stats['raw_mean']:.1f} ± {stats['raw_std']:.1f}, range=[{stats['raw_min']}, {stats['raw_max']}]")
    print(f"  Token lengths (after truncation):  mean={stats['truncated_mean']:.1f} ± {stats['truncated_std']:.1f}, range=[{stats['truncated_min']}, {stats['truncated_max']}]")
    print(f"  Abstracts exceeding 512 tokens: {truncated_count}/{len(texts)} ({stats['pct_truncated']:.1f}%)")
    
    # Save to file
    stats_path = 'results/token_length_stats.json'
    os.makedirs('results', exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Token stats saved to: {stats_path}")
    
    return stats


# ===== SINGLE FOLD TRAINING =====
def train_single_fold(model_name, base_model_path, train_texts, train_labels, 
                      eval_texts, eval_labels, fold_idx, output_dir):
    """Train a model on a single fold and return metrics"""
    
    logger.info(f"Training fold {fold_idx + 1}/{N_FOLDS}...")
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(base_model_path, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Apply dropout regularization
    model.config.hidden_dropout_prob = HYPERPARAMS['hidden_dropout_prob']
    model.config.attention_probs_dropout_prob = HYPERPARAMS['attention_probs_dropout_prob']
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    eval_dataset = TextDataset(eval_texts, eval_labels, tokenizer)
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    
    # Custom Trainer for weighted loss
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
    
    # Training arguments
    fold_output_dir = os.path.join(output_dir, f'fold_{fold_idx}')
    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        logging_dir=os.path.join(fold_output_dir, 'logs'),
        num_train_epochs=HYPERPARAMS['num_train_epochs'],
        per_device_train_batch_size=HYPERPARAMS['per_device_train_batch_size'],
        learning_rate=HYPERPARAMS['learning_rate'],
        weight_decay=HYPERPARAMS['weight_decay'],
        lr_scheduler_type=HYPERPARAMS['lr_scheduler_type'],
        save_steps=500,
        logging_steps=1,
        logging_strategy='steps',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model=HYPERPARAMS['metric_for_best_model'],
        greater_is_better=False,
        report_to='none',  # Disable wandb/tensorboard
    )
    
    # Training callback
    log_callback = TrainingLogCallback()
    
    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=HYPERPARAMS['early_stopping_patience']), log_callback],
    )
    
    # Train
    trainer.train()
    
    # Get predictions on eval set (test fold)
    predictions = trainer.predict(eval_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    probs = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
    
    # Calculate metrics
    accuracy = sk_metrics.accuracy_score(eval_labels, preds)
    precision = sk_metrics.precision_score(eval_labels, preds, average='macro', zero_division=0)
    recall = sk_metrics.recall_score(eval_labels, preds, average='macro', zero_division=0)
    f1 = sk_metrics.f1_score(eval_labels, preds, average='macro', zero_division=0)
    
    fold_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    # Prepare predictions dataframe
    pred_df = pd.DataFrame({
        'abstract_text': eval_texts,
        'true_label': eval_labels,
        'predicted_label': preds.tolist(),
        'probability_class_0': probs[:, 0].tolist(),
        'probability_class_1': probs[:, 1].tolist(),
        'fold': fold_idx
    })
    
    # Clean up to free memory
    del model, trainer, tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return fold_metrics, pred_df, log_callback.log_history


# ===== RUN K-FOLD CV FOR SINGLE MODEL =====
def run_kfold_for_model(model_name, base_model_path, all_df):
    """Run stratified k-fold CV for a single model"""
    
    logger.info(f"{'='*60}")
    logger.info(f"Running {N_FOLDS}-fold CV for: {model_name}")
    logger.info(f"Base model: {base_model_path}")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    
    texts = all_df['abstract_text'].tolist()
    labels = all_df['label'].tolist()
    
    output_dir = f'results_{model_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    all_fold_metrics = []
    all_predictions = []
    all_log_histories = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]
        
        fold_metrics, pred_df, log_history = train_single_fold(
            model_name, base_model_path,
            train_texts, train_labels,
            test_texts, test_labels,
            fold_idx, output_dir
        )
        
        all_fold_metrics.append(fold_metrics)
        all_predictions.append(pred_df)
        all_log_histories.append(log_history)
        
        logger.info(f"  Fold {fold_idx + 1} metrics: Acc={fold_metrics['accuracy']:.4f}, F1={fold_metrics['f1']:.4f}")
    
    elapsed_time = time.time() - start_time
    
    # Aggregate metrics
    metrics_df = pd.DataFrame(all_fold_metrics)
    aggregated_metrics = {
        'accuracy_mean': metrics_df['accuracy'].mean(),
        'accuracy_std': metrics_df['accuracy'].std(),
        'precision_mean': metrics_df['precision'].mean(),
        'precision_std': metrics_df['precision'].std(),
        'recall_mean': metrics_df['recall'].mean(),
        'recall_std': metrics_df['recall'].std(),
        'f1_mean': metrics_df['f1'].mean(),
        'f1_std': metrics_df['f1'].std(),
        'elapsed_time_seconds': elapsed_time,
        'elapsed_time_formatted': f"{elapsed_time/60:.1f} min"
    }
    
    # Save fold-level predictions
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df['model'] = model_name
    predictions_path = f'results/fold_predictions_{model_name}.csv'
    os.makedirs('results', exist_ok=True)
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to: {predictions_path}")
    
    # Save aggregated metrics
    metrics_path = f'results/metrics_{model_name}.json'
    with open(metrics_path, 'w') as f:
        json.dump(aggregated_metrics, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_path}")
    
    logger.info(f"{model_name} Summary:")
    logger.info(f"  Accuracy: {aggregated_metrics['accuracy_mean']:.4f} ± {aggregated_metrics['accuracy_std']:.4f}")
    logger.info(f"  F1 Score: {aggregated_metrics['f1_mean']:.4f} ± {aggregated_metrics['f1_std']:.4f}")
    logger.info(f"  Time: {aggregated_metrics['elapsed_time_formatted']}")
    
    return aggregated_metrics, predictions_df, all_fold_metrics


# ===== ERROR ANALYSIS =====
def generate_error_analysis(all_predictions_dict):
    """Generate error analysis across all models"""
    
    logger.info("Generating error analysis...")
    
    # Combine all predictions
    all_preds = []
    for model_name, pred_df in all_predictions_dict.items():
        pred_df = pred_df.copy()
        pred_df['model'] = model_name
        all_preds.append(pred_df)
    
    combined_df = pd.concat(all_preds, ignore_index=True)
    
    # Find misclassified samples
    combined_df['is_misclassified'] = combined_df['true_label'] != combined_df['predicted_label']
    misclassified = combined_df[combined_df['is_misclassified']].copy()
    
    # Count how many models misclassified each abstract
    misclass_counts = misclassified.groupby('abstract_text').size().reset_index(name='n_models_misclassified')
    
    # Merge back
    misclassified = misclassified.merge(misclass_counts, on='abstract_text', how='left')
    
    # Sort by number of models that misclassified (most problematic first)
    misclassified = misclassified.sort_values(['n_models_misclassified', 'abstract_text'], ascending=[False, True])
    
    # Save error analysis
    error_path = 'figures/error_analysis.csv'
    os.makedirs('figures', exist_ok=True)
    misclassified.to_csv(error_path, index=False)
    
    logger.info(f"Error analysis saved to: {error_path}")
    logger.info(f"Total misclassifications across all models/folds: {len(misclassified)}")
    
    # Summary: unique abstracts misclassified by multiple models
    multi_model_errors = misclass_counts[misclass_counts['n_models_misclassified'] > 1]
    logger.info(f"Abstracts misclassified by 2+ models: {len(multi_model_errors)}")
    
    return misclassified


# ===== VISUALIZATION FUNCTIONS =====
def generate_comparison_bar_chart(all_metrics_dict):
    """Generate comparison bar chart with error bars"""
    
    logger.info("Generating comparison bar chart...")
    
    models = list(all_metrics_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.2
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    for i, metric in enumerate(metrics):
        means = [all_metrics_dict[m][f'{metric}_mean'] for m in models]
        stds = [all_metrics_dict[m][f'{metric}_std'] for m in models]
        
        bars = ax.bar(x + i * width, means, width, label=metric.capitalize(), 
                     yerr=stds, capsize=3, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Model Comparison: {N_FOLDS}-Fold Cross-Validation Results', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/model_comparison.png', dpi=150)
    
    # Also return base64 for HTML
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    print("  Saved to: figures/model_comparison.png")
    return f"data:image/png;base64,{img_base64}"


def generate_confusion_matrices(all_predictions_dict):
    """Generate confusion matrices for all models"""
    
    print("\nGenerating confusion matrices...")
    
    n_models = len(all_predictions_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    labels = ['Solo', 'Dual']
    
    for idx, (model_name, pred_df) in enumerate(all_predictions_dict.items()):
        true_labels = pred_df['true_label'].values
        pred_labels = pred_df['predicted_label'].values
        
        cm = sk_metrics.confusion_matrix(true_labels, pred_labels, labels=[0, 1])
        
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, 
                   cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{model_name}')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('figures/confusion_matrices_comparison.png', dpi=150)
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    print("  Saved to: figures/confusion_matrices_comparison.png")
    return f"data:image/png;base64,{img_base64}"


def generate_timing_chart(all_metrics_dict):
    """Generate timing comparison chart"""
    
    print("\nGenerating timing chart...")
    
    models = list(all_metrics_dict.keys())
    times = [all_metrics_dict[m]['elapsed_time_seconds'] / 60 for m in models]  # Convert to minutes
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    bars = ax.barh(models, times, color=colors)
    
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_title(f'Training Time per Model ({N_FOLDS}-Fold CV)', fontsize=14)
    
    # Add time labels
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
               f'{t:.1f} min', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/timing_comparison.png', dpi=150)
    plt.close()
    
    print("  Saved to: figures/timing_comparison.png")


# ===== HTML REPORT GENERATION =====
def create_comparison_html_report(all_metrics_dict, comparison_img, cm_img, error_df):
    """Create comprehensive HTML comparison report"""
    
    print("\nCreating HTML report...")
    
    # Build metrics table rows
    metrics_rows = ""
    for model_name, metrics in all_metrics_dict.items():
        metrics_rows += f"""
        <tr>
            <td><strong>{model_name}</strong></td>
            <td>{metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}</td>
            <td>{metrics['precision_mean']:.4f} ± {metrics['precision_std']:.4f}</td>
            <td>{metrics['recall_mean']:.4f} ± {metrics['recall_std']:.4f}</td>
            <td>{metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}</td>
            <td>{metrics['elapsed_time_formatted']}</td>
        </tr>
        """
    
    # Error analysis summary
    if len(error_df) > 0:
        top_errors = error_df.head(10)
        error_rows = ""
        for _, row in top_errors.iterrows():
            error_rows += f"""
            <tr>
                <td>{row['abstract_text'][:100]}...</td>
                <td>{'Dual' if row['true_label'] == 1 else 'Solo'}</td>
                <td>{'Dual' if row['predicted_label'] == 1 else 'Solo'}</td>
                <td>{row['model']}</td>
                <td>{row.get('n_models_misclassified', 1)}</td>
            </tr>
            """
    else:
        error_rows = "<tr><td colspan='5'>No misclassifications found</td></tr>"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>BioBERT vs BERT Baseline Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; max-width: 1200px; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; }}
            .highlight {{ background-color: #e8f6f3; }}
            .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
            .config {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            code {{ background-color: #ecf0f1; padding: 2px 6px; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <h1>BioBERT vs BERT Baseline Comparison Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Experiment Configuration</h2>
        <div class="config">
            <p><strong>Cross-validation:</strong> Stratified {N_FOLDS}-Fold</p>
            <p><strong>Total samples:</strong> 56</p>
            <p><strong>Random seed:</strong> {SEED}</p>
            <p><strong>Max epochs:</strong> {HYPERPARAMS['num_train_epochs']} (with early stopping, patience={HYPERPARAMS['early_stopping_patience']})</p>
            <p><strong>Learning rate:</strong> {HYPERPARAMS['learning_rate']}</p>
            <p><strong>Batch size:</strong> {HYPERPARAMS['per_device_train_batch_size']}</p>
            <p><strong>Dropout:</strong> {HYPERPARAMS['hidden_dropout_prob']}</p>
        </div>
        
        <h2>Model Comparison Results</h2>
        <p>All metrics reported as mean ± standard deviation across {N_FOLDS} folds.</p>
        <table>
            <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
                <th>Training Time</th>
            </tr>
            {metrics_rows}
        </table>
        
        <h2>Comparison Visualization</h2>
        <img src="{comparison_img}" alt="Model Comparison Chart">
        
        <h2>Confusion Matrices</h2>
        <img src="{cm_img}" alt="Confusion Matrices">
        
        <h2>Error Analysis (Top 10 Misclassifications)</h2>
        <p>Samples sorted by number of models that misclassified them.</p>
        <table>
            <tr>
                <th>Abstract (truncated)</th>
                <th>True Label</th>
                <th>Predicted</th>
                <th>Model</th>
                <th># Models Wrong</th>
            </tr>
            {error_rows}
        </table>
        <p><em>Full error analysis available in: figures/error_analysis.csv</em></p>
        
        <h2>Fold-Level Predictions</h2>
        <p>Detailed predictions for each fold are saved in:</p>
        <ul>
            {''.join(f"<li><code>results/fold_predictions_{m}.csv</code></li>" for m in all_metrics_dict.keys())}
        </ul>
        
        <h2>Reviewer Comments Addressed</h2>
        <ul>
            <li><strong>R1-2a:</strong> Added baseline model comparison (BERT-uncased, BERT-cased, BioBERT, BiodivBERT)</li>
            <li><strong>R1-2c, R2-4:</strong> Implemented stratified {N_FOLDS}-fold cross-validation</li>
            <li><strong>R1-5b:</strong> Added error analysis with multi-model misclassification flagging</li>
            <li><strong>R2-5:</strong> Clarified validation approach through k-fold methodology</li>
        </ul>
        
    </body>
    </html>
    """
    
    with open('project_report_comparison.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("  Report saved to: project_report_comparison.html")


# ===== MAIN FUNCTION =====
def main():
    """Main function to run the complete comparison pipeline"""
    
    total_start_time = time.time()
    
    print("\n" + "="*70)
    print("BioBERT vs BERT Baseline Comparison Pipeline")
    print("Stratified {}-Fold Cross-Validation".format(N_FOLDS))
    print("="*70 + "\n")
    
    # Load data
    all_df = load_all_data()
    
    # Compute token length statistics (before and after truncation)
    token_stats = compute_token_length_stats(all_df)
    
    # Store results
    all_metrics_dict = {}
    all_predictions_dict = {}
    
    # Run k-fold CV for each model (with resume capability)
    for model_name, base_model_path in MODELS.items():
        metrics_file = RESULTS_DIR / f"metrics_{model_name}.json"
        predictions_file = RESULTS_DIR / f"fold_predictions_{model_name}.csv"
        
        # Check if this model already completed (resume capability)
        if metrics_file.exists() and predictions_file.exists():
            logger.info(f"{'='*60}")
            logger.info(f"SKIPPING {model_name} - results already exist")
            logger.info(f"  Metrics: {metrics_file}")
            logger.info(f"  Predictions: {predictions_file}")
            logger.info(f"{'='*60}")
            
            # Load existing results
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            predictions = pd.read_csv(predictions_file)
            all_metrics_dict[model_name] = metrics
            all_predictions_dict[model_name] = predictions
            continue
        
        try:
            metrics, predictions, fold_metrics = run_kfold_for_model(
                model_name, base_model_path, all_df
            )
            all_metrics_dict[model_name] = metrics
            all_predictions_dict[model_name] = predictions
        except Exception as e:
            logger.error(f"FAILED: {model_name} - {type(e).__name__}: {e}")
            logger.exception("Full traceback:")
            # Continue to next model instead of crashing
            continue
    
    # Generate error analysis
    error_df = generate_error_analysis(all_predictions_dict)
    
    # Generate visualizations
    comparison_img = generate_comparison_bar_chart(all_metrics_dict)
    cm_img = generate_confusion_matrices(all_predictions_dict)
    generate_timing_chart(all_metrics_dict)
    
    # Create HTML report
    create_comparison_html_report(all_metrics_dict, comparison_img, cm_img, error_df)
    
    # Save combined metrics
    combined_metrics_path = 'results/all_model_metrics.json'
    with open(combined_metrics_path, 'w') as f:
        json.dump(all_metrics_dict, f, indent=2)
    logger.info(f"Combined metrics saved to: {combined_metrics_path}")
    
    total_elapsed = time.time() - total_start_time
    logger.info("="*70)
    logger.info(f"Pipeline completed successfully!")
    logger.info(f"Total time: {total_elapsed/60:.1f} minutes")
    logger.info("="*70)
    
    # Print final summary
    logger.info("FINAL SUMMARY")
    logger.info("-" * 50)
    for model_name, metrics in all_metrics_dict.items():
        logger.info(f"{model_name:20s}: Acc={metrics['accuracy_mean']:.4f}±{metrics['accuracy_std']:.4f}, "
              f"F1={metrics['f1_mean']:.4f}±{metrics['f1_std']:.4f}, "
              f"Time={metrics['elapsed_time_formatted']}")


# ===== INSTRUCTIONS =====
INSTRUCTIONS = """
================================================================================
BioBERT vs BERT Baseline Comparison Pipeline (v2)
================================================================================

This script runs stratified 5-fold cross-validation on four BERT models:
  1. bert-base-uncased  - Standard BERT baseline (uncased)
  2. bert-base-cased    - Standard BERT baseline (cased)
  3. biobert            - Domain-specific biomedical model
  4. biodivbert         - Domain-specific biodiversity model

Outputs:
  - results/fold_predictions_{model}.csv  - Per-fold predictions
  - results/metrics_{model}.json          - Aggregated metrics per model
  - results/all_model_metrics.json        - Combined comparison metrics
  - figures/model_comparison.png          - Bar chart with error bars
  - figures/confusion_matrices_comparison.png - Side-by-side confusion matrices
  - figures/timing_comparison.png         - Training time comparison
  - figures/error_analysis.csv            - Misclassification analysis
  - project_report_comparison.html        - Full HTML report

Usage:
  python generate_report_v2.py

Note: This will take considerable time as it trains 4 models × 5 folds = 20 training runs.
================================================================================
"""

if __name__ == '__main__':
    logger.info(INSTRUCTIONS)
    main()
