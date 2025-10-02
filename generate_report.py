#!/usr/bin/env python3

"""
generate_report.py
Script to run the key components of the NLP project and generate a comprehensive HTML report.

This script:
1. Loads and preprocesses data
2. Fine-tunes the BioBERT model for dual lifestyle classification
3. Evaluates the model
4. Generates plots and visualizations
5. Creates an HTML report with results
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics as sk_metrics
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback, set_seed, EarlyStoppingCallback
import datasets
import numpy as np
import torch
from torch.utils.data import Dataset
import base64
from io import BytesIO
import random
# ===== COMPREHENSIVE REPRODUCIBILITY SETUP =====
# Set all seeds for reproducible results
SEED = 1998

# Set Python random seed
random.seed(SEED)

# Set NumPy random seed
np.random.seed(SEED)

# Set PyTorch random seed
torch.manual_seed(SEED)

# Set CUDA seeds (if using GPU)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # for multi-GPU

# Set cuDNN to deterministic mode (may slow down training slightly)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set transformers seed
set_seed(SEED)

# Set environment variable for Python hash seed
os.environ['PYTHONHASHSEED'] = str(SEED)

print(f"✅ All random seeds set to {SEED} for reproducible results")
# ===== END REPRODUCIBILITY SETUP =====

# Import the project modules
import sys
sys.path.append('.')

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
def plot_curves_from_log(log_history, output_path):
    if not log_history:
        print("DEBUG: No log_history provided to plot_curves_from_log")
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    steps = [entry.get('step', 0) for entry in log_history if 'loss' in entry]
    train_loss = [entry['loss'] for entry in log_history if 'loss' in entry]
    eval_loss = [entry.get('eval_loss', None) for entry in log_history if 'eval_loss' in entry]
    eval_accuracy = [entry.get('eval_accuracy', None) for entry in log_history if 'eval_accuracy' in entry]

    print(f"DEBUG plot_curves_from_log: train steps: {steps}, train losses: {train_loss}")
    print(f"DEBUG plot_curves_from_log: eval steps (non-None): {[s for s, l in zip([e.get('step',0) for e in log_history if 'eval_loss' in e], eval_loss) if l is not None]}")
    print(f"DEBUG plot_curves_from_log: eval accuracies (non-None): {[a for a in eval_accuracy if a is not None]}")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(steps, train_loss, label='Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    valid_steps = [entry.get('step', 0) for entry in log_history if 'eval_loss' in entry]
    if eval_loss:
        plt.subplot(1, 3, 2)
        plt.plot(valid_steps, eval_loss, label='Validation Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()

    if eval_accuracy:
        plt.subplot(1, 3, 3)
        plt.plot(valid_steps, eval_accuracy, label='Validation Accuracy')
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


class IncrementalPlotCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        print(f"IncrementalPlotCallback: Updating plot after evaluation. Log history length: {len(state.log_history)}")
        plot_curves_from_log(state.log_history, 'figures/training_curves.png')
        print("IncrementalPlotCallback: Plot updated successfully.")


def run_data_loading():
    """Run data loading component"""
    print("Running data loading...")
    # Import and run parse_bib.py
    import parse_bib
    parse_bib.main()
    print("Data loading completed.")

def compute_average_token_length():
    """Compute average token length of abstracts using BioBERT tokenizer"""
    print("Computing average token length...")
    tokenizer = AutoTokenizer.from_pretrained('monologg/biobert_v1.1_pubmed')
    abstracts_df = pd.read_csv('abstracts.csv')
    lengths = []
    for text in abstracts_df['abstract_text'].tolist():
        encoding = tokenizer.encode_plus(
            text,
            truncation=True,
            max_length=512,
            return_tensors=None
        )
        lengths.append(len(encoding['input_ids']))
    avg_length = sum(lengths) / len(lengths) if lengths else 0
    print(f"Average token length (after truncation to 512): {avg_length:.2f}")
    return avg_length

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    accuracy = sk_metrics.accuracy_score(labels, preds)
    return {'eval_accuracy': accuracy}

def run_classification_evaluation():
    """Run classification evaluation and return results"""
    print("Running classification evaluation...")
    model_path = 'fine_tuned_biobert_classification'
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    base_model = 'monologg/biobert_v1.1_pubmed'
    # Compute data distribution stats
    train_class_counts = train_df['label'].value_counts()
    train_total = len(train_df)
    train_percentages = (train_class_counts / train_total * 100).round(2)
    test_class_counts = test_df['label'].value_counts()
    test_total = len(test_df)
    test_percentages = (test_class_counts / test_total * 100).round(2)
    data_stats = {
        'train': {
            'total': train_total,
            'class_counts': train_class_counts.to_dict(),
            'percentages': train_percentages.to_dict()
        },
        'test': {
            'total': test_total,
            'class_counts': test_class_counts.to_dict(),
            'percentages': test_percentages.to_dict()
        }
    }
    # Force retrain by removing existing model
    if os.path.exists(model_path):
        import shutil
        try:
            shutil.rmtree(model_path)
            print(f"Successfully removed existing model directory: {model_path}")
        except OSError as e:
            print(f"Warning: Failed to remove model directory '{model_path}': {e}. Proceeding with training anyway.")

    # Check if model files exist; if not, train
    model_file = os.path.join(model_path, 'pytorch_model.bin')
    if not os.path.exists(model_file):
        print("Training classification model...")
        model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)
        # Increase dropout for regularization
        model.config.hidden_dropout_prob = 0.2
        model.config.attention_probs_dropout_prob = 0.2
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        train_texts = train_df['abstract_text'].tolist()
        train_labels = train_df['label'].tolist()
        # Split train into train and eval for monitoring
        from sklearn.model_selection import train_test_split
        train_texts, eval_texts, train_labels, eval_labels = train_test_split(
            train_texts, train_labels, test_size=0.2, random_state=42, stratify=train_labels
        )
        train_dataset = TextDataset(train_texts, train_labels, tokenizer)
        eval_dataset = TextDataset(eval_texts, eval_labels, tokenizer)
        # Compute class weights for imbalance
        from sklearn.utils.class_weight import compute_class_weight
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
        training_args = TrainingArguments(
            output_dir='./results_classification',
            logging_dir='./logs',  # Ensure logs are saved
            num_train_epochs=20,  # Allow more epochs, early stopping will halt
            per_device_train_batch_size=8,
            learning_rate=5e-5,
            weight_decay=0.05,  # Increased regularization
            lr_scheduler_type='linear',  # Learning rate scheduler
            save_steps=500,
            logging_steps=10,
            logging_strategy='steps',  # Explicit logging strategy
            eval_strategy='epoch',
            save_strategy='epoch',  # Save checkpoints for best model
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
        )
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3), IncrementalPlotCallback()],
        )
        trainer.train()
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print("Classification model trained and saved.")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Using test split with {len(test_df)} rows")
    test_texts = test_df['abstract_text'].tolist()
    test_labels = test_df['label'].tolist()
    print(f"Test texts: {len(test_texts)}, test labels: {len(test_labels)}")
    # Filter out NaN labels
    valid_indices = [i for i, label in enumerate(test_labels) if pd.notna(label)]
    test_texts = [test_texts[i] for i in valid_indices]
    test_labels = [int(test_labels[i]) for i in valid_indices]
    print(f"After filtering NaN labels: {len(test_texts)} valid samples")
    test_dataset = TextDataset(test_texts, test_labels, tokenizer)
    trainer = Trainer(model=model)
    predictions = trainer.predict(test_dataset)
    preds = torch.argmax(torch.tensor(predictions.predictions), axis=1).numpy()
    true_labels = test_labels
    accuracy = sk_metrics.accuracy_score(true_labels, preds)
    precision = sk_metrics.precision_score(true_labels, preds, average='macro')
    recall = sk_metrics.recall_score(true_labels, preds, average='macro')
    f1 = sk_metrics.f1_score(true_labels, preds, average='macro')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'data_stats': data_stats
    }, true_labels, preds

def generate_classification_curves():
    """Generate classification training curves plot"""
    print("Generating classification training curves...")
    # Try to find trainer_state in results_classification
    import glob
    trainer_state_paths = glob.glob('results_classification/checkpoint-*/trainer_state.json')
    if trainer_state_paths:
        trainer_state_path = sorted(trainer_state_paths)[-1]  # latest checkpoint
    else:
        return None
    try:
        with open(trainer_state_path, 'r') as f:
            trainer_state = json.load(f)
        log_history = trainer_state.get('log_history', [])
        if not log_history:
            return None
        steps = [entry.get('step', 0) for entry in log_history if 'loss' in entry]
        train_loss = [entry['loss'] for entry in log_history if 'loss' in entry]
        eval_loss = [entry.get('eval_loss', None) for entry in log_history if 'eval_loss' in entry]
        eval_accuracy = [entry.get('eval_accuracy', None) for entry in log_history if 'eval_accuracy' in entry]
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(steps, train_loss, label='Training Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        valid_steps = [entry.get('step', 0) for entry in log_history if 'eval_loss' in entry]
        if eval_loss:
            plt.subplot(1, 3, 2)
            plt.plot(valid_steps, eval_loss, label='Validation Loss')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.title('Validation Loss')
            plt.legend()
        if eval_accuracy:
            plt.subplot(1, 3, 3)
            plt.plot(valid_steps, eval_accuracy, label='Validation Accuracy')
            plt.xlabel('Step')
            plt.ylabel('Accuracy')
            plt.title('Validation Accuracy')
            plt.legend()
        plt.tight_layout()
        plt.savefig('figures/training_curves.png')
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        print("Training curves saved to figures/training_curves.png")
        return f"data:image/png;base64,{image_base64}"
    except:
        return None

def generate_classification_confusion_matrix(true_labels, pred_labels):
    """Generate confusion matrix plot for classification"""
    print("Generating classification confusion matrix...")
    print(f"True labels length: {len(true_labels)}")
    print(f"Pred labels length: {len(pred_labels)}")
    if len(true_labels) == 0 or len(pred_labels) == 0:
        print("No labels for classification confusion matrix, skipping...")
        return None
    labels = ['Solo', 'Dual']
    cm = sk_metrics.confusion_matrix(true_labels, pred_labels, labels=[0, 1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('figures/confusion_matrix.png')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    print("Confusion matrix saved to figures/confusion_matrix.png")
    return f"data:image/png;base64,{image_base64}"

def generate_classification_predictions():
    """Generate example classification predictions from test set"""
    print("Generating classification predictions...")
    model_path = 'fine_tuned_biobert_classification'
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    test_df = pd.read_csv('test.csv')
    # Take first 5 examples
    sample_df = test_df.head(5)
    examples = sample_df['abstract_text'].tolist()
    actual_labels = ['Dual' if int(label) == 1 else 'Solo' for label in sample_df['label'].tolist()]
    model.eval()
    predictions = []
    confidences = []
    for text in examples:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1).item()
            conf = probs[0][pred].item()
            predictions.append('Dual' if pred == 1 else 'Solo')
            confidences.append(conf)
    # Return list of dictionaries
    return [
        {
            'abstract_text': text,
            'actual_label': actual,
            'predicted_label': pred,
            'confidence': conf
        }
        for text, actual, pred, conf in zip(examples, actual_labels, predictions, confidences)
    ]

def create_html_report(classification_metrics, classification_curve_img, classification_cm_img, prediction_list, avg_token_length, data_stats):
    """Create HTML report"""
    print("Creating HTML report...")

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>BioBERT Dual Lifestyle Classification Project Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #555; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>BioBERT Dual Lifestyle Classification Project Report</h1>

        <h2>Classification Evaluation Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Accuracy</td><td>{classification_metrics.get('accuracy', 'N/A'):.4f}</td></tr>
            <tr><td>Precision</td><td>{classification_metrics.get('precision', 'N/A'):.4f}</td></tr>
            <tr><td>Recall</td><td>{classification_metrics.get('recall', 'N/A'):.4f}</td></tr>
            <tr><td>F1 Score</td><td>{classification_metrics.get('f1', 'N/A'):.4f}</td></tr>
        </table>

        <h2>Training Curves</h2>
        {f'<img src="{classification_curve_img}" alt="Classification Training Curves">' if classification_curve_img else '<p>No training curves available</p>'}

        <h2>Confusion Matrix</h2>
        <img src="{classification_cm_img}" alt="Classification Confusion Matrix">

        <h2>Data Distribution</h2>
        <h3>Training Set</h3>
        <p>Total samples: {data_stats['train']['total']}</p>
        <p>Solo: {data_stats['train']['percentages'].get(0, 0)}%, Dual: {data_stats['train']['percentages'].get(1, 0)}%</p>
        <h3>Test Set</h3>
        <p>Total samples: {data_stats['test']['total']}</p>
        <p>Solo: {data_stats['test']['percentages'].get(0, 0)}%, Dual: {data_stats['test']['percentages'].get(1, 0)}%</p>

        <h2>Average Token Length</h2>
        <p>{avg_token_length:.2f}</p>

        <h2>Example Predictions</h2>
        <ul>
        {"".join(f"<li><strong>Actual: {item['actual_label']}, Predicted: {item['predicted_label']}, Confidence: {item['confidence']:.4f}</strong><br>{item['abstract_text'][:200]}...</li>" for item in prediction_list)}
        </ul>

    </body>
    </html>
    """

    with open('project_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("HTML report saved as 'project_report.html'")

def main():
    """Main function to run the entire pipeline"""
    print("Starting BioBERT Classification Project Pipeline...")

    # Run components
    run_data_loading()

    # Compute average token length
    avg_token_length = compute_average_token_length()

    # Run classification
    classification_metrics, class_true_labels, class_pred_labels = run_classification_evaluation()
    classification_curve_img = generate_classification_curves()
    classification_cm_img = generate_classification_confusion_matrix(class_true_labels, class_pred_labels)
    prediction_list = generate_classification_predictions()

    # Create report
    create_html_report(classification_metrics, classification_curve_img, classification_cm_img, prediction_list, avg_token_length, classification_metrics['data_stats'])

    print("Pipeline completed successfully!")

# Instructions
INSTRUCTIONS = """
To run this script:

1. Ensure all dependencies are installed:
    pip install transformers datasets torch scikit-learn matplotlib seaborn pandas

2. Run the script:
    python generate_report.py

3. The script will:
    - Load and preprocess data
    - Fine-tune the BioBERT model for dual lifestyle classification (with incremental plotting of training curves after each epoch)
    - Evaluate the model
    - Generate plots and visualizations
    - Create an HTML report (project_report.html)

Note: Fine-tuning may take some time depending on your hardware. The training curves plot will update progressively during training, allowing you to monitor progress in real-time.
"""

if __name__ == '__main__':
    print(INSTRUCTIONS)
    main()