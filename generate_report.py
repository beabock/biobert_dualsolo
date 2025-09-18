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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
import datasets
import numpy as np
import torch
from torch.utils.data import Dataset
import base64
from io import BytesIO

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
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
    # Force retrain by removing existing model
    if os.path.exists(model_path):
        import shutil
        shutil.rmtree(model_path)
    if not os.path.exists(model_path):
        print("Training classification model...")
        model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)
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
        training_args = TrainingArguments(
            output_dir='./results_classification',
            num_train_epochs=10,
            per_device_train_batch_size=8,
            learning_rate=5e-5,
            weight_decay=0.01,
            save_steps=500,
            logging_steps=10,
            eval_strategy='epoch',
            save_strategy='no',
            load_best_model_at_end=False,
            metric_for_best_model='eval_loss',
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[IncrementalPlotCallback()],
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
        'f1': f1
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
        # Fallback to NER if not found
        trainer_state_path = 'results/checkpoint-6/trainer_state.json'
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
    """Generate example classification predictions"""
    print("Generating classification predictions...")
    model_path = 'fine_tuned_biobert_classification'
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    examples = [
        "This fungus can switch between saprotrophic decomposition of dead organic matter and forming mutualistic symbioses with plant roots, demonstrating a dual trophic lifestyle.",
        "This saprotrophic fungus specializes exclusively in decomposing dead plant material and cannot form symbiotic associations.",
        "The pathogenic fungus infects living plants but can also survive saprotrophically on dead plant tissues, exhibiting dual trophic modes.",
        "This obligate symbiont fungus can only survive in mutualistic association with its host plant and cannot decompose dead organic matter independently."
    ]
    model.eval()
    predictions = []
    for text in examples:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            predictions.append('Dual' if pred == 1 else 'Solo')
    html_viz = "<h3>Example Predictions</h3><ul>"
    for i, (text, pred) in enumerate(zip(examples, predictions)):
        html_viz += f"<li><strong>{pred}</strong>: {text}</li>"
    html_viz += "</ul>"
    return html_viz

def create_html_report(classification_metrics, classification_curve_img, classification_cm_img, classification_preds):
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

        <h2>Example Predictions</h2>
        {classification_preds}

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

    # Run classification
    classification_metrics, class_true_labels, class_pred_labels = run_classification_evaluation()
    classification_curve_img = generate_classification_curves()
    classification_cm_img = generate_classification_confusion_matrix(class_true_labels, class_pred_labels)
    classification_preds = generate_classification_predictions()

    # Create report
    create_html_report(classification_metrics, classification_curve_img, classification_cm_img, classification_preds)

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