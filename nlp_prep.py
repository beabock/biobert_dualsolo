import datasets
from transformers import AutoTokenizer

# Load datasets
train_dataset = datasets.load_dataset('csv', data_files='train.csv')
test_dataset = datasets.load_dataset('csv', data_files='test.csv')

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('monologg/biobert_v1.1_pubmed')

# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples['abstract_text'], padding='max_length', truncation=True, max_length=512)

# Apply tokenization
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Save tokenized datasets
tokenized_train.save_to_disk('tokenized_train')
tokenized_test.save_to_disk('tokenized_test')

print("Tokenization completed and datasets saved.")