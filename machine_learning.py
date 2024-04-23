import torch
from transformers import BertTokenizerFast, BertModel
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import torch.nn as nn
import torch.optim as optim
import pandas as pd

def labelize(labels, id_to_label=[]):
    label_to_id = dict()
    for i, label in enumerate(id_to_label):
        label_to_id[label] = i

    for label in labels:
        if label not in label_to_id:
            label_to_id[label] = len(id_to_label)
            id_to_label.append(label)
    
    labelsi = [label_to_id[label] for label in labels]

    return (labelsi, id_to_label)



class DataSet:
    def __init__(self, sentences, labels, labels_to_text):
        self.sentences = sentences
        self.labels = labels
        self.labels_to_text = labels_to_text
    
    def get_label_text(self, i):
        return self.labels_to_text[i]



class Data:
    def __init__(self):
        self.train_claim = pd.read_csv("data/checkworthy/english_train.tsv", sep='\t')
        self.dev_claim = pd.read_csv("data/checkworthy/english_dev.tsv", sep='\t')
        self.devtest_claim = pd.read_csv("data/checkworthy/english_dev-test.tsv", sep='\t')

        self.train_stance = pd.read_csv("data/stance/cleaned_train.tsv", sep='\t')
        self.dev_stance = pd.read_csv("data/stance/cleaned_dev.tsv", sep='\t')
    
    def get_claim(self, df, labels2=[]):
        sentences = list(df["Text"])
        (labels, labels_to_text) = labelize(df["class_label"], labels2)
        return DataSet(sentences, labels, labels_to_text)

    def get_train_claim(self, labels=[]):
        return self.get_claim(self.train_claim, labels)

    def get_dev_claim(self, labels=[]):
        return self.get_claim(self.dev_claim, labels)

    def get_devtest_claim(self, labels=[]):
        return self.get_claim(self.devtest_claim, labels)

    def get_stance(self, df, labels2=[]):
        sentences = list(df["rumor"])
        (labels, labels_to_text) = labelize(df["label"], labels2)
        return DataSet(sentences, labels, labels_to_text)

    def get_train_stance(self, labels=[]):
        return self.get_stance(self.train_stance, labels)

    def get_dev_stance(self, labels=[]):
        return self.get_stance(self.dev_stance, labels)



class MultiTaskBERT(nn.Module):
    def __init__(self, num_labels_task1, num_labels_task2):
        """
        Initialization of the multitask model.

        Parameters:
        - num_labels_task1: number of unique labels for task 1
        - num_labels_task2: number of unique labels for task 2

        Returns:
        - MultiTaskBERT: the multitask neural network with the bert encoder.
        """
        super(MultiTaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Task-specific layers
        self.dropout = nn.Dropout(0.1)
        self.classifier_task1 = nn.Linear(self.bert.config.hidden_size, num_labels_task1)
        self.classifier_task2 = nn.Linear(self.bert.config.hidden_size, num_labels_task2)
        
    def forward(self, input_ids, attention_mask, task):
        """
        Forward pass for multitask learning.
        
        Parameters:
        - input_ids: Tensor of input IDs
        - attention_mask: Tensor for attention mask
        - task: Integer specifying the task (1 for task1, 2 for task2)
        
        Returns:
        - logits: Task-specific logits
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = self.dropout(outputs.pooler_output) # zero out 10% of the output at random
        
        # Determine which task is being asked for and use the appropriate classifier
        if task == 1:
            logits = self.classifier_task1(pooled_output)
        elif task == 2:
            logits = self.classifier_task2(pooled_output)
        else:
            raise ValueError("Invalid task identifier.")
        
        return logits



class TrainingSet:
    def __init__(self, data_set, tokenizer):
        self.size = len(data_set.labels_to_text)

        # Tokenize and prepare datasets separately for each task
        self.inputs = tokenizer(data_set.sentences, padding=True, truncation=True, return_tensors="pt")
        self.labels = torch.tensor(data_set.labels)
        self.dataset = TensorDataset(self.inputs.input_ids, self.inputs.attention_mask, self.labels)

        # DataLoaders for each task
        batch_size = 2
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)



class TestSet:
    def __init__(self, data_set, tokenizer):
        # Tokenizing test data for each task
        self.inputs = tokenizer(data_set.sentences, return_tensors='pt', padding=True, truncation=True, max_length=512)
        self.labels = torch.tensor(data_set.labels)



gpu = torch.device('cpu')

data = Data()

claim_train_set = data.get_train_claim()
claim_dev_set = data.get_dev_claim(claim_train_set.labels_to_text)
claim_devtest_set = data.get_devtest_claim(claim_dev_set.labels_to_text)

stance_train_set = data.get_train_stance()
stance_dev_set = data.get_dev_stance(stance_train_set.labels_to_text)

print(claim_train_set.labels_to_text)
print(claim_dev_set.labels_to_text)
print(claim_devtest_set.labels_to_text)

print(stance_train_set.labels_to_text)
print(stance_dev_set.labels_to_text)

# Assume tokenizer is already initialized
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Data for Task 1: claim detection
task1 = TrainingSet(claim_train_set, tokenizer)

# Data for Task 2: stance detection
task2 = TrainingSet(stance_train_set, tokenizer)

# Assume the MultiTaskBERT model is already defined and initialized
model = MultiTaskBERT(num_labels_task1=task1.size, num_labels_task2=task2.size)
optimizer = optim.Adam(model.parameters(), lr=2e-5)

def handle_task(model, optimizer, batch, task_num):
    input_ids, attention_mask, labels = [item.to(device=gpu) for item in batch]
    logits = model(input_ids, attention_mask, task=task_num)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()  # Accumulate gradients

model.train() # enter training

# Training Loop Handling Different Sentences for Each Task
for epoch in range(10):
    for (batch1, batch2) in zip(task1.dataloader, task2.dataloader):

        optimizer.zero_grad()
        # Handle Task 1
        handle_task(model, optimizer, batch1, 1)

        # Handle Task 2
        handle_task(model, optimizer, batch2, 2)

        optimizer.step()  # Perform optimization step for both tasks

    print(f"Epoch {epoch+1} completed.")
    
model.eval() # enter evaluation/measurements

# Synthetic Test Data for Task 1 (Claim Detection) and Task 2 (Stance Detection)
test_task1 = TestSet(claim_dev_set, tokenizer)
test_task2 = TestSet(stance_dev_set, tokenizer)

def evaluate(model, inputs, labels, task):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        logits = model(inputs.input_ids, inputs.attention_mask, task)
        predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        accuracy = (predictions == labels).float().mean()
    return accuracy.item()

# Evaluate Task 1 (Claim Detection)
accuracy_task1 = evaluate(model, test_task1.inputs, test_task1.labels, task=1)
print(f"Task 1 (Claim Detection) Accuracy: {accuracy_task1:.4f}")

# Evaluate Task 2 (Stance Detection)
accuracy_task2 = evaluate(model, test_task2.inputs, test_task2.labels, task=2)
print(f"Task 2 (Stance Detection) Accuracy: {accuracy_task2:.4f}")

