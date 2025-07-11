import argparse
import os
import time
from typing import Dict, Tuple, Union, Optional, Callable, List, Any
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import transformers
import yaml
from datasets import (
    Dataset,
    load_dataset,
    DatasetDict,
    IterableDatasetDict,
    IterableDataset,
)
from datasets import Dataset as HFDataset, DatasetDict
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import KFold
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    PreTrainedModel,
    AutoConfig,
)

# Data
import dnalongbench
from dnalongbench.utils import load_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


root = '/work/magroup/shared/DNA_LLM/DNALongBench/'

train_loader, valid_loader, test_loader = load_data(root = root, task_name = 'enhancer_target_gene_prediction', organism = None, cell_type = None, batch_size = 1)
for batch in train_loader: 
        x, y = batch
        print('x:',x.size())
        print('y:',y.size())
        break

def collate_fn(batch, tokenizer, max_length=450000):
    """
    Custom collate function for DNA data that converts one-hot encoded sequences to raw sequences
    and tokenizes them.
    
    Args:
        batch: List of tuples where each tuple is (x, y)
               x is one-hot encoded DNA sequence of shape (seq_len, 4)
               y is gene expression data of shape (10, seq_len)
        tokenizer: The GENERator tokenizer
        max_length: Maximum sequence length for tokenization
    
    Returns:
        Dictionary with tokenized inputs and original gene expression data
    """
    # Separate x and y from the batch
    x_batch, y_batch = zip(*batch)
    
    # Convert one-hot encoded sequences to raw sequences
    raw_sequences = []
    nucleotides = ['A', 'C', 'G', 'T']
    for one_hot_seq in x_batch:
        # Ensure one_hot_seq is a PyTorch tensor
        if not isinstance(one_hot_seq, torch.Tensor):
            one_hot_seq = torch.tensor(one_hot_seq)
        
        # Get indices of 1s in one-hot encoding (argmax along axis 1)
        indices = torch.argmax(one_hot_seq, dim=1).cpu().numpy()
        
        # Convert indices to nucleotides
        raw_seq = ''.join([nucleotides[idx] for idx in indices])
        raw_sequences.append(raw_seq)
    
    # Tokenize the raw sequences
    tokenizer.padding_side = "right"
    inputs = tokenizer(
        raw_sequences,
        add_special_tokens=True,
        return_tensors="pt",
        padding=False,
        truncation=True,
        # max_length=max_length
    )

    
    # Convert y arrays to tensors and stack them
    y_tensors = []
    for y in y_batch:
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
        y_tensors.append(y)
    
    y_stacked = torch.stack(y_tensors)
    
    # Return tokenized inputs and original y
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "y": y_stacked
    }


# Set logging level for transformers
transformers.logging.set_verbosity_info()

# Define optimization direction for each metric (whether higher or lower is better)
METRICS_DIRECTION: Dict[str, str] = {
    "accuracy": "max",
    "f1_score": "max",
    "mcc": "max",
    "f1_max": "max",
    "auprc_micro": "max",
    "mse": "min",
    "mae": "min",
    "r2": "max",
    "pearson": "max",
}


def is_main_process() -> bool:
    """
    Check if current process is the main process (rank 0) in distributed training.

    Returns:
        bool: True if this is the main process, False otherwise
    """
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def dist_print(*args, **kwargs) -> None:
    """
    Print only from the main process (rank 0) in distributed training.
    Prevents duplicate outputs in multi-GPU settings.

    Args:
        *args: Arguments to pass to print function
        **kwargs: Keyword arguments to pass to print function
    """
    if is_main_process():
        print(*args, **kwargs)


def setup_tokenizer(
    model_name: str, padding_and_truncation_side: str
) -> PreTrainedTokenizer:
    """
    Load and configure tokenizer for sequence understanding.

    Args:
        model_name: Name or path of the HuggingFace model
        padding_and_truncation_side: Side for padding and truncation (left or right)

    Returns:
        PreTrainedTokenizer: Configured tokenizer for the model
    """
    dist_print(f"🔤 Loading tokenizer from: {model_name}")
    start_time = time.time()

    # Load tokenizer with trust_remote_code to support custom models
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Configure padding and truncation settings
    tokenizer.padding_side = padding_and_truncation_side
    tokenizer.truncation_side = padding_and_truncation_side

    # Set pad_token to eos_token if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dist_print(
        f"⏱️ Tokenizer loading completed in {time.time() - start_time:.2f} seconds"
    )

    return tokenizer

tokenizer = AutoTokenizer.from_pretrained("GenerTeam/GENERator-eukaryote-1.2b-base", trust_remote_code=True) # "GenerTeam/GENERator-eukaryote-3b-base"

train_loader2 = DataLoader(
        train_loader.dataset,
        batch_size=1,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length=450_000)
    )

for batch in train_loader2: 
        print(batch)
        break

print(batch['input_ids'].shape, batch['attention_mask'].shape)

class LongSequenceClassificationModel(nn.Module):
    def __init__(self, base_model_name, num_labels=2, max_subsequence_length=9375, num_subsequences=8, gradient_checkpointing=True):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)
        self.classification_head = nn.Linear(num_subsequences * self.base_model.config.hidden_size, num_labels, bias=False)
        if gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()
        self.max_subsequence_length = max_subsequence_length
        self.num_subsequences = num_subsequences

    # def forward(self, input_ids, attention_mask, labels=None):
    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        hidden_states = []

        for i in range(self.num_subsequences):
            start_idx = i * self.max_subsequence_length
            end_idx = (i + 1) * self.max_subsequence_length
            sub_input_ids = input_ids[:, start_idx:end_idx]
            sub_attention_mask = attention_mask[:, start_idx:end_idx]

            outputs = self.base_model(input_ids=sub_input_ids, attention_mask=sub_attention_mask)
            last_hidden_state = outputs.last_hidden_state
            cls_embedding = last_hidden_state[:, -1, :]
            hidden_states.append(cls_embedding)

        combined_hidden_states = torch.cat(hidden_states, dim=-1)
        logits = self.classification_head(combined_hidden_states)

        # loss = None
        # if labels is not None:
        #     loss_fn = nn.CrossEntropyLoss()
        #     loss = loss_fn(logits, labels)

        # return {"logits": logits, "loss": loss}
        return {"logits": logits}

model = LongSequenceClassificationModel(
    base_model_name="GenerTeam/GENERator-eukaryote-1.2b-base",
    num_labels=2,
    max_subsequence_length=9375,
    num_subsequences=8
)

model=model.to(torch.bfloat16).to(device)

print(model)

# Train
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Dict, Any, Optional, Callable
import time
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import os
from tqdm import tqdm
def train_model_custom(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 0,
    max_grad_norm: float = 1.0,
    save_dir: str = "/work/magroup/wenduoc/DNALongBench/experiments/GENERator/results/ETGP/v3",
    eval_steps: int = 10,
    device: str = "cuda",
    gradient_accumulation_steps: int = 1,
) -> Dict[str, Any]:
    import os

    model = model.to(device)
    model.train()

    # wrap datasets
    train_loader = DataLoader(train_loader.dataset, batch_size=1,
                              collate_fn=lambda b: collate_fn(b, tokenizer))
    val_loader   = DataLoader(val_loader.dataset,   batch_size=1,
                              collate_fn=lambda b: collate_fn(b, tokenizer))
    test_loader = DataLoader(test_loader.dataset, batch_size=1,
                                  collate_fn=lambda b: collate_fn(b, tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )
    criterion = nn.CrossEntropyLoss()

    best_auroc = 0.0
    global_step = 0

    history = {
        'train_loss': [],
        'val_loss_steps': [],
        'val_auroc_steps': [],
        'epoch_val_loss': [],
        'epoch_val_auroc': [],
        'learning_rates': [],
    }

    best_ckpt = os.path.join(save_dir, "best_model.pt")

    print(f"🚀 Training for {num_epochs} epochs, step‐eval every {eval_steps} steps, epoch‐eval each epoch.")

    start_time = time.time()
    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['y'].long().to(device)

            with torch.cuda.amp.autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)['logits']
                loss   = criterion(logits, labels) / gradient_accumulation_steps

            loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # ——— Step‐level eval & checkpoint
                if eval_steps and global_step % eval_steps == 0:
                    print(f"\n🔄 Step {global_step} eval…")
                    vm = evaluate_model_custom(model, val_loader, device)
                    auroc_s, loss_s = vm['auroc'], vm['loss']
                    history['val_auroc_steps'].append(auroc_s)
                    history['val_loss_steps'].append(loss_s)
                    print(f"  AUROC {auroc_s:.4f} | Loss {loss_s:.4f}")

                    if auroc_s > best_auroc:
                        best_auroc = auroc_s
                        torch.save(model.state_dict(), best_ckpt)
                        print(f"🏆 New best at step {global_step}: {best_ckpt}")

            epoch_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1

        # record train stats
        history['train_loss'].append(epoch_loss / num_batches)
        history['learning_rates'].append(scheduler.get_last_lr()[0])

        # ——— Epoch‐level eval & checkpoint
        print(f"\n🔄 Epoch {epoch+1} eval…")
        vm = evaluate_model_custom(model, val_loader, device)
        auroc_e, loss_e = vm['auroc'], vm['loss']
        history['epoch_val_auroc'].append(auroc_e)
        history['epoch_val_loss'].append(loss_e)
        print(f"  AUROC {auroc_e:.4f} | Loss {loss_e:.4f}")

        if auroc_e > best_auroc:
            best_auroc = auroc_e
            torch.save(model.state_dict(), best_ckpt)
            print(f"🏆 New best at epoch {epoch+1}: {best_ckpt}")

    elapsed = (time.time() - start_time) / 60
    print(f"\n✅ Done in {elapsed:.2f}min – best AUROC {best_auroc:.4f}")

    results = {
        'training_history': history,
        'best_val_auroc': best_auroc,
    }

    if test_loader:
        print("\n🧪 Final test eval…")
        tm = evaluate_model_custom(model, test_loader, device)
        results['test_metrics'] = tm
        print(f"  Test AUROC {tm['auroc']:.4f} | Loss {tm['loss']:.4f}")

    return results

from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score
)

def evaluate_model_custom(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str
) -> Dict[str, float]:
    """
    Evaluate a 2-class model that returns logits of shape [B,2].
    Uses CrossEntropyLoss internally, and softmax+argmax for preds,
    then computes AUROC/AUPRC on the P(class=1) scores.
    """
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    batches = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['y'].view(-1).long().to(device)  # [B]

            # forward
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs['logits']                              # [B,2]

            # CE loss
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            # probabilities of class=1
            probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()  # [B]
            preds = (probs > 0.5).astype(int)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
            batches += 1

    avg_loss = total_loss / batches
    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
        'auprc': auprc,
        'num_samples': len(all_labels)
    }

# Train the model
training_results = train_model_custom(
    model=model,
    tokenizer=tokenizer,
    train_loader=train_loader,
    val_loader=valid_loader,
    test_loader=test_loader,
    num_epochs=5,
    learning_rate=1e-5,
    # batch_size=1,  
    device=device, 
    gradient_accumulation_steps=16,  # Effective batch size = 8
)


# Final evaluation on test set 
final_metrics = {}

test_loader_custom = DataLoader(
            test_loader.dataset,
            batch_size=1,
            collate_fn=lambda b: collate_fn(b, tokenizer)
        )

criterion = nn.CrossEntropyLoss()

print("\n🧪 Evaluating on test set...")
test_metrics = evaluate_model_custom(model, test_loader_custom, device)
final_metrics['test_metrics'] = test_metrics

print("📊 Final Test Metrics:")
for key, value in test_metrics.items():
    print(f"  {key}: {value:.4f}")

print(final_metrics)