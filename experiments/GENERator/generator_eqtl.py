import argparse
import os
import time
from typing import Dict, Tuple, Union, Optional, Callable, List, Any
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import torch
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = '/work/magroup/shared/DNA_LLM/DNALongBench/'

# Data
import torch
import dnalongbench
from dnalongbench.utils import load_data


train_loader, valid_loader, test_loader = load_data(root = root, task_name = 'eqtl_prediction', organism = None, cell_type = 'Adipose_Subcutaneous', batch_size = 1)
for batch in train_loader: 
        print('x_ref:', batch['x_ref'].size())
        print('x_alt', batch['x_alt'].size())
        print('y:',batch['y'].size())
        break
def collate_fn(batch, tokenizer, max_length=450_000):
    """
    Ultra-fast version with further optimizations for very long sequences.
    """
    nucleotides = np.array(['A', 'C', 'G', 'T'], dtype='U1')  # Single character strings
    
    def one_hot_to_sequence_ultra(one_hot_array):
        """Ultra-fast conversion using numpy operations"""
        # Check for N positions more efficiently
        row_sums = np.sum(one_hot_array, axis=1)
        n_mask = np.abs(row_sums - 1.0) > 1e-6  # N positions sum to ~1.0, others sum to 1.0
        
        # Get argmax indices
        max_indices = np.argmax(one_hot_array, axis=1)
        
        # Create sequence array
        sequence_array = nucleotides[max_indices]
        
        # Set N positions
        if np.any(n_mask):
            sequence_array[n_mask] = 'N'
        
        # Fast join using numpy
        return sequence_array.tobytes().decode('ascii')
    
    # Process batch with minimal Python loops
    sequences_data = []
    y_values = []
    
    for item in batch:
        x_ref_seq = one_hot_to_sequence_ultra(item['x_ref'])
        x_alt_seq = one_hot_to_sequence_ultra(item['x_alt'])
        sequences_data.append((x_ref_seq, x_alt_seq))
        y_values.append(item['y'])
    
    # Separate sequences for tokenization
    x_ref_sequences, x_alt_sequences = zip(*sequences_data)
    
    # Tokenize in parallel if possible
    x_ref_tokenized = tokenizer(
        list(x_ref_sequences),
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors='pt'
    )
    
    x_alt_tokenized = tokenizer(
        list(x_alt_sequences),
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors='pt'
    )
    
    # Convert y values to tensor
    y_batch = torch.tensor([y.item() if hasattr(y, 'item') else y for y in y_values])
    
    return {
        'x_ref_input_ids': x_ref_tokenized["input_ids"],
        'x_ref_attention_mask': x_ref_tokenized["attention_mask"],
        'x_alt_input_ids': x_alt_tokenized["input_ids"],
        'x_alt_attention_mask': x_alt_tokenized["attention_mask"],
        'y': y_batch
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

tokenizer = setup_tokenizer("GenerTeam/GENERator-eukaryote-1.2b-base", 'right')

train_loader2 = DataLoader(
        train_loader.dataset,
        batch_size=1,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length=450_000)
    )

for batch in train_loader2: 
        print(batch)
        break


# Model 
import torch
import torch.nn as nn
from transformers import AutoModel

class EqtlSiameseModel(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        num_labels: int = 2,
        max_subsequence_length: int = 9375,
        num_subsequences: int = 8,
        gradient_checkpointing: bool = True
    ):
        super().__init__()
        # shared encoder
        self.encoder = AutoModel.from_pretrained(
            base_model_name, trust_remote_code=True
        )
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        self.max_sub_len = max_subsequence_length
        self.num_subseqs = num_subsequences
        hidden_size = self.encoder.config.hidden_size * self.num_subseqs

        # [allele; ref; |allele–ref|] → logits
        self.classification_head = nn.Linear(3 * hidden_size, num_labels, bias=False)

    def _encode(self, input_ids: torch.LongTensor):
        """
        Break into chunks, encode each, grab final token embedding,
        concat along seq‐chunks.
        """
        seq_states = []
        for i in range(self.num_subseqs):
            start = i * self.max_sub_len
            end   = (i + 1) * self.max_sub_len

            chunk_ids = input_ids[:, start:end]
            # create a full‐ones mask so every token is attended
            chunk_mask = torch.ones_like(chunk_ids)

            out = self.encoder(input_ids=chunk_ids, attention_mask=chunk_mask)
            # final token as CLS proxy
            cls_emb = out.last_hidden_state[:, -1, :]  # [B, hidden]
            seq_states.append(cls_emb)

        return torch.cat(seq_states, dim=-1)  # [B, num_subseqs*hidden]

    def forward(
        self,
        x_alt: torch.LongTensor,   # your “allele” seqs
        x_ref: torch.LongTensor,   # your “reference” seqs
    ):
        emb_alt = self._encode(x_alt)
        emb_ref = self._encode(x_ref)

        delta = torch.abs(emb_alt - emb_ref)
        features = torch.cat([emb_alt, emb_ref, delta], dim=-1)  # [B, 3*H]
        logits = self.classification_head(features)
        return {"logits": logits}

model = EqtlSiameseModel(
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

def train_model_custom(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,       # still used by collate_fn
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 0,
    max_grad_norm: float = 1.0,
    save_dir: str = "/work/magroup/wenduoc/DNALongBench/experiments/GENERator/results/EQTL/v3",
    eval_steps: int = 10,
    device: str = "cuda",
    gradient_accumulation_steps: int = 1,
) -> Dict[str, Any]:
    model = model.to(device)
    model.train()

    # wrap datasets with your collate_fn (returns x_alt, x_ref, y)
    train_loader = DataLoader(train_loader.dataset, batch_size=1,
                              collate_fn=lambda b: collate_fn(b, tokenizer))
    val_loader   = DataLoader(val_loader.dataset,   batch_size=1,
                              collate_fn=lambda b: collate_fn(b, tokenizer))
    if test_loader is not None:
        test_loader = DataLoader(test_loader.dataset, batch_size=1,
                                  collate_fn=lambda b: collate_fn(b, tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )
    criterion = nn.CrossEntropyLoss()

    best_auroc   = 0.0
    global_step  = 0
    best_ckpt    = os.path.join(save_dir, "best_model.pt")
    history = {
        'train_loss': [],
        'val_loss_steps': [],
        'val_auroc_steps': [],
        'epoch_val_loss': [],
        'epoch_val_auroc': [],
        'learning_rates': [],
    }

    print(f"🚀 Training for {num_epochs} epochs, step‐eval every {eval_steps} steps, epoch‐eval each epoch.")
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
        model.train()
        epoch_loss   = 0.0
        num_batches  = 0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            alt_ids = batch['x_alt_input_ids'].to(device)
            ref_ids = batch['x_ref_input_ids'].to(device)
            labels  = batch['y'].long().to(device)

            with torch.cuda.amp.autocast():
                outputs = model(alt_ids, ref_ids)
                logits  = outputs['logits']
                loss    = criterion(logits, labels) / gradient_accumulation_steps

            loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # — Step‐level eval & checkpoint —
                if eval_steps and global_step % eval_steps == 0:
                    print(f"\n🔄 Step {global_step} eval…")
                    vm = evaluate_eqtl_model(model, val_loader, device, criterion)
                    auroc_s, loss_s = vm['auroc'], vm['loss']
                    history['val_auroc_steps'].append(auroc_s)
                    history['val_loss_steps'].append(loss_s)
                    print(f"  AUROC {auroc_s:.4f} | Loss {loss_s:.4f}")

                    if auroc_s > best_auroc:
                        best_auroc = auroc_s
                        torch.save(model.state_dict(), best_ckpt)
                        print(f"🏆 New best at step {global_step}: {best_ckpt}")

            epoch_loss  += loss.item() * gradient_accumulation_steps
            num_batches += 1

        # record train stats
        history['train_loss'].append(epoch_loss / num_batches)
        history['learning_rates'].append(scheduler.get_last_lr()[0])

        # — Epoch‐level eval & checkpoint —
        print(f"\n🔄 Epoch {epoch+1} eval…")
        vm = evaluate_eqtl_model(model, val_loader, device, criterion)
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

    if test_loader is not None:
        print("\n🧪 Final test eval…")
        tm = evaluate_eqtl_model(model, test_loader, device)
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

def evaluate_eqtl_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    criterion: nn.Module
) -> Dict[str, float]:
    """
    Evaluate a Siamese eQTL classification model on a dataset.
    Assumes model(batch['x_alt'], batch['x_ref']) → {'logits': Tensor[B,2]}.
    
    Args:
        model:        Siamese eQTL model
        data_loader:  DataLoader yielding {'x_alt', 'x_ref', 'y'}
        device:       torch device
        criterion:    loss function (e.g. CrossEntropyLoss())
    
    Returns:
        dict with loss, accuracy, precision, recall, f1, auc, auprc, num_samples
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    batches = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            x_alt = batch['x_alt_input_ids'].to(device)
            x_ref = batch['x_ref_input_ids'].to(device)
            labels = batch['y'].long().to(device)

            # forward pass
            outputs = model(x_alt, x_ref)
            logits = outputs['logits']      # [B, 2]
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # probabilities for class=1
            probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
            batches += 1

    # average loss
    avg_loss = total_loss / batches

    # convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    # compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    auc   = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auc,
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
    num_epochs=3,
    learning_rate=1e-5,
    # batch_size=1,  # Start small due to memory constraints
    device=device,
    gradient_accumulation_steps=16,  # Effective batch size = 1 * 8 = 8
)

# Final evaluation on test set if provided
final_metrics = {}

test_loader_custom = DataLoader(
            test_loader.dataset,
            batch_size=1,
            collate_fn=lambda b: collate_fn(b, tokenizer)
        )

criterion = nn.CrossEntropyLoss()

print("\n🧪 Evaluating on test set...")
test_metrics = evaluate_eqtl_model(model, test_loader_custom, device, criterion)
final_metrics['test_metrics'] = test_metrics

print("📊 Final Test Metrics:")
for key, value in test_metrics.items():
    print(f"  {key}: {value:.4f}")

print(final_metrics)