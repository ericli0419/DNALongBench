import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
import numpy as np
from scipy.stats import pearsonr
import os
from transformers import get_linear_schedule_with_warmup
import tqdm
from tqdm import tqdm
from dnalongbench.utils import get_genomes, GenomicSignalFeatures, RandomPositionsSampler
from selene_sdk.samplers.dataloader import SamplerDataLoader
from torch.utils.data import Dataset, DataLoader, Subset
import json
from functools import partial
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, BackwardPrefetch, StateDictType, ShardingStrategy

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

# 2) Load your FSDP config file
with open("fsdp.json") as f:
    cfg = json.load(f)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "GenerTeam/GENERator-eukaryote-1.2b-base"

# --------------------- data --------------------- 
root = '/work/magroup/shared/DNA_LLM/DNALongBench/'
batch_size = 4
genome, noblacklist_genome = get_genomes(root+"transcription_initiation_signal_prediction/seqs/Homo_sapiens.GRCh38.dna.primary_assembly.fa")
tfeature = GenomicSignalFeatures([root+"transcription_initiation_signal_prediction/targets/agg.plus.bw.bedgraph.bw",
root+"transcription_initiation_signal_prediction/targets/agg.encodecage.plus.v2.bedgraph.bw",
root+"transcription_initiation_signal_prediction/targets/agg.encoderampage.plus.v2.bedgraph.bw",
root+"transcription_initiation_signal_prediction/targets/agg.plus.grocap.bedgraph.sorted.merged.bw",
root+"transcription_initiation_signal_prediction/targets/agg.plus.allprocap.bedgraph.sorted.merged.bw",
root+"transcription_initiation_signal_prediction/targets/agg.minus.allprocap.bedgraph.sorted.merged.bw",
root+"transcription_initiation_signal_prediction/targets/agg.minus.grocap.bedgraph.sorted.merged.bw",
root+"transcription_initiation_signal_prediction/targets/agg.encoderampage.minus.v2.bedgraph.bw",
root+"transcription_initiation_signal_prediction/targets/agg.encodecage.minus.v2.bedgraph.bw",
root+"transcription_initiation_signal_prediction/targets/agg.minus.bw.bedgraph.bw"],
                               ['cage_plus','encodecage_plus','encoderampage_plus', 'grocap_plus','procap_plus','procap_minus','grocap_minus'
,'encoderampage_minus', 'encodecage_minus','cage_minus'],
                               (100000,),
                               [root+"transcription_initiation_signal_prediction/targets/blacklists/fantom.blacklist8.plus.bed.gz",root+"transcription_initiation_signal_prediction/targets/blacklists/fantom.blacklist8.minus.bed.gz"],
                               [0,9], [1,8], [0.61357, 0.61357])

sampler = RandomPositionsSampler(
                reference_sequence = genome,
                target= tfeature,
                features = [''],
                test_holdout=['chr8', 'chr9'],
                validation_holdout= ['chr10'],
                sequence_length= 100000,
                center_bin_to_predict= 100000,
                position_resolution=1,
                random_shift=0,
                random_strand=False
)
sampler.mode="train"
train_loader = SamplerDataLoader(sampler, num_workers=0, batch_size=batch_size, seed=3)

for batch in train_loader: 
        x, y = batch
        print('x:',x.size())
        print('y:',y.size())
        break

def collate_fn(batch, tokenizer, max_length=100_000):
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
    for one_hot_seq in x_batch:
        # Ensure one_hot_seq is a PyTorch tensor
        if not isinstance(one_hot_seq, torch.Tensor):
            one_hot_seq = torch.tensor(one_hot_seq)
            
        # Map nucleotides
        nucleotides = ['A', 'C', 'G', 'T']
        
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
        padding=True,
        truncation=True,
        max_length=max_length
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


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
max_samples = 100000
# take the first 10k examples
subset = Subset(train_loader.dataset, list(range(max_samples)))
print('Original dataset size:', len(train_loader.dataset))
print("Subset size:", len(subset))

train_sampler = DistributedSampler(subset, shuffle=True)

train_loader2 = DataLoader(
        subset,
        batch_size=1,
        sampler=train_sampler,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length=100_000),
        num_workers=0
    )

for batch in train_loader2:
    print(batch['input_ids'].shape, batch['y'].shape)
    # out = model(batch['input_ids'].to(device))
    # print(out.shape)
    break


validseq = noblacklist_genome.get_encoding_from_coords("chr10", 0, 114364328)
validcage = tfeature.get_feature_data("chr10", 0, 114364328)
class ValidDataset(Dataset):
    def __init__(self, seq, cage, window_size=100000, step_size=50000):
        """
        seq: (N, 4) numpy array, the one-hot-encoded genomic sequence
        cage: (10, N) numpy array, the target features
        window_size: int, size of the sliding window
        step_size: int, step size for the sliding window
        """
        self.seq = seq
        self.cage = cage
        self.window_size = window_size
        self.step_size = step_size
        self.num_windows = (seq.shape[0] - window_size) // step_size + 1

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        """
        Returns a tuple (input_sequence, target_features) for the idx-th window.
        """
        start = idx * self.step_size
        end = start + self.window_size
        input_seq = self.seq[start:end, :]  # Shape: (window_size, 4)
        target_cage = self.cage[:, start:end]  # Shape: (10, window_size)
        return input_seq, target_cage


# Create the validation dataset
valid_dataset = ValidDataset(validseq, validcage, window_size=100000, step_size=50000)

# Create the validation DataLoader
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda b: collate_fn(b, tokenizer, max_length=100_000))
batch = next(iter(valid_loader))
print(batch['input_ids'].shape)

# --------------------- model --------------------- 

class DNAFeatureExtractor(nn.Module):
    """
    Feature extractor that uses the DNA model to get hidden states
    and converts them to predictions through a regression head.
    Sequences longer than 10k tokens are processed in chunks.
    """
    def __init__(self, model_name, output_dim=10):
        super().__init__()
        # 1) Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # 2) Backbone in bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            output_hidden_states=True,
            trust_remote_code=True,
        )

        # 3) Regression head (float → bfloat16)
        hidden_size = self.model.config.hidden_size
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
        )
        # move & cast head to bfloat16 on the same device
        head_device = next(self.model.parameters()).device
        self.regression_head = self.regression_head.to(device=head_device, dtype=torch.bfloat16)

    def forward(self, batch, target_length=100000):
        # Unpack
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)
        else:
            input_ids = batch
            attention_mask = None

        batch_size, seq_len = input_ids.shape
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Chunked inference if too long
        # if seq_len > 10_000:
        #     chunks = []
        #     with torch.no_grad():
        #         for start in range(0, seq_len, 10_000):
        #             end = min(start + 10_000, seq_len)
        #             out = self.model(
        #                 input_ids=input_ids[:, start:end],
        #                 attention_mask=(attention_mask[:, start:end] if attention_mask is not None else None),
        #                 output_hidden_states=True
        #             )
        #             chunks.append(out.hidden_states[-1])
        #     hidden_states = torch.cat(chunks, dim=1)  # [B, seq_len, H]
        # else:
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        hidden_states = out.hidden_states[-1]

        # 5) Ensure dtype matches the head’s weights
        head_dtype = self.regression_head[0].weight.dtype
        hidden_states = hidden_states.to(head_dtype)
     

        # 6) Apply regression head per sequence
        batch_preds = []
        for i in range(batch_size):
            feats = hidden_states[i]                    # [seq_len, H] in bfloat16
            token_preds = self.regression_head(feats)   # [seq_len, D], bfloat16
            token_preds = token_preds.transpose(0, 1)   # [D, seq_len]
            if token_preds.size(1) > 1:
                token_preds = token_preds[:, 1:]        # drop BOS
            resized = nn.functional.interpolate(
                token_preds.unsqueeze(0),
                size=target_length,
                mode="linear",
                align_corners=False
            ).squeeze(0)                                # [D, target_length]
            batch_preds.append(resized)
        out = torch.stack(batch_preds, dim=0)

        return out         # [B, D, target_length]



# Initialize model
# model = DNAFeatureExtractor(model_name, output_dim=10).to(device=device)




auto_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls=(LlamaDecoderLayer,),
    # min_num_params=1e8,           # ~100M parameters threshold
)

fsdp_kwargs = {
    "auto_wrap_policy": auto_wrap_policy,
    "sync_module_states": cfg["sync_module_states"].lower() == "true",
    "sharding_strategy": ShardingStrategy[cfg["sharding_strategy"]],
    "backward_prefetch": BackwardPrefetch[cfg["backward_prefetch_policy"]],
    "forward_prefetch": cfg["forward_prefetch"].lower() == "true",
    "cpu_offload": CPUOffload(offload_params=(cfg["offload_params"].lower() == "true")),
}
# move model to its GPU, then wrap
model = DNAFeatureExtractor(model_name, output_dim=10) #.to(device)
model = FSDP(model, **fsdp_kwargs)

print(model)
print("Number of parameters:")
print(sum(p.numel() for p in model.parameters()))

# --------------------- metric --------------------- 

def compute_pcc(outputs, targets):
    """
    Compute Pearson correlation coefficient between predictions and targets
    
    Args:
        outputs: Tensor of shape [batch_size, output_dim, seq_len]
        targets: Tensor of shape [batch_size, output_dim, seq_len]
        
    Returns:
        Mean PCC across all dimensions
    """
    # Move to CPU and convert to numpy
    outputs = outputs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    
    batch_size, output_dim, seq_len = outputs.shape
    
    # Calculate PCC for each dimension
    pccs = []
    for i in range(output_dim):
        for b in range(batch_size):
            pred = outputs[b, i]
            targ = targets[b, i]
            
            # Skip if target has no variation
            if np.std(targ) < 1e-6:
                continue
                
            # Calculate PCC
            pcc, _ = pearsonr(pred, targ)
            
            # Only include valid PCCs
            if not np.isnan(pcc):
                pccs.append(pcc)
    
    # Return mean PCC if we have any valid values
    if pccs:
        return np.mean(pccs)
    else:
        return 0.0

# --------------------- trainer----------------------
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

def train_model(
    model,
    train_loader,
    valid_loader,
    test_loader=None,
    lr=5e-5,
    epochs=10,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    weight_decay=0.01
):
    """
    Train the model using the provided dataloaders
    """
    print("Training model")

    # model.to(device)

    # Only parameters with requires_grad=True will be optimized
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay
    )

    # MSE for regression
    criterion = nn.MSELoss()

    # total training steps (for scheduler)
    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_valid_pcc = -float('inf')

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_pcc  = 0.0
        optimizer.zero_grad()

        train_sampler.set_epoch(epoch)

        for step, batch in enumerate(train_loader, 1):
            sequences = batch['input_ids'] #.to(device)
            targets   = batch['y'].float() # .to(device)

            # forward + loss
            outputs = model(sequences)
            loss    = criterion(outputs, targets)
            loss    = loss / gradient_accumulation_steps
            loss.backward()

            # step & zero grads
            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # accumulate metrics
            train_loss += loss.item() * gradient_accumulation_steps
            train_pcc  += compute_pcc(
                outputs,
                targets
            )
            print(f"[Epoch {epoch+1}] Train step {step} | loss: {train_loss:.4f} | PCC: {train_pcc:.4f}")
            # optional logging
            if step % 100 == 0:
                avg_loss = train_loss / step
                avg_pcc  = train_pcc  / step
                print(f"[Epoch {epoch+1}] Train step {step} | loss: {avg_loss:.4f} | PCC: {avg_pcc:.4f}")

        # epoch-level averages
        train_loss /= len(train_loader)
        train_pcc  /= len(train_loader)

        print(f"Epoch {epoch+1}/{epochs} — Train Loss: {train_loss:.4f}, Train PCC: {train_pcc:.4f}")

        # --- Validation ---
        model.eval()
        valid_loss = 0.0
        valid_pcc  = 0.0

        with torch.no_grad():
            for batch in valid_loader:
                sequences = batch['input_ids'] #.to(device)
                targets   = batch['y'].float() #.to(device)

                outputs = model(sequences)
                loss    = criterion(outputs, targets)

                valid_loss += loss.item()
                valid_pcc  += compute_pcc(
                    outputs,
                    targets
                )

        valid_loss /= len(valid_loader)
        valid_pcc  /= len(valid_loader)

        print(f"Epoch {epoch+1}/{epochs} — Valid Loss: {valid_loss:.4f}, Valid PCC: {valid_pcc:.4f}")

        # checkpoint
        if valid_pcc > best_valid_pcc:
            best_valid_pcc = valid_pcc
            torch.save(
                model.state_dict(),
                "/work/magroup/wenduoc/DNALongBench/experiments/GENERator/results/best_dna_model2.pt"
            )
            print(f"New best model saved with PCC: {valid_pcc:.4f}")

    return model



# ------------------------Train model---------------------
model = train_model(
    model, 
    train_loader2, 
    valid_loader, 
    # test_loader2,
    lr=2e-5,
    epochs=15,
    gradient_accumulation_steps=8,  # Effective batch size of 32
    warmup_steps=200,
    weight_decay=0.01
)

print("Fine-tuning completed!")
