import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Subset, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    CPUOffload,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from functools import partial
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

# Import your existing code
from dnalongbench.utils import get_genomes, GenomicSignalFeatures, RandomPositionsSampler
from selene_sdk.samplers.dataloader import SamplerDataLoader

# Setup argument parser
parser = argparse.ArgumentParser(description='FSDP Training for DNA Feature Extraction')
parser.add_argument('--local_rank', '--local-rank', type=int, default=-1, help='Local rank for distributed training')
parser.add_argument('--config', type=str, default='fsdp.json', help='FSDP configuration file')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU')
parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Gradient accumulation steps')
parser.add_argument('--warmup_steps', type=int, default=200, help='Warmup steps')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
parser.add_argument('--max_samples', type=int, default=100000, help='Maximum number of samples to use')
parser.add_argument('--save_dir', type=str, default='/work/magroup/wenduoc/DNALongBench/experiments/GENERator/results/', help='Directory to save model')
args = parser.parse_args()

# Function to load FSDP config
def load_fsdp_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# Initialize distributed environment
def init_distributed():
    # Check if environment variables are set
    if "LOCAL_RANK" not in os.environ:
        # Try to get local_rank from args
        from sys import argv
        for arg in argv:
            if "--local-rank=" in arg:
                os.environ["LOCAL_RANK"] = arg.split("=")[1]
                break
            elif "--local_rank=" in arg:
                os.environ["LOCAL_RANK"] = arg.split("=")[1]
                break
    
    # Initialize the process group
    dist.init_process_group(backend="nccl")
    
    # Set the device for this process
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank

# # Function to setup FSDP wrapping policy based on config
# def setup_fsdp_wrapping_policy(config, model_name):
#     # Identify the transformer layer class based on the model name
#     if "GENERator" in model_name:
#         # You may need to adjust this based on the actual class name in the GENERator architecture
#         transformer_layer_cls = config.get("transformer_layer_cls_to_wrap", None)
#         if transformer_layer_cls == "LlamaDecoderLayer":
#             # For GENERator model, find the appropriate layer class
#             from transformers.models.llama.modeling_llama import LlamaDecoderLayer
#             transformer_layer_cls = LlamaDecoderLayer
#     else:
#         # Default to a generic approach if specific layer class is not identified
#         return size_based_auto_wrap_policy(min_num_params=1e6)
    
#     # Return the transformer-based wrapping policy
#     return transformer_auto_wrap_policy(transformer_layer_cls=transformer_layer_cls)

# Function to get FSDP MixedPrecision config
def get_mixed_precision_config():
    return MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

# Define the data collate function
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

# Define the model class
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
        
    def forward(self, batch, target_length=100000):
        # Unpack
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)
        else:
            input_ids = batch
            attention_mask = None

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

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
        #     with torch.no_grad():
        #         out = self.model(
        #             input_ids=input_ids,
        #             attention_mask=attention_mask,
        #             output_hidden_states=True
        #         )
        #     hidden_states = out.hidden_states[-1]
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids.to(device),
                attention_mask=(attention_mask.to(device) if attention_mask is not None else None),
                output_hidden_states=True
            )
        hidden_states = out.hidden_states[-1]  # [B, seq_len, H]

        # 5) Ensure dtype matches the head's weights
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


# Function to compute PCC metric
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

# Main training function
def train_model_fsdp(
    model,
    train_loader,
    valid_loader,
    test_loader=None,
    lr=5e-5,
    epochs=10,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    weight_decay=0.01,
    local_rank=0,
    save_dir="/work/magroup/wenduoc/DNALongBench/experiments/GENERator/results/"
):
    """
    Train the model using FSDP
    """
    # Only parameters with requires_grad=True will be optimized
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay
    )

    # MSE for regression
    criterion = nn.MSELoss()

    # Total training steps (for scheduler)
    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_valid_pcc = -float('inf')
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_pcc = 0.0
        optimizer.zero_grad()

        # Set the sampler's epoch for shuffling
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        # Using tqdm only on the main process
        if local_rank == 0:
            train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        else:
            train_iter = train_loader

        for step, batch in enumerate(train_iter, 1):
            sequences = batch['input_ids']
            targets = batch['y'].float()

            # Forward + loss
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss = loss / gradient_accumulation_steps
            
            # Backward
            loss.backward()

            # Step & zero grads
            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Accumulate metrics
            train_loss += loss.item() * gradient_accumulation_steps
            batch_pcc = compute_pcc(outputs, targets)
            train_pcc += batch_pcc

            # Optional logging on rank 0
            if local_rank == 0 and step % 20 == 0:
                avg_loss = train_loss / step
                avg_pcc = train_pcc / step
                if isinstance(train_iter, tqdm):
                    train_iter.set_postfix({"loss": f"{avg_loss:.4f}", "PCC": f"{avg_pcc:.4f}"})

        # Compute epoch-level averages
        train_loss = train_loss / len(train_loader)
        train_pcc = train_pcc / len(train_loader)

        # All-reduce to get global averages across all processes
        if dist.is_initialized():
            world_size = dist.get_world_size()
            loss_tensor = torch.tensor([train_loss]).cuda()
            pcc_tensor = torch.tensor([train_pcc]).cuda()
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(pcc_tensor, op=dist.ReduceOp.SUM)
            train_loss = loss_tensor.item() / world_size
            train_pcc = pcc_tensor.item() / world_size

        if local_rank == 0:
            print(f"Epoch {epoch+1}/{epochs} — Train Loss: {train_loss:.4f}, Train PCC: {train_pcc:.4f}")

        # --- Validation ---
        model.eval()
        valid_loss = 0.0
        valid_pcc = 0.0

        # Using tqdm only on the main process
        if local_rank == 0:
            valid_iter = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]")
        else:
            valid_iter = valid_loader

        with torch.no_grad():
            for batch in valid_iter:
                sequences = batch['input_ids']
                targets = batch['y'].float()

                outputs = model(sequences)
                loss = criterion(outputs, targets)

                valid_loss += loss.item()
                batch_pcc = compute_pcc(outputs, targets)
                valid_pcc += batch_pcc

        # Compute epoch-level averages
        valid_loss = valid_loss / len(valid_loader)
        valid_pcc = valid_pcc / len(valid_loader)

        # All-reduce to get global averages
        if dist.is_initialized():
            world_size = dist.get_world_size()
            loss_tensor = torch.tensor([valid_loss]).cuda()
            pcc_tensor = torch.tensor([valid_pcc]).cuda()
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(pcc_tensor, op=dist.ReduceOp.SUM)
            valid_loss = loss_tensor.item() / world_size
            valid_pcc = pcc_tensor.item() / world_size

        if local_rank == 0:
            print(f"Epoch {epoch+1}/{epochs} — Valid Loss: {valid_loss:.4f}, Valid PCC: {valid_pcc:.4f}")

            # Save checkpoint if it's the best so far
            if valid_pcc > best_valid_pcc:
                best_valid_pcc = valid_pcc
                
                # For FSDP, we need to use special handling for state dict
                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                    model_state = model.state_dict()
                    if local_rank == 0:
                        torch.save(
                            model_state,
                            os.path.join(save_dir, "best_dna_model_fsdp.pt")
                        )
                        print(f"New best model saved with PCC: {valid_pcc:.4f}")

    return model

def main():
    # Initialize distributed environment
    local_rank = init_distributed()
    world_size = dist.get_world_size()
    
    # Parse arguments
    args_dict = vars(args)
    # Override local_rank from environment if not specified in args
    if args.local_rank == -1:
        args_dict['local_rank'] = local_rank
    else:
        local_rank = args.local_rank
        
    if local_rank == 0:
        print(f"Starting FSDP training with {world_size} GPUs")
    
    # Load FSDP config
    fsdp_config = load_fsdp_config(args.config)
    
    # Define model name
    model_name = "GenerTeam/GENERator-eukaryote-1.2b-base"
    
    # Set up data paths
    root = '/work/magroup/shared/DNA_LLM/DNALongBench/'
    
    # Get genomes and features
    if local_rank == 0:
        print("Loading genomes and features...")
    
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

    # Create sampler
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
    train_loader = SamplerDataLoader(sampler, num_workers=0, batch_size=4, seed=3)

    
    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Create subset of data
    max_samples = args.max_samples
    subset = Subset(train_loader.dataset, list(range(max_samples)))
    print('Original dataset size:', len(train_loader.dataset))
    print("Subset size:", len(subset))

    train_sampler = DistributedSampler(subset, shuffle=True)

    train_loader = DataLoader(
            subset,
            batch_size=1,
            sampler=train_sampler,
            collate_fn=lambda b: collate_fn(b, tokenizer, max_length=100_000),
            num_workers=0
        )

    for batch in train_loader:
        print(batch['input_ids'].shape, batch['y'].shape)
        # out = model(batch['input_ids'].to(device))
        # print(out.shape)
        break

    # Create validation dataset
    validseq = noblacklist_genome.get_encoding_from_coords("chr10", 0, 114364328)
    validcage = tfeature.get_feature_data("chr10", 0, 114364328)
    
    class ValidDataset(Dataset):
        def __init__(self, seq, cage, window_size=100000, step_size=50000):
            self.seq = seq
            self.cage = cage
            self.window_size = window_size
            self.step_size = step_size
            self.num_windows = (seq.shape[0] - window_size) // step_size + 1

        def __len__(self):
            return self.num_windows

        def __getitem__(self, idx):
            start = idx * self.step_size
            end = start + self.window_size
            input_seq = self.seq[start:end, :]  # Shape: (window_size, 4)
            target_cage = self.cage[:, start:end]  # Shape: (10, window_size)
            return input_seq, target_cage
    
    # Create validation dataset
    valid_dataset = ValidDataset(validseq, validcage, window_size=100000, step_size=50000)
    
    # Create validation sampler and dataloader
    # valid_sampler = DistributedSampler(
    #     valid_dataset,
    #     num_replicas=world_size,
    #     rank=local_rank,
    #     shuffle=False,
    #     seed=42
    # )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        # sampler=valid_sampler,
        num_workers=4, 
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length=100_000)
    )
    
    # Initialize model
    model = DNAFeatureExtractor(model_name, output_dim=10)
    
    # Unfreeze layers for training
    # model.unfreeze_layers(num_layers=3)
    
    if local_rank == 0:
        print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    
    # Configure FSDP wrapping
    # auto_wrap_policy = setup_fsdp_wrapping_policy(fsdp_config, model_name)
    auto_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls=(LlamaDecoderLayer,),
    # min_num_params=1e8,           # ~100M parameters threshold
    )

    # Configure mixed precision
    mixed_precision_config = get_mixed_precision_config()
    
    # Configure backward prefetch
    backward_prefetch = BackwardPrefetch.BACKWARD_PRE if fsdp_config.get("backward_prefetch_policy") == "BACKWARD_PRE" else None
    
    # Configure sharding strategy
    sharding_strategy_mapping = {
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    sharding_strategy = sharding_strategy_mapping.get(fsdp_config.get("sharding_strategy"), ShardingStrategy.SHARD_GRAD_OP)
    
    # Configure CPU offload
    cpu_offload = CPUOffload(offload_params=fsdp_config.get("offload_params", False) == "true")
    
    # Wrap model with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_config,
        device_id=local_rank,
        sharding_strategy=sharding_strategy,
        backward_prefetch=backward_prefetch,
        cpu_offload=cpu_offload,
        use_orig_params=fsdp_config.get("use_orig_params", "true") == "true",
        sync_module_states=fsdp_config.get("sync_module_states", "true") == "true",
    )
    
    if local_rank == 0:
        print("Model wrapped with FSDP")
    
    # Train model
    train_model_fsdp(
        model,
        train_loader,
        valid_loader,
        lr=args.lr,
        epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        local_rank=local_rank,
        save_dir=args.save_dir
    )
    
    if local_rank == 0:
        print("FSDP Training completed!")
    
    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()