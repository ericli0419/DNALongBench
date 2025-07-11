

# CMP: Contact Map Prediction  
**Supported subsets:**  
- `HFF`  
- `H1hESC`  
- `GM12878`  
- `IMR90`  
- `HCT116`  

```bash
python train.py \
  --task_name contact_map_prediction \
  --save_dir ./experiments/CNN/results/CMP/GM12878 \
  --subset GM12878 \
  --batch_size 2 \
  --num_epochs 5
```


# ETGP: enhancer_target_gene_prediction
'''
python train.py --task_name enhancer_target_gene_prediction --save_dir /work/magroup/wenduoc/DNALongBench/experiments/CNN/results/ETGP --subset None --batch_size 2
'''

# EQTL: eQTL Prediction  
**Supported cell types:**  
- `Adipose_Subcutaneous`  
- `Artery_Tibial`  
- `Cells_Cultured_fibroblasts`  
- `Muscle_Skeletal`  
- `Nerve_Tibial`  
- `Skin_Not_Sun_Exposed_Suprapubic`  
- `Skin_Sun_Exposed_Lower_leg`  
- `Thyroid`  
- `Whole_Blood`  

```bash
python train.py \
  --task_name eqtl_prediction \
  --save_dir /work/magroup/wenduoc/DNALongBench/experiments/CNN/results/EQTL/Adipose_Subcutaneous \
  --subset Adipose_Subcutaneous \
  --batch_size 2


# RSAP: Regulatory Sequence Activity Prediction
Specify a subset: human, mouse
'''
python train.py --task_name regulatory_sequence_activity --save_dir /work/magroup/wenduoc/DNALongBench/experiments/CNN/results/RSAP/mouse --subset mouse --batch_size 2 --num_epochs 5
'''

# TISP: transcription_initiation_signal_prediction
'''
python train.py --task_name transcription_initiation_signal_prediction --save_dir /work/magroup/wenduoc/DNALongBench/experiments/CNN/results/TISP --subset None --batch_size 2 --num_epochs 5
'''
