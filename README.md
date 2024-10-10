<h1>DNALongBench: A Benchmark Suite for Long-Range DNA Prediction Tasks</h1>

<h2>Introduction</h2>

DNALongBench is a benchmark of realistic and biologically meaningful genomic DNA prediction tasks that require long-range sequence input and involve long-range dependencies. There are five tasks in our DNALongBench.

![image](./Figure1.v1.png)

<h2>Data Download</h2>

The data for each task could be downloaded via the following link, and the corresponding dataloader file is also provided. Therefore, you could run your own experiments by just replacing the dataloader files. 

### 1. [Regulatory Sequence Activity Prediction](https://dataverse.harvard.edu/privateurl.xhtml?token=4c6b250c-26fc-412a-b3e1-bc15f1332f0c)

### 2. [Transcription Initiation Signal Prediction](https://dataverse.harvard.edu/privateurl.xhtml?token=9810103a-b8b8-4a4d-95c4-b26b6e153446)

### 3. [Enhancer-Target Gene Prediction](https://dataverse.harvard.edu/privateurl.xhtml?token=c238c0dd-528f-4d04-a3c8-0ff1eee1d651)

### 4. [Contact Map Data](https://dataverse.harvard.edu/privateurl.xhtml?token=a990b515-d76e-4b63-ba74-5c78c469ae53)

### 5. [eQTL Data](https://dataverse.harvard.edu/privateurl.xhtml?token=93d446a5-9c75-44bf-be1c-7622563c48d0)


## Setup
We recommend installing DNALongBench in a conda environment with Python 3.8.

1. Clone the GitHub repository

2. Change to the directory:
   ```bash
   cd DNALongBench
   ```

3. To run the code, install the dependencies:
   ```bash
   sh setup.sh 
   ```

4. Download the data:
   ```bash
   python scripts/download_data.py
   ```
## Data Loaders
We provide data loaders for each task in scripts/data_loaders.py.

## Experiments
### HyenaDNA

### CNN
We provide the CNN model for each task in experiments/CNN/SimpleCNN.ipynb.

## Citation 
The datasets included in DNALongBench were collected from various sources. Citing the corresponding original sources is required when using the data provided with DNALongBench.

### Enhancer-Target Gene Prediction
```bibtex
@article{fulco2019activity,
  title={Activity-by-contact model of enhancer--promoter regulation from thousands of CRISPR perturbations},
  author={Fulco, Charles P and Nasser, Joseph and Jones, Thouis R and Munson, Glen and Bergman, Drew T and Subramanian, Vidya and Grossman, Sharon R and Anyoha, Rockwell and Doughty, Benjamin R and Patwardhan, Tejal A and others},
  journal={Nature genetics},
  volume={51},
  number={12},
  pages={1664--1669},
  year={2019},
  publisher={Nature Publishing Group US New York}
}
```

### Contact Map Prediction

```bibtex
@article{fudenberg2020predicting,
  title={Predicting 3D genome folding from DNA sequence with Akita},
  author={Fudenberg, Geoff and Kelley, David R and Pollard, Katherine S},
  journal={Nature methods},
  volume={17},
  number={11},
  pages={1111--1117},
  year={2020},
  publisher={Nature Publishing Group US New York}
}
```

### eQTL prediction
```bibtex
@article{avsec2021effective,
  title={Effective gene expression prediction from sequence by integrating long-range interactions},
  author={Avsec, {\v{Z}}iga and Agarwal, Vikram and Visentin, Daniel and Ledsam, Joseph R and Grabska-Barwinska, Agnieszka and Taylor, Kyle R and Assael, Yannis and Jumper, John and Kohli, Pushmeet and Kelley, David R},
  journal={Nature methods},
  volume={18},
  number={10},
  pages={1196--1203},
  year={2021},
  publisher={Nature Publishing Group US New York}
}
```

### Regulatory Sequence Activity Prediction
```bibtex
@article{avsec2021effective,
  title={Effective gene expression prediction from sequence by integrating long-range interactions},
  author={Avsec, {\v{Z}}iga and Agarwal, Vikram and Visentin, Daniel and Ledsam, Joseph R and Grabska-Barwinska, Agnieszka and Taylor, Kyle R and Assael, Yannis and Jumper, John and Kohli, Pushmeet and Kelley, David R},
  journal={Nature methods},
  volume={18},
  number={10},
  pages={1196--1203},
  year={2021},
  publisher={Nature Publishing Group US New York}
}
```

### Transcription Initiation Signal Prediction
```bibtex
@article{dudnyk2024sequence,
  title={Sequence basis of transcription initiation in the human genome},
  author={Dudnyk, Kseniia and Cai, Donghong and Shi, Chenlai and Xu, Jian and Zhou, Jian},
  journal={Science},
  volume={384},
  number={6694},
  pages={eadj0116},
  year={2024},
  publisher={American Association for the Advancement of Science}
}
```

