<h1>DNALongBench: A Benchmark Suite for Long-Range DNA Prediction Tasks</h1>

<h2>Introduction</h2>

DNALongBench is a benchmark of realistic and biologically meaningful genomic DNA prediction tasks that require long-range sequence input and involve long-range dependencies. There are five tasks in our DNALongBench.

![image](./Figure1.v1.png)

<h2>Data Download</h2>

The data for each task could be downloaded via the following link, and the corresponding dataloader file is also provided. Therefore, you could run your own experiments by just replacing the dataloader files. 

<h3>Regulatory Sequence Activity Prediction</h3>

<h4>Data Link</h4>

The data can be downloaded at <a href="https://dataverse.harvard.edu/privateurl.xhtml?token=4c6b250c-26fc-412a-b3e1-bc15f1332f0c">Regulatory Sequence Activity Prediction</a>. 

<h4>Data Details</h4>

We provide the sequences.bed, statistics.json, hg38.ml.fa.fai and hg38.ml.fa.gz files, and reformulate these data into train*/valid*/test*.tfr. The only files you needed are the corresponding tfr files.

<h3>Transcription Initiation Signal Prediction</h3>

<h4>Data Link</h4>

The data can be downloaded at <a href="https://dataverse.harvard.edu/privateurl.xhtml?token=9810103a-b8b8-4a4d-95c4-b26b6e153446">Transcription Initiation Signal Prediction</a>. 

<h4>Data Details</h4>

We provide all the correspoding bed files.

<h3>Enhancer-Target Gene Prediction</h3>

<h4>Data Link</h4>

The data can be downloaded at <a href="https://dataverse.harvard.edu/privateurl.xhtml?token=c238c0dd-528f-4d04-a3c8-0ff1eee1d651">Enhancer-Target Gene Prediction</a>. 

<h4>Data Details</h4>

The sequences, fa and metrics data are provided.

<h3>Contact Map Prediction</h3>

<h4>Data Link</h4>

The data can be downloaded at <a href="https://dataverse.harvard.edu/privateurl.xhtml?token=a990b515-d76e-4b63-ba74-5c78c469ae53">Contact Map Prediction</a>. 

<h4>Data Details</h4>

We provide the well-split train/valid/test files.

<h3>eQTL</h3>

<h4>Data Link</h4>

The data can be downloaded at <a href="https://dataverse.harvard.edu/privateurl.xhtml?token=93d446a5-9c75-44bf-be1c-7622563c48d0">eQTL</a>. 

<h4>Data Details</h4>

The corresponding bed files are provided. 


<!--
### 1. [Regulatory Sequence Activity Prediction](https://dataverse.harvard.edu/privateurl.xhtml?token=4c6b250c-26fc-412a-b3e1-bc15f1332f0c)

### 2. [Transcription Initiation Signal Prediction](https://dataverse.harvard.edu/privateurl.xhtml?token=9810103a-b8b8-4a4d-95c4-b26b6e153446)

### 3. [Enhancer-Target Gene Prediction](https://dataverse.harvard.edu/privateurl.xhtml?token=c238c0dd-528f-4d04-a3c8-0ff1eee1d651)

### 4. [Contact Map Data](https://dataverse.harvard.edu/privateurl.xhtml?token=a990b515-d76e-4b63-ba74-5c78c469ae53)

### 5. [eQTL Data](https://dataverse.harvard.edu/privateurl.xhtml?token=93d446a5-9c75-44bf-be1c-7622563c48d0)
-->


<h2>Experiments</h2>

We've provided the performance of three types of models, which are Expert Model, a lightweight CNN baseline, and a finetuned DNA foundation model (HyenaDNA, Caduceus-Ph and Caduceus-PS). We'll introduce below how to run these models by taking the task of Enhancer-Target Gene Prediction (ETGP) as an example.

| Model |   Expert Model   |  CNN   |  HyenaDNA  |  Caduceus-Ph  |   Caduceus-PS    |   
|:---------------|:---------:|:---------:|:---------:|:---------:|:----------:|
| ETGP        |   **0.926**   |  0.797   |   0.828    |   0.826    |   0.821    |   

<Download Code>

Following the commands below to download our code:

```ruby
git clone git@github.com:wenduocheng/DNALongBench.git
cd DNALongBench
```

<h2>CNN</h2>

<h3>Environment Setup</h3>

The dependencies can be set up using the following commands:

```ruby
conda create -n cnn python=3.8 -y 
conda activate cnn 
bash setup.sh 
```

<h3>Training</h3>

Following the guidance provided at experiments/CNN/SimpleCNN.ipynb

<h2>HyenaDNA</h2>

<h3>Environment Setup</h3>

We used the official code of HyenaDNA. The environment setup can be found at <a href="https://github.com/HazyResearch/hyena-dna?tab=readme-ov-file#dependencies">HyenaDNA Enviroment Eetup</a>.

Be careful if you would like to use flash attention. Sometimes there are some issues when installing flash attention. We recommend first setup the environment, then activate the enviroment, and finally install flash attention inside the environment. 

<h3>Training</h3>

To finetune the model on ETGP task: 

```ruby
bash experiments/HyenaDNA_ETGP_CMP_eQTLP/train.sh
```

<h2>Caduceus</h2>

<h3>Environment Setup</h3>

We used the offical code of Caduceus. The environment setup can be found at <a href="https://github.com/kuleshov-group/caduceus?tab=readme-ov-file#getting-started-in-this-repository">Caduceus Enviroment Eetup</a>.

When meeting issues, please use the similar solutions as suggested in HyenaDNA enviroment setup.

<h3>Training</h3>

The backbone model of Caduceus is similar to HyenaDNA. Please follow the similar training script with pretrained model replaced with Caduceus. 

<h2>Citation</h2>
If you find our work helpful, please consider citing our paper.

```
@inproceedings{chengdna,
  title={DNALongBench: A Benchmark Suite for Long-Range DNA Prediction Tasks},
  author={Cheng, Wenduo and Song, Zhenqiao and Zhang, Yang and Wang, Shike and Wang, Danqing and Yang, Muyu and Li, Lei and Ma, Jian}
}
```

<!--
## Setup
We recommend installing DNALongBench in a conda environment with Python 3.9.

1. Clone the GitHub repository

2. Change to the directory:
   ```bash
   cd DNALongBench
   ```

3. To run the code, install the dependencies:
   ```bash
   pip install .
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
-->

