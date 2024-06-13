# DNALongBench: A Benchmark Suite for Long-Range DNA Prediction Tasks

The DNALongBench is a collection of realistic and biologically meaningful genomic DNA prediction tasks that require long-range sequence input and involve long-range dependencies.
This GitHub repository is under active construction. 

## Data
All data is available for download at [this link](https://cmu.app.box.com/s/cyn3tqfej3v4tg4xwv1god3jemq7916y).

The data can be downloaded via a script, see below.

## Setup
We recommend installing DNALongBench in a conda environment with Python 3.8.

1. Clone the GitHub repository:
   ```bash
   git clone https://github.com/wenduocheng/DNALongBench.git
    ```

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

