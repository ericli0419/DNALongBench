# DNALongBench: A Benchmark Suite for Long-Range DNA Prediction Tasks

## Data
All data is available for download at [this link](https://cmu.app.box.com/s/cyn3tqfej3v4tg4xwv1god3jemq7916y).

The data can be downloaded via a script, see below.

## Setup
We recommend installing DNALongBench in a conda environment with Python 3.9.

1. Clone the GitHub repository:
   ```bash
   git clone https://github.com/frederikkemarin/BEND.git
    ```

2. Change to the directory:
   ```bash
   cd DNALongBench

5. Install the requirements: pip install -r requirements.txt

6. Download the data: python scripts/download_data.py

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

