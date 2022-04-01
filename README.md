#PPLM Revisited: Steering and Beaming a Lumbering Mammoth to Control Text Generation
This repository contains code for our blog post at ICLR 2022: [PPLM Revisited: Steering and Beaming a Lumbering Mammoth to Control Text Generation](https://mcmi-group.github.io/iclr-blog-track.github.io//2022/03/25/PPLM/)

This code is based on https://github.com/uber-research/PPLM

## Computational Environment

Install dependencies via conda:

```sh
conda env update -f environment.yml
conda activate pplm
```

Start jupyter notebook:

```sh
jupyter notebook
```
- Experiments 
   - Reproducibility (section 2), Generating questions on machine learning (Section 3): [notebook](Reproduce-ML-Question-Experiment.ipynb)
    - Hyperparameter analysis: [notebook](hyperparameter-analysis.ipynb) 
   - Controlling text complexity (style)(section 4): [notebook](Style-Experiments.ipynb)
   