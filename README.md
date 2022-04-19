# PPLM Revisited: Steering and Beaming a Lumbering Mammoth to Control Text Generation
This repository contains code for our blog post at ICLR 2022: [PPLM Revisited: Steering and Beaming a Lumbering Mammoth to Control Text Generation](https://iclr-blog-track.github.io/2022/03/25/PPLM/)

This code is based on the original PPLM implementation: https://github.com/uber-research/PPLM

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

Experiments:
- Reproducibility (Section 2): [notebook](Reproduce-ML-Question-Experiment.ipynb)
- Generating questions on machine learning (Section 3): [notebook](Reproduce-ML-Question-Experiment.ipynb)
- Hyperparameter analysis: [notebook](hyperparameter-analysis.ipynb) 
- Controlling text complexity (Section 4): [notebook](Style-Experiments.ipynb)
   
## Contact

If you have any questions, please contact Van Bach Nguyen (`vanbach.nguyen[at]uni-due.de`).

If you make use of the resources developed in this blog post, please cite the work below.

```bibtex
@inproceedings{vanbach2022pplmrevisitedsteering,
  author        = {Nguyen, Van Bach and Trienes, Jan and Nauta, Meike and Pathak, Shreyasi and
                  Youssef, Paul and Imangaliyev, Sultan and Schlötterer, Jörg and Seifert, Christin},
  title         = {PPLM Revisited: Steering and Beaming a Lumbering Mammoth to Control Text Generation},
  booktitle     = {ICLR Blog Track},
  year          = {2022},
  note          = {https://iclr-blog-track.github.io/2022/03/25/PPLM/},
  url           = {https://iclr-blog-track.github.io/2022/03/25/PPLM/}
}
```
