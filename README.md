# Overview

This is the official code repository for the paper ["Unearthing Skill-level Insights for Understanding Tradeoffs of Foundation Models"](https://arxiv.org/abs/2410.13826). All rationales, localized skills, and skill-slices for the 12 datasets studied in the paper can also be accessed through this repo. 

# Set up

**Quick start**: Complete the below installation, and then check out example.ipynb for some key functionality on navigating our skill-slice annotations. 

After installing the relevant packages (see requirements.txt), download and unzip the following zip file with our annotations and other useful pre-computed entities: [cached.zip](https://umd.box.com/s/5w26f4t1mbokyugufem3nsq07uqdjr5w). Click the link and then download from there. Soon, we plan to post our annotations to huggingface, to make downloading and viewing previews easier. 

Also, **be sure to update `_CACHE_ROOT` and `_DATA_ROOT` in `constants.py`**. These paths are described below:
- `_CACHED_ROOT` is where all model outputs and embeddings are cached. **This path should point to the `cached` directory downloaded and unzipped from the above link.
- `_DATA_ROOT` is where all dataset images are downloaded to. You may also want to update your environment variable `$HF_DATASETS_CACHE` so that huggingface downloads all relevant files to the same place. Note that most of the datasets we use are downloaded through huggingface.
- If you end up making plots in `analysis.py`, update `_PLOTS_ROOT` as well.

Note: our implementation of commercial models follows from the [Eureka codebase](https://github.com/microsoft/eureka-ml-insights). **Critically, the API keys are missing**, as you will need to add your own keys yourself, if you would like to use those commercial models. Be sure to update any SECRET_KEY_PARAMS in models/models.py. 

 # Citation

If you find this work insightful or its code of use, we'd appreciate if you could cite us. Here is the bibtex:

```
@misc{moayeri2024unearthingskilllevelinsightsunderstanding,
      title={Unearthing Skill-Level Insights for Understanding Trade-Offs of Foundation Models}, 
      author={Mazda Moayeri and Vidhisha Balachandran and Varun Chandrasekaran and Safoora Yousefi and Thomas Fel and Soheil Feizi and Besmira Nushi and Neel Joshi and Vibhav Vineet},
      year={2024},
      eprint={2410.13826},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.13826}, 
}
```
