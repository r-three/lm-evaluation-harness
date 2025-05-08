# Tiny XQuAD
This is a lighter version of the XQUAD dataset, where each (subset, split) pair includes a randomly selected 10% of the original data. To maintain sufficient sample sizes, if this 10% subset contains fewer than 100 examples, it is expanded to include at least 100 samples. In cases where the total number of available samples is less than 100, all samples are retained. 

### Paper

Title: `On the Cross-lingual Transferability of Monolingual Representations`

Abstract: https://aclanthology.org/2020.acl-main.421.pdf

XQuAD (Cross-lingual Question Answering Dataset) is a benchmark dataset for evaluating cross-lingual question answering performance. The dataset consists of a subset of 240 paragraphs and 1190 question-answer pairs from the development set of SQuAD v1.1 (Rajpurkar et al., 2016) together with their professional translations into ten languages: Spanish, German, Greek, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, and Hindi. Consequently, the dataset is entirely parallel across 11 languages.

Homepage: https://github.com/deepmind/xquad


### Citation

```
@article{Artetxe:etal:2019,
      author    = {Mikel Artetxe and Sebastian Ruder and Dani Yogatama},
      title     = {On the cross-lingual transferability of monolingual representations},
      journal   = {CoRR},
      volume    = {abs/1910.11856},
      year      = {2019},
      archivePrefix = {arXiv},
      eprint    = {1910.11856}
}
```

### Groups and Tasks

#### Groups

* `tiny_xquad`: All available languages.

#### Tasks
Perform extractive question answering for each language's subset of XQuAD.
* `tiny_xquad_ar`: Arabic
* `tiny_xquad_de`: German
* `tiny_xquad_el`: Greek
* `tiny_xquad_en`: English
* `tiny_xquad_es`: Spanish
* `tiny_xquad_hi`: Hindi
* `tiny_xquad_ro`: Romanian
* `tiny_xquad_ru`: Russian
* `tiny_xquad_th`: Thai
* `tiny_xquad_tr`: Turkish
* `tiny_xquad_vi`: Vietnamese
* `tiny_xquad_zh`: Chinese



### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
