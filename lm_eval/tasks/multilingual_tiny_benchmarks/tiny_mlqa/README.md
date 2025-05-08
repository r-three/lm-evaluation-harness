# Tiny MLQA
This is a lighter version of the MLQA dataset, where each (subset, split) pair includes a randomly selected 10% of the original data. To maintain sufficient sample sizes, if this 10% subset contains fewer than 100 examples, it is expanded to include at least 100 samples. In cases where the total number of available samples is less than 100, all samples are retained. 

### Paper

Title: `MLQA: Evaluating Cross-lingual Extractive Question Answering`

Abstract: `https://arxiv.org/abs/1910.07475`

MLQA (MultiLingual Question Answering) is a benchmark dataset for evaluating cross-lingual question answering performance.
MLQA consists of over 5K extractive QA instances (12K in English) in SQuAD format in seven languages - English, Arabic,
German, Spanish, Hindi, Vietnamese and Simplified Chinese. MLQA is highly parallel, with QA instances parallel between
4 different languages on average

Homepage: `https://github.com/facebookresearch/MLQA`


### Citation

```
@misc{lewis2020mlqaevaluatingcrosslingualextractive,
      title={MLQA: Evaluating Cross-lingual Extractive Question Answering},
      author={Patrick Lewis and Barlas Oğuz and Ruty Rinott and Sebastian Riedel and Holger Schwenk},
      year={2020},
      eprint={1910.07475},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1910.07475},
}
```

### Groups, Tags, and Tasks

#### Groups

* Not part of a group yet

#### Tasks

Tasks of the form `tiny_mlqa_context-lang_question-lang.yaml`
* `tiny_mlqa_ar_ar.yaml`
* `tiny_mlqa_ar_de.yaml`
* `tiny_mlqa_ar_vi.yaml`
* `tiny_mlqa_ar_zh.yaml`
* `tiny_mlqa_ar_en.yaml`
* `tiny_mlqa_ar_es.yaml`
* `tiny_mlqa_ar_hi.yaml`
* `tiny_mlqa_de_ar.yaml`
* `tiny_mlqa_de_de.yaml`
* `tiny_mlqa_de_vi.yaml`
* `tiny_mlqa_de_zh.yaml`
* `tiny_mlqa_de_en.yaml`
* `tiny_mlqa_de_es.yaml`
* `tiny_mlqa_de_hi.yaml`
* `tiny_mlqa_vi_ar.yaml`
* `tiny_mlqa_vi_de.yaml`
* `tiny_mlqa_vi_vi.yaml`
* `tiny_mlqa_vi_zh.yaml`
* `tiny_mlqa_vi_en.yaml`
* `tiny_mlqa_vi_es.yaml`
* `tiny_mlqa_vi_hi.yaml`
* `tiny_mlqa_zh_ar.yaml`
* `tiny_mlqa_zh_de.yaml`
* `tiny_mlqa_zh_vi.yaml`
* `tiny_mlqa_zh_zh.yaml`
* `tiny_mlqa_zh_en.yaml`
* `tiny_mlqa_zh_es.yaml`
* `tiny_mlqa_zh_hi.yaml`
* `tiny_mlqa_en_ar.yaml`
* `tiny_mlqa_en_de.yaml`
* `tiny_mlqa_en_vi.yaml`
* `tiny_mlqa_en_zh.yaml`
* `tiny_mlqa_en_en.yaml`
* `tiny_mlqa_en_es.yaml`
* `tiny_mlqa_en_hi.yaml`
* `tiny_mlqa_es_ar.yaml`
* `tiny_mlqa_es_de.yaml`
* `tiny_mlqa_es_vi.yaml`
* `tiny_mlqa_es_zh.yaml`
* `tiny_mlqa_es_en.yaml`
* `tiny_mlqa_es_es.yaml`
* `tiny_mlqa_es_hi.yaml`
* `tiny_mlqa_hi_ar.yaml`
* `tiny_mlqa_hi_de.yaml`
* `tiny_mlqa_hi_vi.yaml`
* `tiny_mlqa_hi_zh.yaml`
* `tiny_mlqa_hi_en.yaml`
* `tiny_mlqa_hi_es.yaml`
* `tiny_mlqa_hi_hi.yaml`

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
