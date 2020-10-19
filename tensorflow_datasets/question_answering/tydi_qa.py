# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TyDi QA: Information-Seeking QA in Typologically Diverse Languages."""

# TODO(adarob): Add primary tasks (SelectP and MinSpan).

import os

import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.question_answering import qa_utils

_CITATION = """\
@article{tydiqa,
   title = {TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages},
  author = {Jonathan H. Clark and Eunsol Choi and Michael Collins and Dan Garrette and Tom Kwiatkowski and Vitaly Nikolaev and Jennimaria Palomaki}
    year = {2020},
 journal = {Transactions of the Association for Computational Linguistics}
}
"""

_DESCRIPTION = """\
TyDi QA is a question answering dataset covering 11 typologically diverse \
languages with 204K question-answer pairs. The languages of TyDi QA are \
diverse with regard to their typology -- the set of linguistic features that \
each language expresses -- such that we expect models performing well on this \
set to generalize across a large number of the languages in the world. It \
contains language phenomena that would not be found in English-only corpora. \
To provide a realistic information-seeking task and avoid priming effects, \
questions are written by people who want to know the answer, but don’t know \
the answer yet, (unlike SQuAD and its descendents) and the data is collected \
directly in each language without the use of translation (unlike MLQA and \
XQuAD).

For now, only the Gold passage (GoldP) task is available in TFDS.
The "translate-train-*" splits consist of machine translations of the English data.
This data was used for the translate-train baselines in the XTREME paper [https://arxiv.org/abs/2003.11080]
"""


LANGUAGES = {
    "ar": "arabic",
    "bn": "bengali",
    "en": "english",
    "fi": "finnish",
    "id": "indonesian",
    "ko": "korean",
    "ru": "russian",
    "sw": "swahili",
    "te": "telugu",
}

_GOLD_URL_PREFIX = "https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-"
_GOLD_TRANSLATE_URL_FORMAT = "https://storage.googleapis.com/xtreme_translations/TyDiQA-GoldP/translate-train/tydiqa.translate.train.en-{lang_iso}.json"


class TydiQAConfig(tfds.core.BuilderConfig):
  """BuilderConfig for TydiQa."""


class TydiQA(tfds.core.GeneratorBasedBuilder):
  """TyDi QA: Information-Seeking QA in Typologically Diverse Languages."""

  BUILDER_CONFIGS = [
      TydiQAConfig(
          name="goldp",
          description="Gold passage (GoldP) task (https://github.com/google-research-datasets/tydiqa/tree/master/gold_passage_baseline).",
          version=tfds.core.Version("2.0.0"),
      ),
  ]

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=qa_utils.SQUADLIKE_FEATURES,
        # No default supervised_keys (as we have to pass both question
        # and context as input).
        supervised_keys=None,
        homepage="https://github.com/google-research-datasets/tydiqa",
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    urls_to_download = {
        "train": _GOLD_URL_PREFIX + "train.json",
        "validation": _GOLD_URL_PREFIX + "dev.json",
        "lang-validation": _GOLD_URL_PREFIX + "dev.tgz",
    }
    for lang_iso in LANGUAGES:
      if lang_iso=='en':
        continue
      urls_to_download["translate-train-{lang_iso}".format(
          lang_iso=lang_iso)] = _GOLD_TRANSLATE_URL_FORMAT.format(
              lang_iso=lang_iso)
    downloaded_files = dl_manager.download_and_extract(urls_to_download)

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={"filepath": downloaded_files["train"]}),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={"filepath": downloaded_files["validation"]}),
    ] + [
        tfds.core.SplitGenerator(  # pylint:disable=g-complex-comprehension
            name=f"validation-{lang_iso}",
            gen_kwargs={
                "filepath":
                os.path.join(downloaded_files["lang-validation"],
                             f"tydiqa-goldp-v1.1-dev/tydiqa-goldp-dev-{lang_name}.json")
            })
        for lang_iso, lang_name in LANGUAGES.items()
    ] + [
        tfds.core.SplitGenerator(  # pylint:disable=g-complex-comprehension
            name="translate-train-{lang_iso}".format(
          lang_iso=lang_iso),
            gen_kwargs={
                "filepath": downloaded_files["translate-train-{lang_iso}".format(lang_iso=lang_iso)]
                })
        for lang_iso, lang_name in LANGUAGES.items() if lang_iso!="en"
    ]

  def _generate_examples(self, filepath):
    return qa_utils.generate_squadlike_examples(filepath)
