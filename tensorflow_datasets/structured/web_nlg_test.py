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

"""web_nlg dataset."""

import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.structured import web_nlg


class WebNlgTest(tfds.testing.DatasetBuilderTestCase):
  DATASET_CLASS = web_nlg.WebNlg
  SPLITS = {
      "train": 12,
      "validation": 4,
      "test_all": 5,
      "test_unseen": 2,
  }


if __name__ == "__main__":
  tfds.testing.test_main()
