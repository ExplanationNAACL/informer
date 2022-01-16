from metrics.confidence_explanation_agreement import confidence_explanation_agreement
from metrics.confidence_indication import confidence_indication
from metrics.dataset_consistency.data_consistency_v2 import Informers as InfoPlus
from ctypes import *

from salience_basic_util import SEQ_TASK, CORPUS_LEVEL, INSTANCE_LEVEL
from salience_basic_util import generate_serialization_files, display_heatmap

import random
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(".")


class Informer:
    def __init__(self, explainer, dataset, predictor, task=SEQ_TASK, num_classes=2,
                 scores_path=None, preds_path=None, ui=False):
        self._explainer = explainer
        self._dataset = dataset
        self._predictor = predictor
        self._task = task
        self._num_classes = num_classes
        self._informer = InfoPlus(dataset, predictor, explainer)
        self.scores_path = scores_path
        self.preds_path = preds_path
        if scores_path is None or preds_path is None:
            self.generate_serialization_files()
        self.ui = ui

    def __iter__(self):
        return iter(self._dataset)

    def __len__(self):
        return len(self._dataset)

    def update(self, explainer=None, dataset=None, predictor=None,
               task=None, num_classes=None,
               scores_path="./serialized_salience-scores.json",
               preds_path="./serialized_predictions-and-labels.json"):
        if explainer:
            self._explainer = explainer
            self._informer.explainer_fn = explainer
        if dataset:
            self._dataset = dataset
            self._informer.data = dataset
        if predictor:
            self._predictor = predictor
            self._informer.model_fn = predictor
        if task is not None:
            self._task = task
        if num_classes is not None:
            self._num_classes = num_classes

    def generate_serialization_files(self, scores_path=None, preds_path=None):
        if scores_path and preds_path:
            generate_serialization_files(self._task, self._num_classes, self._explainer, self._predict_fn,
                                         self._dataset, scores_path, preds_path)
        else:
            scores_path, preds_path = generate_serialization_files(self._task, self._num_classes, self._explainer,
                                                                   self._predict_fn, self._dataset)
        self.scores_path = scores_path
        self.preds_path = preds_path

    @staticmethod
    def display_explanation(scores_1, title, scores_2=None, title_2=None, ui=False, cell_labels=None, cell_labels_2=None):
        plt = display_heatmap(salience_scores=scores_1, salience_scores_2=scores_2 if scores_2 is not None else None,
                              title=title,
                              title_2=title_2 if title_2 is not None else None,
                              normalized=False,
                              cell_labels=cell_labels if cell_labels is not None else None,
                              cell_labels_2=cell_labels_2 if cell_labels_2 is not None else None
                              )
        if ui:
            filename = f"{title}.png"
            plt.savefig(os.path.join("static", f"{title}.png"))
            return filename
        else:
            plt.show()

    @staticmethod
    def _perturb_data(text):
        lib = cdll.LoadLibrary("./original")
        r = lib.perturb(text)
        print(r)
        return

    @staticmethod
    def summary_statistics(metric_list, metric_key: str, num_bins: int = 1, num_outputs: int = 1):
        return Histogram(metric_list, metric_key, num_bins, num_outputs)

    @staticmethod
    def summary_statistics_from_cea(metric_list, num_bins: int = 1, num_outputs: int = 1):
        return Informer.summary_statistics(metric_list, 'CEA score', num_bins, num_outputs)

    def confidence_explanation_agreement_slow(self, instance: bool = True, visual: bool = False):
        output = INSTANCE_LEVEL if instance else CORPUS_LEVEL
        return confidence_explanation_agreement(self._task, False, self._explainer, self._predictor, self._dataset,
                                                output=output, visual=visual)

    def confidence_explanation_agreement_point(self, sentence: str, label: str, visual: bool = False):
        output = INSTANCE_LEVEL
        dataset = [{'sentence': sentence, 'label': label}]
        ans = confidence_explanation_agreement(self._task, False, self._explainer, self._predictor, dataset,
                                               output=output, visual=visual)
        return ans[0]

    def confidence_explanation_agreement(self, instance: bool = True, visual: bool = False):
        output = INSTANCE_LEVEL if instance else CORPUS_LEVEL
        return confidence_explanation_agreement(self._task,
                                                serialized=True,
                                                output=output,
                                                scores_path=self.scores_path,
                                                preds_path=self.preds_path,
                                                visual=visual)

    def confidence_indication_slow(self, instance: bool = False, visual: bool = False, seed: int = 123):
        return confidence_indication(self._task, self._num_classes, False, self._explainer, self._predictor,
                                     self._dataset, analyze_istance_level=instance, visual_lr=visual, seed=seed)

    def confidence_indication(self, instance: bool = False, visual: bool = False, seed: int = 123):
        return confidence_indication(self._task, self._num_classes, True, scores_path=self.scores_path,
                                     preds_path=self.preds_path, analyze_instance_level=instance, visual_lr=visual,
                                     seed=seed)

    def activation_similarity(self, model, layers, inst_x, inst_y):
        return self._informer.activation_similarity(model, layers, inst_x, inst_y)

    def explanation_similarity(self, inst_x, inst_y):
        return self._informer.explanation_similarity(inst_x, inst_y)

    def data_consistency(self, layers, thresh=2000, model=None, re_format=None):
        return self._informer.data_consistency(layers, self.scores_path, self.preds_path, self._num_classesthresh, model, re_format)

    def faithfulness(self):
        pass


class Histogram:
    def __init__(self, metric_list, metric_key: str, num_bins: int, num_outputs: int):
        self._sorted_list = sorted(metric_list, key=lambda x: x[metric_key])
        self._input_keys = [x for x in metric_list[0] if x != metric_key]
        self._metric_key = metric_key
        self._num_bins = num_bins
        self._num_outputs = num_outputs
        self.ranges = []
        self.set_ranges(num_bins, num_outputs)

    def set_ranges(self, num_bins: int, num_outputs: int):
        self.ranges = []
        len_list = len(self._sorted_list)
        real_num_outputs = min(num_outputs, len_list // num_bins)
        self._num_bins, self._num_outputs = num_bins, real_num_outputs
        for i in range(num_bins):
            min_index = len_list * i // num_bins
            max_index = len_list * (i + 1) // num_bins
            sample = random.sample(self._sorted_list[min_index:max_index], real_num_outputs)
            range_i = Range(self._sorted_list[min_index][self._metric_key],
                            self._sorted_list[max_index - 1][self._metric_key])

            for sent in sample:
                x = sent.copy()
                x.pop(self._metric_key, None)
                range_i.add(x)

            self.ranges.append(range_i)

    def chart(self, num_bins: int, low=None, high=None, title: str = "", ui: bool = False):
        low_val = low if low else self._sorted_list[0][self._metric_key]
        high_val = high if high else self._sorted_list[-1][self._metric_key]
        dist = high_val - low_val
        sep = dist / num_bins

        counter = {}
        keys = []
        i = low_val
        while i < high_val + sep:
            counter[i] = 0
            keys.append(i)
            i += sep
        keys.append(high_val + sep)

        curr, j, k = 0, keys[0], keys[1]
        for i in self._sorted_list:
            while i[self._metric_key] > j:
                curr += 1
                j, k = keys[curr], keys[curr + 1]

            counter[j] += 1

        plt.bar(x=counter.keys(), height=counter.values())
        plt.xticks(np.arange(low_val, high_val + sep, sep))
        plt.title(title)
        if ui:
            filename = f"Hist_{title}.png"
            plt.savefig(os.path.join("static", filename))
            return counter, filename
        else:
            plt.show()
            return counter, None

    def __repr__(self):
        return f'<Histogram {num_bins} {num_outputs}>'

    def __str__(self):
        return "\n\n".join([str(r) for r in self.ranges])

    def __len__(self):
        return self._num_bins


class Range:
    def __init__(self, low_val, high_val):
        self.low_val = low_val
        self.high_val = high_val
        self.sents = []

    def add(self, sent):
        self.sents.append(sent)

    def __repr__(self):
        return f'<Range {len(self.sents)}>'

    def __str__(self):
        range_str = f'{self.low_val}-{self.high_val}:\n\t'
        sent_str = "\n\t".join(["; ".join([f'{key}: {sent[key]}' for key in sent]) for sent in self.sents])
        return range_str + sent_str

    def __len__(self):
        return len(self.sents)
