"""
Utility functions for the Informers Class, and some demo usages of the LIT library.
TODO: Remove LIT demo usage later.

To do the LIT demos:
- Load the weights of whatever pretrained ML model you will be using. Example, the BERT-based model
  provided by LIT for binary sentiment classification can be downloaded using the command line:
    - wget https://storage.googleapis.com/what-if-tool-resources/lit-models/sst2_tiny.tar.gz;
      tar -xvf sst2_tiny.tar.gz;
    - change this line to be wherever those weights were extracted to:
      line 31 --> path_to_pretrained_model = "__"

Trained models we have access to:
TODO: Remove later for final product. We have predictions and explanations, for certain datasets serialized, for demo.
- https://storage.googleapis.com/what-if-tool-resources/lit-models/sst2_tiny.tar.gz (trained on SST2 dataset)
- https://storage.googleapis.com/what-if-tool-resources/lit-models/sst2_small.tar.gz (trained on SST2 dataset)
- https://storage.googleapis.com/what-if-tool-resources/lit-models/mbert_mnli.tar.gz (trained on MNLI dataset)
"""

from lit_nlp.components import lime_explainer
from lit_nlp.examples.datasets import glue
from lit_nlp.examples.models import glue_models
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import json

from typing import Callable, List, Dict, Union, Tuple


path_to_pretrained_model = 'demo/models/sst2-tiny-bert'

# These are library-wide used constants that can be imported in other files
SEQ_TASK = "seq"
NLI_TASK = "nli"
CORPUS_LEVEL = "corpus"
INSTANCE_LEVEL = "instance"
JSON_HEADER_SCORES = "Saliency Scores for all Classes, per Instance"


def generate_serialization_files(task: str,
                                 num_classes: int,
                                 explainer: Union[Callable[[Dict[str, str]], List[Dict[str, float]]],
                                                  Callable[[Dict[str, str]], List[Dict[str, Dict[str, float]]]]],
                                 predict_fn: Callable[[Dict[str, str]], np.ndarray],
                                 dataset: List[Dict[str, str]],
                                 scores_path: str = "./serialized_salience-scores.json",
                                 preds_path: str = "./serialized_predictions-and-labels.json") -> Tuple[str, str]:
    """
    Takes in an explainer, model predict function, and dataset (all adhering to the Informer Specifications)
    and serializes explanation saliency output and model prediction and (optional) gold labels). Serialized data
    will be found in the specified, or default, pathnames to the JSON files. Saliency scores will be found
    in one file, and model predictions are found in another file

    Input:
        - task: str. We currently only support "seq" or "nli".
        - num_classes: int. Number of classes in the task
        - explainer, predict_fn, dataset: See Informers documentation, available in the Informers GitHub.
        - scores_path & preds_path: str. Pathname for the files the serialized info will be saved to.
          Default is as shown above.

    Output:
        - Returns the file path-names with the serialized saliency scores and model prediction output
    """
    # Checking input formatting
    assert task == SEQ_TASK or task == NLI_TASK  # currently only n-class sequence classification and NLI task supported

    if task == NLI_TASK:
        assert num_classes == 3

    assert len(dataset) > 0  # need at least one datapoint to serialize.

    #if task == NLI_TASK:
    #    assert type(explainer) == Callable[[Dict[str, str]], List[Dict[str, Dict[str, float]]]]
    #elif task == SEQ_TASK:
    #    assert type(explainer) == Callable[[Dict[str, str]], List[Dict[str, float]]]

    json_object_saliencies = {JSON_HEADER_SCORES: []}
    json_object_preds = {"label": [], "logits": []}

    # iterate through dataset and get explanations and model prediction for each instance
    for index, data in enumerate(dataset):
        serialized_instance = None
        if task == SEQ_TASK:
            assert len(list(dataset[index].keys())) == 2 and \
                   'sentence' in list(dataset[index].keys()) and \
                   'label' in list(dataset[index].keys())
        elif task == NLI_TASK:
            assert len(list(dataset[index].keys())) == 3 and \
                   'premise' in list(dataset[index].keys()) and \
                   'hypothesis' in list(dataset[index].keys()) and \
                   'label' in list(dataset[index].keys())

        # run the explainer and model prediction function
        # this will take the most time
        # TODO: Optimize to run batch explanations if necessary
        explanations = explainer(data)
        prediction = predict_fn(data)

        # I think this assertion holds for NLI explainer output as well
        assert len(explanations) == num_classes
        assert prediction.shape == (num_classes,)

        # Serialize the predictions and gold label in the standard format
        # Works for both sequence classification and nli datasets
        json_object_preds['label'].append(data['label'])
        json_object_preds['logits'].append(list(map(lambda x: x.item(), list(prediction))))

        # Serialize the explanations in the standard format
        # Different for seq classification and nli datasets since explainer output is different
        if task == SEQ_TASK:
            # Explainer outputs:    [{"a", 123, "b": 45}, {"a", -123, "b": -45}]
            # We want:              [{"token": "a", "0": 123, "1", -123}, {"token": "b", "0": 45, "1", -45}]
            serialized_instance = []
            tokens = []
            salience_scores_cls = []
            for cls, exp in enumerate(explanations):
                if cls == 0:
                    tokens = list(exp.keys())
                else:
                    # the explanation for each class should be for the same tokens
                    assert len(list(exp.keys())) == len(tokens)
                salience_scores_cls.append(list(exp.values()))

            for index_t, token_text in enumerate(tokens):
                token = {"token": token_text}
                for cls in range(num_classes):
                    token[str(cls)] = salience_scores_cls[cls][index_t]
                serialized_instance.append(token)
        elif task == NLI_TASK:
            # Explainer outputs:    [{"premise": {'a': 1, 'b': 2}, "hypothesis": {"c": 45}},
            #                        {"premise": {'a': 9, 'b': 3}, "hypothesis": {"c": 5}},
            #                        {"premise": {'a': 7, 'b': 6}, "hypothesis": {"c": 6}}]
            # We want:              [{"premise": [{"token": 'a', "0": 1, "1": 9, "2": 7},
            #                                     {"token": 'b', "0": 2, "1": 3, "2": 6}]
            #                         "hypothesis": [{"token": 'c', "0": 45, "1": 5, "2": 6}]
            #                         }]
            serialized_instance = {"premise": [], "hypothesis": []}
            tokens_p = []
            tokens_h = []
            salience_scores_cls_p = []
            salience_scores_cls_h = []
            for cls, exp in enumerate(explanations):
                if cls == 0:
                    tokens_p = list(exp['premise'].keys())
                    tokens_h = list(exp['hypothesis'].keys())
                else:
                    # the explanation for each class should be for the same tokens
                    assert len(list(exp['premise'].keys())) == len(tokens_p)
                    assert len(list(exp['hypothesis'].keys())) == len(tokens_h)
                salience_scores_cls_p.append(list(exp['premise'].values()))
                salience_scores_cls_h.append(list(exp['hypothesis'].values()))

            # premise
            for index_t, token_text in enumerate(tokens_p):
                token = {"token": token_text}
                for cls in range(num_classes):
                    token[str(cls)] = salience_scores_cls_p[cls][index_t]
                serialized_instance["premise"].append(token)

            # hypothesis
            for index_t, token_text in enumerate(tokens_h):
                token = {"token": token_text}
                for cls in range(num_classes):
                    token[str(cls)] = salience_scores_cls_h[cls][index_t]
                serialized_instance["hypothesis"].append(token)

        json_object_saliencies[JSON_HEADER_SCORES].append([serialized_instance])

    # write JSON files
    with open(scores_path, 'w') as outfile:
        json.dump(json_object_saliencies, outfile, indent=1)
    with open(preds_path, 'w') as outfile:
        json.dump(json_object_preds, outfile, indent=1)

    return scores_path, preds_path


def display_heatmap(salience_scores,
                    salience_scores_2=None,
                    title=None,
                    title_2=None,
                    cell_labels=None,
                    cell_labels_2=None,
                    normalized=True,
                    ui=False):
    """
    A utility function that displays a Seaborn heatmap.

    Input:
        - ('salience scores') A list of floats .
          If task is something like NLI, then these are the salience scores for the premise, or first
          sequence.

        - ('salience_scores_2') A list of floats .
          Optional. Only necessary when task is a relation labeling task between 2 sequences
          like NLI. Then these are the salience scores for the hypothesis, or second sequence.

        - ('title') Any object (string, integer, float, etc.) that can be printed.
          Optional.
          Usually is descriptive blurb for the heatmap for ('salience_scores')

        - ('title_2') Any object (string, integer, float, etc.) that can be printed.
          Optional. Usually is descriptive blurb for the heatmap for ('salience scores_2')

        - ('cell_labels') Optional. list of the same size as ('salience_scores') that is printed
          on the corresponding cell. Usually something like salience score values.

        - ('cell_labels_2') Optional. list of the same size as ('salience_scores_2') that is printed
          on the corresponding cell. Usually something like salience score values.

        - ('normalized') A boolean denoting whether the data is normalized or not. If normalized,
          the range is from -1 to 1.

        - ('ui') A boolean for option of saving the plot instead to a file and returning the filename

    Output:
        - Return the matplotlib object
    """
    if cell_labels is not None:
        assert len(cell_labels) == len(salience_scores)
    if cell_labels_2 is not None:
        assert len(cell_labels_2) == len(salience_scores_2)

    cmap = sns.diverging_palette(10, 240, as_cmap=True)

    if salience_scores_2 is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        ax1.set_title(title if title is not None else "")
        ax2.set_title(title_2 if title_2 is not None else "")
        sns.heatmap([salience_scores],
                    ax=ax1,
                    annot=[cell_labels] if cell_labels is not None else False,
                    fmt='',
                    cmap=cmap,
                    linewidths=0.5,
                    square=True,
                    center=0,
                    vmin=-1 if normalized else None,
                    vmax=1 if normalized else None)

        sns.heatmap([salience_scores_2],
                    ax=ax2,
                    annot=[cell_labels_2] if cell_labels_2 is not None else False,
                    fmt='',
                    cmap=cmap,
                    linewidths=0.5,
                    square=True,
                    center=0,
                    vmin=-1 if normalized else None,
                    vmax=1 if normalized else None)
    else:
        m = sns.heatmap([salience_scores],
                        annot=[cell_labels] if cell_labels is not None else False,
                        fmt='',
                        linewidths=0.5,
                        square=True,
                        cmap=cmap,
                        center=0,
                        vmin=-1 if normalized else None,
                        vmax=1 if normalized else None)
        plt.title(title if title is not None else "")

    #plt.show()
    return plt


def normalize_scores(saliency_scores: np.ndarray):
    """
    A utility function that normalizes token importance, or salience, scores in
    an explanation. The transformation is such that the relative rank of the scores
    across the explanation are preserved, but the absolute value of the scores would add up to 1.

    Input:
        - saliency_scores: A Numpy n-dimensional array of shape (n,)

    Output:
        - A Numpy n-dimensional array of shape (n,)
    """
    return saliency_scores / np.abs(saliency_scores).sum(-1)


def perturb(sent_dict):
    """
    Given a dictionary of the format {'sentence': {whitespace-separated string}, 'label': {integer label}}
    Return an identical dictionary but with 15% of the tokens replaced with '[MASK]'
    """
    sentence_mod_array = sent_dict['sentence'].split()
    for i, word in enumerate(sentence_mod_array):
        if random.random() < 0.15:
            sentence_mod_array[i] = "[MASK]"
    sentence_mod = " ".join(sentence_mod_array)
    sent_dict_mod = {'sentence': sentence_mod, 'label': sent_dict['label']}
    return sent_dict_mod

def kl_divergence(p, q):
    # Use softmax to normalize the two distributions for proper comparison
    p_norm = np.abs(p)
    q_norm = np.abs(q)
    return np.sum([x * np.log2(x / y) for x, y in zip(p_norm, q_norm)])

def jsd(p, q):
    m = np.add(np.abs(p), np.abs(q))/2
    return kl_divergence(p, m)/2 + kl_divergence(q, m)/2

def salience_score_analysis(model, dataset):
    """
    Print out some basic stats for binary sentiment classification:
    - Model predicted class
    - Prediction score/probability
    - LIME salience scores
    - Modified version of the input
    - Salience score on the modified input
    - Jensen-Shannon Divergence between the two (normalized) salience maps
    """
    lime = lime_explainer.LIME()
    num_examples = 10 
    dataset_mods = [perturb(sent) for sent in dataset.examples[:num_examples]]  # Introduce perturbations to data
    lime_outputs = lime.run(dataset.examples[:num_examples], model, dataset)
    lime_outputs_mod = lime.run(dataset_mods, model, dataset)   # Run lime on modified dataset

    for index, lime_output in enumerate(lime_outputs):
        datapoint = dataset.examples[index]
        datapoint_mod = dataset_mods[index]
        model_prediction = np.array([o['probas'] for o in model.predict([datapoint])])
        pred = np.argmax(model_prediction[0])
        prob = np.max(model_prediction[0])
        salience_scores = lime_output['sentence'].salience
        salience_scores_mod = lime_outputs_mod[index]['sentence'].salience
        #display_heatmap(salience_scores, positive=True)
        #display_heatmap(salience_scores_mod, positive=True)
        comparison = jsd(salience_scores, salience_scores_mod) * 100    # Get the Jensen-Shannon Divergence score
        print(f"Sentence: {datapoint['sentence']}\n>>>model prediction class = {pred} with "
              f"prob: {prob}\n"
              f">>>salience scores: {salience_scores}\n"
              f"Modified sentence: {datapoint_mod['sentence']}\n"       # Write out the modified sentence
              f">>>modified salience scores: {salience_scores_mod}\n"   # Modified sentence salience scores
              f">>>Difference between scores: {comparison:.2f}\n")      # Comparison between the two


def histogram(model, dataset):
    lime = lime_explainer.LIME()
    num_examples = 100
    num_bins = 10
    num_outputs = 5
    dataset_mods = [perturb(sent) for sent in dataset.examples[:num_examples]]
    lime_outputs = lime.run(dataset.examples[:num_examples], model, dataset)
    lime_outputs_mod = lime.run(dataset_mods, model, dataset)

    matches = []
    changes = []
    for index, lime_output in enumerate(lime_outputs):
        datapoint = dataset.examples[index]
        datapoint_mod = dataset_mods[index]

        model_prediction = np.array([o['probas'] for o in model.predict([datapoint])])
        pred = np.argmax(model_prediction[0])
        model_prediction_mod = np.array([o['probas'] for o in model.predict([datapoint_mod])])
        pred_mod = np.argmax(model_prediction_mod[0])

        salience_scores = lime_output['sentence'].salience
        salience_scores_mod = lime_outputs_mod[index]['sentence'].salience
        comparison = jsd(salience_scores, salience_scores_mod) * 100    # Get the Jensen-Shannon Divergence score

        datapoint_dict = {'sentence': datapoint['sentence'], 'sentence_mod': datapoint_mod['sentence'],
                'scores': salience_scores, 'scores_mod': salience_scores_mod, 'comparison': comparison}
        
        if pred == pred_mod:
            matches.append(datapoint_dict)
        else:
            changes.append(datapoint_dict)

    sorted_matches = sorted(matches, key=lambda x: x['comparison'])
    real_num_outputs = min(num_outputs, len(sorted_matches) // num_bins)
    for i in range(num_bins):
        min_index = len(sorted_matches) * i // num_bins
        max_index = len(sorted_matches) * (i+1) // num_bins
        sample = random.sample(sorted_matches[min_index:max_index], real_num_outputs)
        print(f"\nJSD RANGE: {sorted_matches[min_index]['comparison']:.2f} - "
              f"{sorted_matches[max_index-1]['comparison']:.2f}")
        for s in sample:
            print(s['sentence'])
            print(s['sentence_mod'])
            print()

    real_num_outputs_changes = min(len(changes), num_outputs)
    change_sample = random.sample(changes, real_num_outputs_changes)
    print("\nCHANGED OUTPUTS")
    for s in change_sample:
        print(s['sentence'])
        print(s['sentence_mod'])
        print()


def main():
    # See https://github.com/PAIR-code/lit/blob/main/lit_nlp/examples/models/glue_models.py for demo model api
    #    you can swap out models and datasets using the model api.

    # There are also a bunch of demo datasets and models provided. We will be using the validation set of the
    #    Stanford Sentiment Treebank (binary) dataset provided as a demo.
    dataset = glue.SST2Data('validation')

    # model is downloaded in the current file (sst_tiny.tar.gz)
    model = glue_models.SST2Model(path_to_pretrained_model)
    salience_score_analysis(model, dataset)
    # histogram(model, dataset)


if __name__ == "__main__":
    main()
