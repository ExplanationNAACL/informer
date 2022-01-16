import json
import math

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from typing import List, Callable, Dict
import random
from salience_basic_util import SEQ_TASK, NLI_TASK, JSON_HEADER_SCORES
from perturbations.synonym_perturbations import perturb as get_synonym_perturbation

nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()

MU = "\u03BC"
SIGMA = "\u03C3"


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def display_score_heatmap(scores, labels: List[str], title: str):
    """
    Displays a seaborn heatmap of salience scores.
    """
    cmap = sns.diverging_palette(10, 240, as_cmap=True)
    sns.heatmap([scores],
                linewidths=0.8,
                cmap=cmap,
                annot=[labels],
                annot_kws={'rotation': 90},
                fmt="",
                vmin=-1,
                vmax=1)
    plt.title(f"{title}")
    plt.show()


def display_std_region_histogram(regions: List,
                                 title: str):
    """
    Displays histogram of frequency of salience scores for a focus token in standard deviations away from mean
    and whether they are negative or positive scores.
    """
    regions_counts_neg_scores = dict(Counter(region[0] for region in regions if region[3]))
    regions_counts_pos_scores = dict(Counter(region[0] for region in regions if region[3] is False))
    region_mapping = {1: "medium-high", 2: "high", 3: "very high", -3: "very low", -2: "low", -1: "medium-low"}

    # tick_labels = ["", "f{MU}-2{SIGMA}", f"{MU}-{SIGMA}", f"{MU}+{SIGMA}", f"{MU}+2{SIGMA}", ""]
    tick_labels = ["", "low", "medium-low", "medium-high", "high", ""]
    regions = [-3, -2, -1, 1, 2, 3]
    for i in regions:
        if i not in regions_counts_neg_scores.keys():
            regions_counts_neg_scores[i] = 0
        if i not in regions_counts_pos_scores.keys():
            regions_counts_pos_scores[i] = 0
    regions_counts_neg_scores = dict(sorted(regions_counts_neg_scores.items(), key=lambda x: x[0]))
    regions_counts_pos_scores = dict(sorted(regions_counts_pos_scores.items(), key=lambda x: x[0]))

    width = 0.25
    legend = ['Negative scores', 'Positive scores']
    plt.bar(x=np.arange(len(regions_counts_neg_scores)) + width, width=width, height=regions_counts_neg_scores.values(),
            color='r', align='center')
    plt.bar(x=np.arange(len(regions_counts_pos_scores)) + 5 * width / 2, width=width,
            height=regions_counts_pos_scores.values(), color='b', align='center')
    plt.xticks(np.arange(len(tick_labels)), tick_labels)
    plt.legend(legend, loc=1)
    plt.title(title)
    plt.show()


def get_std_region(focus_token_lemma: str, sentence_lemmas: List[str], sentence: List[str], scores: np.ndarray):
    """
    Takes a look at each lemma in the input sentence and finds the focus token lemma, and
    calculates the region in saliency scores distribution that the focus token lemma lies in,
    and produces the identifying information for it.

    Input:
        - focus_token_lemma: str. The lemma of the focus token.
        - sentence_lemmas: List[str]. The lemmas of the tokens in the current sequence
        - sentence: List[str]. The original tokens in the current sequence.
        - scores: np.ndarray. The saliency scores for each token in the sequence
    """
    std_regions = []
    scores_dist = np.abs(scores)

    # regions
    stat_measures = stats.describe(scores_dist)
    standard_deviation = math.sqrt(stat_measures.variance)
    mean = stat_measures.mean

    std_1_right = mean + standard_deviation
    std_2_right = mean + 2 * standard_deviation
    std_1_left = mean - standard_deviation
    std_2_left = mean - 2 * standard_deviation

    for i, token_lemma in enumerate(sentence_lemmas):
        if focus_token_lemma == token_lemma:
            score = scores[i]
            is_negative = True if score < 0 else False
            if mean <= abs(score) < std_1_right:
                region = 1
                std_regions.append((region, sentence, scores, is_negative, i))
            elif std_1_right <= abs(score) < std_2_right:
                region = 2
                std_regions.append((region, sentence, scores, is_negative, i))
            elif std_2_right <= abs(score):
                region = 3
                std_regions.append((region, sentence, scores, is_negative, i))
            elif std_1_left <= abs(score) < mean:
                region = -1
                std_regions.append((region, sentence, scores, is_negative, i))
            elif std_2_left <= abs(score) < std_1_left:
                region = -2
                std_regions.append((region, sentence, scores, is_negative, i))
            elif abs(score) < std_2_left:
                region = -3
                std_regions.append((region, sentence, scores, is_negative, i))
    return std_regions


def get_more_data(focus_cls, predict_fn, explainer, additional: int, instance_sentence: Dict[str, str], focus_token: str, tokenizer: Callable[[str], List[str]]):
    # synonym time
    variations = get_synonym_perturbation(sentence=instance_sentence['sentence'].lower(),
                                          depth=3,
                                          banned_words=[focus_token.lower()],
                                          tokenizer=tokenizer)
    pick = []
    try:
        pick = variations[:additional]
    except:
        variations = get_synonym_perturbation(sentence=instance_sentence['sentence'].lower(),
                                              depth=7,
                                              banned_words=[focus_token.lower()],
                                              tokenizer=tokenizer)
        try:
            pick = variations[:additional]
        except:
            print("Unable to gather a sufficient amount of data to run token consistency")
            return None

    pick = [{'sentence': x, 'label': None} for x in pick
            if np.argmax(predict_fn({'sentence': x, 'label': None})) == focus_cls]

    if len(variations) < additional:  # TODO: Work out this mechanism
        print("Please manually pick data from synonym perturbations with your focus token an add that to your dataset")
        print("You can do this by making a new dataset wrpper with the additional\n"
              "data and reinitialize the Informers class and make new path names for your new data")
        return None

    for instance in pick:
        # TODO: currently here
        scores = explainer(instance)


def token_consistency(task: str,
                      focus_token: str,
                      data: Dict[str, str],
                      scores_path: str,
                      pred_path: str,
                      dataset: List[Dict[str, str]],
                      predict_fn: Callable[[Dict[str, str]], np.ndarray],
                      explanation_method: Callable[[Dict[str, str]], List[Dict[str, float]]],
                      tokenizer: Callable[[str], List[str]],
                      exact_match: bool = False,  # this is if you don't want lemmatization (just lower case),
                      visual: bool = False
                      ):
    """
    TODO: Currently only supports SEQ classification input data format
    Places salience scores by ranges using standard deviations from the mean, and locates
    range of focus token. Then, the range placement of focus token in other
    data in dataset is looked at.

    If there isn't much data with the focus token, we generate more data from the input data by
    making variations with adjuncts removed, and non-focus tokens replaced with synonyms.
    However, perturbed variations are prone to ungrammaticality, and so the user
    might need to manually filter the variations. We will provide a mechanism for this.

    Input:
    -   ('focus token') A string representing the token of interest.
        The focus token must follow the tokenization of the explanation method.
        So when the input sentence ('data') is tokenized by tokenizer, it will contain the focus token.
        Eg. A BERT BPE tokenizer might tokenize "The hilariously funny film" to
                ["the", "hilarious", "##ly", "fun", "##ny", "film"],
            or if your preprocessing pipeline splits on whitespace, it would tokenize to
                ["the", "hilariously", "funny", "film"].

    -   ('data') A single datapoint in the form of {'sentence': 'something is here', 'label': '0'}
    """
    # INPUT FORMATTING CHECKS
    # check if necessary inputs are provided
    assert scores_path is not None and pred_path is not None
    assert task == SEQ_TASK

    # check if data is in correct format
    assert 'sentence' in data.keys() and 'label' in data.keys()

    # check if focus token is normalized according to model input and will be found in explanation
    assert focus_token.lower() in list(map(lambda x: x.lower(), tokenizer(data['sentence'])))

    # open serialized explanations and model logits
    with open(scores_path, 'rb') as f:
        sals = json.load(f)
        instances = sals[JSON_HEADER_SCORES]
    with open(pred_path, 'rb') as f:
        preds = json.load(f)

    # Preprocess focus token and get focus class (model prediction of input data with focus token)
    focus_cls = np.argmax(predict_fn(data))
    if exact_match:
        focus_token_lemma = focus_token.lower()
    else:
        focus_token_pos_tag = None
        for word, tag in pos_tag(tokenizer(data['sentence'])):
            if word == focus_token:
                focus_token_pos_tag = get_wordnet_pos(tag)
        focus_token_lemma = lemmatizer.lemmatize(focus_token.lower(), focus_token_pos_tag)

    # look through dataset for the salience scores of the focus token
    # first count if we have enough supporting data

    std_regions = []  # identifying information about the focus token salience score in different input texts
    for index, instance in enumerate(instances):
        # preprocess tokens in data
        sentence = [token['token'] for token in instance]
        if exact_match:
            sentence_lemmas = [token.lower() for token in sentence]
        else:
            pos_tags_sentence = pos_tag(sentence)
            sentence_lemmas = [
                lemmatizer.lemmatize(token.lower(), get_wordnet_pos(tag)) if get_wordnet_pos(tag) is not None
                else token.lower() for token, tag in pos_tags_sentence]

        # skip this data if focus token not present or if predicted label is not the focus class
        if focus_token_lemma not in sentence_lemmas:
            continue
        predicted_label = np.argmax(preds['logits'][index])
        if predicted_label != focus_cls:
            continue

        # calculate which region salience score for the focus token is in with respect to relative score distribution

        scores = np.array([token[str(focus_cls)] for token in instance])
        std_regions = get_std_region(focus_token_lemma=focus_token_lemma,
                                     sentence_lemmas=sentence_lemmas,
                                     sentence=sentence, scores=scores)



    # Display histogram and return a random sample or extreme values
    sorted_regions = sorted(std_regions, key=lambda x: x[0])
    print(sorted_regions)
    indicies_for_variance = get_variance_indicies(std_regions)
    if len(std_regions) == 1:
        tc_score = None
    else:
        tc_score = stats.describe(indicies_for_variance).variance

    print(tc_score)

    if visual:
        region_mapping = {1: "medium-high", 2: "high", 3: "very high", -3: "very low", -2: "low", -1: "medium-low"}
        display_std_region_histogram(std_regions,
                                     f"Distribution of salience scores for focus token: '{focus_token}' (model prediction = {focus_cls})")
        displays = list(range(len(sorted_regions)))
        # displays = [d for i, d in enumerate(displays) if i == 0 or i == le()]
        for (i, (region, sentence, scores, isNegative, index)) in enumerate(sorted_regions):
            if i == 0 or i == len(std_regions) - 1:
                combined_sentence = " ".join(sentence)
                labels = [token for i, token in enumerate(sentence)]
                display_score_heatmap(scores, labels,
                                      f"salience score for '{focus_token}' is in the '{region_mapping[region]}' region")

    high_and_low_instances = [d for i, d in enumerate(sorted_regions) if i == 0 or i == len(sorted_regions)]
    return {'TC score': tc_score,
            f'Samples from the high and low ends of the distribution for {focus_token}': high_and_low_instances}


def get_variance_indicies(regions_output):
    """
    Returns enumeration of different regions that the salience
    score value can be in in order to calculate the overal variance (TC score).
    """
    indicies = list(range(-6, 6, 1))
    indicies_for_variance = []
    for reg in regions_output:
        if reg[0] == -3 and reg[3] is False:
            indicies_for_variance.append(0)
        if reg[0] == -3 and reg[3]:
            indicies_for_variance.append(-1)
        if reg[0] == -2 and reg[3] is False:
            indicies_for_variance.append(1)
        if reg[0] == -2 and reg[3]:
            indicies_for_variance.append(-2)
        if reg[0] == -1 and reg[3] is False:
            indicies_for_variance.append(2)
        if reg[0] == -1 and reg[3]:
            indicies_for_variance.append(-3)
        if reg[0] == 1 and reg[3] is False:
            indicies_for_variance.append(3)
        if reg[0] == 1 and reg[3]:
            indicies_for_variance.append(-4)
        if reg[0] == 2 and reg[3] is False:
            indicies_for_variance.append(4)
        if reg[0] == 2 and reg[3]:
            indicies_for_variance.append(-5)
        if reg[0] == 3 and reg[3] is False:
            indicies_for_variance.append(5)
        if reg[0] == 3 and reg[3]:
            indicies_for_variance.append(-6)
    return indicies_for_variance


def test_std():
    scores_path = "../demo/serialized_data/SST-tiny-bert/lime/val.json"
    pred_path = "../demo/serialized_data/SST-tiny-bert/predictions_val.json"

    word = "mesmerizing"  # not, the, lack, good, ok, okay, positively, synthesis, mesmerizing, most, special, fancy, enough
    data = {'sentence': "even in its most tedious scenes , russian ark is mesmerizing .", 'label': "1"}
    # data = {'sentence': "it 's not the ultimate depression-era gangster movie .", 'label': "0"}
    # data = {'sentence': "overall very good for what it 's trying to do .", 'label': "1"}
    output = token_consistency(focus_token=word,
                               task=SEQ_TASK,
                               tokenizer=lambda x: x.split(),
                               data=data,
                               predict_fn=lambda x: np.array([0.1, 0.9]),
                               explanation_method=lambda x: [{"a": 1.0}],
                               dataset=None,
                               scores_path=scores_path,
                               pred_path=pred_path,
                               visual=True)
    print(output)


def main():
    test_std()


if __name__ == '__main__':
    main()
