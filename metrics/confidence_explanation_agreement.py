"""
Confidence-Explanation Agreement Metric.
This metric calculates values on both the instance and corpus level and is concerned with how well an explanation
method reflects the model confidence for a particular instance.
See documentation or paper for details on calculations.
"""
from salience_basic_util import normalize_scores

#  visualizations
import matplotlib.pyplot as plt
from salience_basic_util import display_heatmap

#  linear regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

import json
import numpy as np

from typing import Callable, List, Dict, Tuple, Union, Any
import sys

#  importing all the switch variables
from salience_basic_util import SEQ_TASK, NLI_TASK, CORPUS_LEVEL, INSTANCE_LEVEL, JSON_HEADER_SCORES

sys.path.append('../..')


def _get_outliers_sum(salience_scores: np.ndarray) -> Tuple[float, float]:
    """
    Calculates the high valued positive and negative outlier of a set of salience scores and sums them
    up. If no outliers present, just gets the highest positive and negative values. Default is 0.

    Input:
        - ('salience_scores') A numpy array that contains floats representing the token-importance scores.
    Output:
        - Tuple: (sum of all the positive outliers, sum of absolute value of the negative outliers)
    """
    q3, q1 = np.percentile(abs(salience_scores), [75, 25])
    iqr = q3 - q1
    outfencelarger = (iqr * 1.5) + q3  # only looking at high salience scores

    outliers_pos = []  # all the high positive salience values
    outliers_neg = []  # all the high negative salience values

    #  in the case that there are no outliers, we pick the max neg and pos salience scores
    max_neg = 0
    max_pos = 0

    for elem in salience_scores:
        if abs(elem) > outfencelarger:
            if elem > 0:
                outliers_pos.append(elem)
            elif elem < 0:
                outliers_neg.append(elem)

        if elem < 0 and abs(elem) > max_neg:
            max_neg = elem
        elif elem > 0 and elem > max_pos:
            max_pos = elem

    sum_pos_outliers = sum(outliers_pos) if len(outliers_pos) > 0 else max_pos
    sum_neg_outliers = sum(list(map(abs, outliers_neg))) if len(outliers_neg) > 0 else abs(max_neg)
    return sum_pos_outliers, sum_neg_outliers


def calculate_cea_score(model_conf: float,
                        salience_scores: np.ndarray,
                        salience_scores_hypothesis: np.ndarray = None,
                        scoring: bool = True) -> Tuple[int, float]:
    """
    Calculates the Confidence-Explanation Agreement score as defined in the specs.
    Finds the index of the model confidence as defined in specs, and the index of
    either the difference between positive and negative outliers in the explanation.

    Input:
        - ('model_conf') A float representing the model probability of the predicted class
        - ('salience_scores') A list of floats representing the importance scores for each token in the input text
        - ('salience_scores_hypothesis') Optional. A list of floats representing the importance score for each token
                                         in the hypothesis text.
        - ('scoring') A boolean that denotes whether or not the calculations will be used to get the CEA score
                      or not. If yes, then salience scores will be normalized. Otherwise, the salience scores will
                      unprocessed.

    Output:
        - An integer and float pair. Integer representing the CEA score, ranging from -4 to 4. Any value past 0 in either direction
          should signal a disagreement, and float representing the outlier difference/
    """
    # Get the index of the model confidence according to defined intervals (see documentation for details)
    model_conf_index = None
    if model_conf < 60:
        model_conf_index = 0
    if model_conf < 70:
        model_conf_index = 1
    elif model_conf < 80:
        model_conf_index = 2
    elif model_conf < 90:
        model_conf_index = 3
    else:
        model_conf_index = 4

    if scoring:
        # For this metric, the absolute value of the salience scores must add up to 1
        salience_scores = normalize_scores(
            salience_scores)  # required in order for difference to fall within known bounds

    # Get the outliers and calculate the difference.
    sum_pos_outliers, sum_neg_outliers = _get_outliers_sum(salience_scores)
    diff_outliers = sum_pos_outliers - sum_neg_outliers

    #  In the NLI case, calculates the average of the difference of the outliers for the premise
    #  and the hypothesis. Should keep the range between 0 and 1.
    if salience_scores_hypothesis is not None:
        if scoring:
            salience_scores_hypothesis = normalize_scores(salience_scores_hypothesis)
        sum_pos_outliers_h, sum_neg_outliers_h = _get_outliers_sum(salience_scores_hypothesis)
        diff_outliers_h = sum_pos_outliers_h - sum_neg_outliers_h
        diff_outliers = (diff_outliers + diff_outliers_h) / 2

    if not scoring:
        return -123, diff_outliers

    # Get the index of the difference of the outliers according to defined intervals (see documentation for details)
    if diff_outliers < 0:
        diff_index = 0
    elif diff_outliers < 0.3:
        diff_index = 1
    elif diff_outliers < 0.5:
        diff_index = 2
    elif diff_outliers < 0.7:
        diff_index = 3
    else:
        diff_index = 4

    # CEA is defined as the difference between model confidence index and outlier difference index
    cea = model_conf_index - diff_index

    return cea, diff_outliers


def confidence_explanation_agreement_global(explanation_method: Callable[[Dict[str, str]], List[Dict[str, float]]],
                                            predict_fn: Callable[[Dict[str, str]], np.ndarray],
                                            data: List[Dict[str, str]],
                                            visual: bool = False) -> Dict[str, float]:
    """
    Confidence-explanation agreement metric on the corpus-level.
    Calculates the explanation confidence (either using diff between pos and neg outliers or Jensen-Shannon
    divergence from uniform) and fits a line of best fit using Ordinary Least Square between explanation
    and model confidence.
    Input:
        - A function that takes in a data point in the form of {'sentence': 'something is here', 'label': '0'} and
          outputs salience scores for each token, per class ('explanation method')
        - A function that outputs a probability distribution over possible labels ('predict_fn')
        - A list of datapoints in the form of {'sentence': 'something is here', 'label': '0'} ('data')
        - A boolean option that can output a scatter plot of the learned linear model ('visual')

    Output:
        - The coefficient of determination (R^2), coefficient of the explanation confidence variable,
          and the mean absolute error of the linear regression model.
    """
    pass


def confidence_explanation_agreement_instance(explanation_method: Callable[[Dict[str, str]], List[Dict[str, float]]],
                                              predict_fn: Callable[[Dict[str, str]], np.ndarray],
                                              data: Dict[str, str],
                                              visual: bool = False) -> Dict[str, Any]:
    """
    Confidence-explanation agreement metric on the instance-level.
    Calculates the disagreement from -4 to 4 between the explanation and the model confidence.
    Input:
        - A function that takes in a data point in the form of {'sentence': 'something is here', 'label': '0'} and
          outputs salience scores for each token, per class ('explanation method')
        - A function that outputs a probability distribution over possible labels ('predict_fn')
        - A single datapoint in the form of {'sentence': 'something is here', 'label': '0'} ('data')
        - A boolean option that can output a heatmap of the explanation for the input ('visual')

    Output:
        - Sequence(s), model confidence, and the CEA score from -4 to 4 for the input data.
    """
    pass


def confidence_explanation_agreement(task: str,
                                     serialized: bool,
                                     explanation_method: Callable[[Dict[str, str]], List[Dict[str, float]]] = None,
                                     predict_fn: Callable[[Dict[str, str]], np.ndarray] = None,
                                     data: List[Dict[str, str]] = None,
                                     output: str = 'instance',
                                     visual: bool = False,
                                     scores_path: str = None,
                                     preds_path: str = None) -> Union[List[Dict[str, Any]],
                                                                      Dict[str, float]]:
    """
    For description, see confidence_explanation_agreement_instance if output = 'instance'
    or confidence_explanation_agreement_global if output = 'corpus'.

    Three new things, however:
    Input:
        - ('task') A tuple where first element is a string denoting the task, which could be either "seq" or "nli",
                   where "seq" signals single sequence classification, and "nli" signals classification of relation
                   between two sequences, i.e. natural language inference (NLI).
        - ('serialized') A boolean denoting whether the explanation scores and model inference outputs
                        have been serialized according to specifications. If this is true,
                        ('predict_fn'), ('explanation_method'), and ('data') will not be used.
        - ('scores_path') A string denoting the path to the serialized scores. Scores are serialized
                          according to specs. Should be in a json file.
        - ('preds_path')  A string denoting the path to the serialized model inference probabilities and gold labels,
                         according to specs. Should be in a json file.
    """
    # Make sure output and task is valid
    assert task in [SEQ_TASK, NLI_TASK] and output in [CORPUS_LEVEL, INSTANCE_LEVEL]

    # Extract serialized salience scores
    if serialized:
        assert scores_path and preds_path  # make sure serialized scores path is specified

        with open(scores_path, 'rb') as f:
            sals = json.load(f)
        with open(preds_path, 'rb') as f:
            preds = json.load(f)
        data = sals[JSON_HEADER_SCORES]
    else:
        # make sure that predict function, explanation method, and data are specified
        assert predict_fn and explanation_method and data

    # Go through dataset
    outputs = []
    x_train = []
    y_train = []
    indices = range(len(data))
    for index in indices:
        if not serialized:
            #  Check data formatting
            if task == SEQ_TASK:
                assert len(list(data[index].keys())) == 2 and \
                       'sentence' in list(data[index].keys()) and \
                       'label' in list(data[index].keys())
            else:
                assert len(list(data[index].keys())) == 3 and \
                       'premise' in list(data[index].keys()) and \
                       'hypothesis' in list(data[index].keys()) and \
                       'label' in list(data[index].keys())

            # model inference
            probs = predict_fn(data[index])
            predicted_cls = np.argmax(probs)
            model_confidence = np.max(probs) * 100

            # salience scores
            if task == SEQ_TASK:
                salience_scores = np.array([score for token, score in explanation_method(data[index])[predicted_cls].items()])
            else:
                salience_scores = (np.array([score for token, score in explanation_method(data[index])[predicted_cls]['premise'].items()]),
                                   np.array([score for token, score in explanation_method(data[index])[predicted_cls]['hypothesis'].items()]))

            # sentence and gold labels
            if task == SEQ_TASK:
                sentence = data[index]['sentence']
            else:
                sentence = (data[index]['premise'], data[index]['hypothesis'])

            gold_label = data[index]['label']
        else:
            # model inference
            probs = preds['logits'][index]
            predicted_cls = np.argmax(probs)
            model_confidence = np.max(probs) * 100

            # salience scores
            if task == SEQ_TASK:
                salience_scores = np.array([token[str(predicted_cls)] for token in data[index]])
                # TODO: specify somewhere that for display reasons, we're concatenating the tokens with whitespace.
                sentence = " ".join([token['token'] for token in data[index]])
            else:
                # salience scores
                salience_scores = (np.array([token[str(predicted_cls)] for token in data[index][0]['premise']]),
                                   np.array([token[str(predicted_cls)] for token in data[index][0]['hypothesis']]))
                sentence = (" ".join([token['token'] for token in data[index][0]['premise']]),
                            " ".join([token['token'] for token in data[index][0]['hypothesis']]))

            gold_label = preds['label'][index]

        # Get the score if instance level, and (unnormalized) outlier difference if corpus level
        if output == INSTANCE_LEVEL:
            if task == SEQ_TASK:
                score, outlier_diff = calculate_cea_score(model_conf=model_confidence,
                                                          salience_scores=salience_scores,
                                                          scoring=True)
            else:
                score, outlier_diff = calculate_cea_score(model_conf=model_confidence,
                                                          salience_scores=salience_scores[0],
                                                          salience_scores_hypothesis=salience_scores[1],
                                                          scoring=True)
        else:
            if task == SEQ_TASK:
                score, outlier_diff = calculate_cea_score(model_conf=model_confidence,
                                                          salience_scores=salience_scores,
                                                          scoring=False)
            else:
                score, outlier_diff = calculate_cea_score(model_conf=model_confidence,
                                                          salience_scores=salience_scores[0],
                                                          salience_scores_hypothesis=salience_scores[1],
                                                          scoring=False)

        if visual and output == INSTANCE_LEVEL:
            print(f"Sequence(s): {sentence}\n"
                  f"Model confidence = {round(model_confidence, 2)}\n"
                  f"Outlier difference = {round(outlier_diff, 2)}\n"
                  f"CEA score = {score}\n"
                  f"Prediction = {predicted_cls}\n"
                  f"Gold label (if available) = {gold_label}\n"
                  f"Heatmap display in separate window:\n\n")
            # TODO: Also, make it known that displays of salience scores automatically make it normalized
            if task == SEQ_TASK:
                display_heatmap(salience_scores=normalize_scores(salience_scores),
                                title=sentence,
                                normalized=True,
                                cell_labels=[token['token'] for token in data[index]]).show()
            else:
                display_heatmap(salience_scores=normalize_scores(salience_scores[0]),
                                salience_scores_2=normalize_scores(salience_scores[1]),
                                title=sentence[0],
                                title_2=sentence[1],
                                cell_labels=[token['token'] for token in data[index][0]['premise']],
                                cell_labels_2=[token['token'] for token in data[index][0]['hypothesis']],
                                normalized=True).show()

        if output == CORPUS_LEVEL:
            x_train.append([outlier_diff])
            y_train.append(model_confidence)
        else:
            outputs.append({'sentence(s)': sentence,
                            'model confidence': model_confidence,
                            'CEA score': score})

    #  Output
    if output == INSTANCE_LEVEL:
        return outputs  # of type List[Dict[str, Any]]
    else:
        # Fit a linear regression model (OLS) on data
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(np.array(x_train))
        reg = LinearRegression().fit(x_train, y_train)
        y_pred = reg.predict(x_train)
        if visual:
            plt.scatter(x_train, y_train, color='black')
            plt.plot(x_train, y_pred, color='blue', linewidth=3)
            plt.ylabel('Model confidence')
            plt.xlabel('Explanation confidence')
            plt.show()
        return {'Coefficient of determination R^2': reg.score(x_train, y_train),
                'Coefficient of independent variable (outlier diff)': reg.coef_[0],
                'Mean absolute error': mean_absolute_error(x_train, y_pred)}  # of type Dict[str, float]

