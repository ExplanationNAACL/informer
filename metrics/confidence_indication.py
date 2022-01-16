"""

Confidence Indication metric as proposed in this paper by Atanasova et. al:
"A Diagnostic Study of Explainability Techniques for Text Classification"
https://arxiv.org/pdf/2009.13295.pdf

Addition functionality includes scatter plot views of saliency distance v. model confidence with learned linear
regression plot.

"""
from collections import defaultdict
import numpy as np
import random
import json

#  Scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

#  Importing utility functions
from salience_basic_util import display_heatmap, normalize_scores, SEQ_TASK, NLI_TASK, JSON_HEADER_SCORES

#  Plotting
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Callable, List, Dict


def up_sample(x, y):
    """
    Balances dataset by up-sampling data based on model confidence,
    so that all confidence intervals from 0-100 in intervals of 10
    have an equal number of samples. The motivation is that high performing
    models, e.g. transformer-based models, have a lot of high confidence examples,
    and the linear regression starts to predict the average confidence.

    Input
          - A list of features ('x').
          - A list of floats ('y') representing model confidence
            probabilities

    Output:
          - A list of up-sampled features
          - A list of up-sampled labels
    """
    buckets_idx = defaultdict(lambda: [])
    buckets_size = defaultdict(lambda: 0)
    for i, _y in enumerate(y):
        buckets_size[int(_y * 10)] += 1
        buckets_idx[int(_y * 10)].append(i)

    sample_size = max(list(buckets_size.values()))

    new_idx = []
    for _, bucket_ids in buckets_idx.items():
        do_replace = True
        if sample_size <= len(bucket_ids):
            do_replace = False
        chosen = np.random.choice(bucket_ids, sample_size, replace=do_replace)
        new_idx += chosen.tolist()

    random.shuffle(new_idx)

    return x[new_idx], y[new_idx]


def confidence_indication(task: str,
                          num_classes: int,
                          serialized: bool = True,  # REMOVE
                          explanation_method: Callable[[Dict[str, str]], List[Dict[str, float]]] = None,  # REMOVE
                          predict_fn: Callable[[Dict[str, str]], np.ndarray] = None,  # REMOVE
                          data: List[Dict[str, str]] = None,  # REMOVE
                          scores_path: str = None,
                          preds_path: str = None,
                          analyze_instance_level=False,
                          visual_lr=False,
                          seed=123) -> Dict[str, float]:
    """
    TODO: Remove explanation method, predict_fn, and data, & put support for serialized score in Informers class
    TODO: Only binary, ternary sequence classification and NLI supported.
    From the paper: https://arxiv.org/pdf/2009.13295.pdf
    Indication of how easily a model’s confidence can be
    identified from the salience score explanation.

    Input
      - A string ('task') specifying which classificaion
        task this is. Supported tasks are 'seq' and 'nli'.
      - A string ('num_classes') that specifies the number of classes in this task
      - A string ('scores_path') specifying path to saliency
        scores in the standard format
      - A string ('preds_path') specifying path to predicted
        logits and labels in the standard format
      - A bool ('analyze_instance_level') specifying whether
        heatmap and classification error should be displayed
        at the instance level. Useful if you want to visualize
        and compare scores. Visualizes salience scores for all
        possible classes.
      - A bool ('visual_lr') specifying whether or not to display a scatter plot and plot of the training data
        and learned linear regression.
      - A random seed ('seed') for splitting the data into train/validation for the linear regression.

    Output:
      -
    """
    assert num_classes == 2 or num_classes == 3
    if task == NLI_TASK:
        assert num_classes == 3
    assert task == NLI_TASK or task == SEQ_TASK
    if visual_lr:
        assert num_classes == 2

    saliency_file_train = scores_path
    pred_file_train = preds_path
    with open(saliency_file_train) as f:
        saliencies = json.load(f)
    with open(pred_file_train) as f:
        preds = json.load(f)

    # dataset split
    saliencies_train, saliencies_val, preds_train, preds_val = train_test_split(saliencies[JSON_HEADER_SCORES],
                                                                                preds['logits'],
                                                                                test_size=0.2,
                                                                                random_state=seed)
    classes = None
    if task == SEQ_TASK:
        classes = list(map(str, range(num_classes)))
    elif task == NLI_TASK:
        classes = list(map(str, range(3)))

    features_train = []
    y_train = []

    for index, instance in enumerate(saliencies_train):
        logits = preds_train[index]
        #  model confidence
        confidence = np.max(logits)
        y_train.append(confidence)
        if num_classes == 2:
            predicted_cls = str(np.argmax(logits))
            remaining_cls = classes.copy()
            remaining_cls.remove(predicted_cls)

            saliency_distance = [token[predicted_cls] - token[remaining_cls[0]] for token in instance]
            saliency_distance = sum(saliency_distance)
            features_train.append([saliency_distance])

        elif num_classes == 3:
            predicted_cls = str(np.argmax(logits))
            remaining_cls = classes.copy()
            remaining_cls.remove(predicted_cls)

            if task == NLI_TASK:
                diff_1_p = [token[predicted_cls] - token[remaining_cls[0]] for token in instance[0]['premise']]
                diff_2_p = [token[predicted_cls] - token[remaining_cls[1]] for token in instance[0]['premise']]
                diff_1_h = [token[predicted_cls] - token[remaining_cls[0]] for token in instance[0]['hypothesis']]
                diff_2_h = [token[predicted_cls] - token[remaining_cls[1]] for token in instance[0]['hypothesis']]

                #  Combining the premise and hypothesis, since LIT does LIME scores differently for NLI
                #  compared to the Atanasova et. al. paper
                diff_1_p.extend(diff_1_h)
                diff_1 = diff_1_p
                diff_2_p.extend(diff_2_h)
                diff_2 = diff_2_p
                assert len(diff_1) == len(diff_2)  # should be the same number of tokens
            else:
                diff_1 = [token[predicted_cls] - token[remaining_cls[0]] for token in instance]
                diff_2 = [token[predicted_cls] - token[remaining_cls[1]] for token in instance]

            #  feature consists of list of max, mean, and min of differences between lime scores
            feats = [sum(np.max([diff_1, diff_2], axis=0)),
                     sum(np.mean([diff_1, diff_2], axis=0)),
                     sum(np.min([diff_1, diff_2], axis=0))]

            features_train.append(feats)

    scaler = MinMaxScaler()
    features_train = scaler.fit_transform(np.array(features_train))

    up_features, up_y = up_sample(np.array(features_train), np.array(y_train))
    reg_upsample = LinearRegression().fit(up_features, up_y)
    reg = LinearRegression().fit(features_train, y_train)

    if num_classes == 2 and visual_lr:  # If binary classification, able to show a scatter plot
        plt.scatter(features_train.reshape(-1, ), y_train, color='black')
        plt.plot(features_train.reshape(-1, ), reg.predict(features_train), color='blue', linewidth=3)
        plt.ylabel('Model confidence')
        plt.xlabel('Saliency distance')
        plt.title('Learned linear regression model on training data')
        plt.show()

        plt.scatter(up_features.reshape(-1, ), up_y, color='black')
        plt.plot(up_features.reshape(-1, ), reg_upsample.predict(up_features), color='blue', linewidth=3)
        plt.ylabel('Model confidence')
        plt.xlabel('Saliency distance')
        plt.title('Learned linear regression learned on up-sampled training data')
        plt.show()

    #  Evaluation of linear regression model on validation set --------------
    features_val = []
    y_val = []
    for index, instance in enumerate(saliencies_val):
        logits = preds_val[index]
        confidence = np.max(logits)
        y_val.append(confidence)

        if num_classes == 3:
            predicted_cls = str(np.argmax(logits))
            remaining_cls = classes.copy()
            remaining_cls.remove(predicted_cls)
            if task == NLI_TASK:
                diff_1_p = [token[predicted_cls] - token[remaining_cls[0]] for token in instance[0]['premise']]
                diff_2_p = [token[predicted_cls] - token[remaining_cls[1]] for token in instance[0]['premise']]
                diff_1_h = [token[predicted_cls] - token[remaining_cls[0]] for token in instance[0]['hypothesis']]
                diff_2_h = [token[predicted_cls] - token[remaining_cls[1]] for token in instance[0]['hypothesis']]

                #  Combining the premise and hypothesis, since LIT does LIME scores differently for NLI
                #  compared to the Atanasova et. al. paper
                diff_1_p.extend(diff_1_h)
                diff_1 = diff_1_p
                diff_2_p.extend(diff_2_h)
                diff_2 = diff_2_p
                assert len(diff_1) == len(diff_2)  # should be the same number of tokens
            else:
                diff_1 = [token[predicted_cls] - token[remaining_cls[0]] for token in instance]
                diff_2 = [token[predicted_cls] - token[remaining_cls[1]] for token in instance]

            #  feature consists of list of max, mean, and min of differences between lime scores
            feats = [sum(np.max([diff_1, diff_2], axis=0)),
                     sum(np.mean([diff_1, diff_2], axis=0)),
                     sum(np.min([diff_1, diff_2], axis=0))]

            features_val.append(feats)

            if analyze_instance_level:
                if task == NLI_TASK:
                    p_ent_scores = [token['0'] for token in instance[0]['premise']]
                    h_ent_scores = [token['0'] for token in instance[0]['hypothesis']]
                    p_neu_scores = [token['1'] for token in instance[0]['premise']]
                    h_neu_scores = [token['1'] for token in instance[0]['hypothesis']]
                    p_con_scores = [token['2'] for token in instance[0]['premise']]
                    h_con_scores = [token['2'] for token in instance[0]['hypothesis']]

                    pred_confidence = reg.predict(scaler.transform([feats]))
                    pred_confidence_up = reg_upsample.predict(scaler.transform([feats]))
                    absolute_error_up = abs(confidence - pred_confidence_up)
                    mean_absolute_error_up = absolute_error_up * (1 / len(saliencies_val))

                    absolute_error = abs(confidence - pred_confidence)
                    mean_absolute_error_instance = absolute_error * (1 / len(saliencies_val))

                    print(f'Confidence: {round(float(confidence), 2)}\n'
                          f'Predicted model confidence: {round(float(pred_confidence), 2)}\n'
                          f'Predicted model confidence (upsampled training): {round(float(pred_confidence_up), 2)}\n'
                          f'Predicted class = {predicted_cls}\n'
                          f'Heatmap display in separate window:\n\n')

                    cmap = sns.diverging_palette(10, 240, as_cmap=True)
                    fig, ((p_ent, h_ent),
                          (p_neu, h_neu),
                          (p_con, h_con)) = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))

                    p_ent.set_title("Premise, entailment")
                    sns.heatmap(normalize_scores(np.array(p_ent_scores)).reshape(-1, 1),
                                ax=p_ent,
                                cmap=cmap,
                                linewidths=0.5,
                                vmin=-1,
                                vmax=1)

                    h_ent.set_title("Hypothesis, entailment")
                    sns.heatmap(normalize_scores(np.array(h_ent_scores)).reshape(-1, 1),
                                ax=h_ent,
                                cmap=cmap,
                                linewidths=0.5,
                                vmin=-1,
                                vmax=1)

                    p_neu.set_title("Premise, neutral")
                    sns.heatmap(normalize_scores(np.array(p_neu_scores)).reshape(-1, 1),
                                ax=p_neu,
                                cmap=cmap,
                                linewidths=0.5,
                                vmin=-1,
                                vmax=1)

                    h_neu.set_title("Hypothesis, neutral")
                    sns.heatmap(normalize_scores(np.array(h_neu_scores)).reshape(-1, 1),
                                ax=h_neu,
                                cmap=cmap,
                                linewidths=0.5,
                                vmin=-1,
                                vmax=1)

                    p_con.set_title("Premise, contradiction")
                    sns.heatmap(normalize_scores(np.array(p_con_scores)).reshape(-1, 1),
                                ax=p_con,
                                cmap=cmap,
                                linewidths=0.5,
                                vmin=-1,
                                vmax=1)

                    h_con.set_title("Hypothesis, contradiction")
                    sns.heatmap(normalize_scores(np.array(h_con_scores)).reshape(-1, 1),
                                ax=h_con,
                                cmap=cmap,
                                linewidths=0.5,
                                vmin=-1,
                                vmax=1)

                    plt.show()
                else:
                    cmap = sns.diverging_palette(10, 240, as_cmap=True)
                    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))

                    ax1.set_title("Explanation for class 0")
                    sns.heatmap(normalize_scores(np.array([token['0'] for token in instance])),
                                ax=ax1,
                                cmap=cmap,
                                linewidths=0.5,
                                vmin=-1,
                                vmax=1)

                    ax2.set_title("Explanation for class 1")
                    sns.heatmap(normalize_scores(np.array([token['1'] for token in instance])),
                                ax=ax2,
                                cmap=cmap,
                                linewidths=0.5,
                                vmin=-1,
                                vmax=1)

                    ax3.set_title("Explanation for class 2")
                    sns.heatmap(normalize_scores(np.array([token['2'] for token in instance])),
                                ax=ax3,
                                cmap=cmap,
                                linewidths=0.5,
                                vmin=-1,
                                vmax=1)
                    plt.show()

        elif num_classes == 2:
            predicted_cls = str(np.argmax(logits))

            heatmap_saliency_0 = [token['0'] for token in instance]
            heatmap_saliency_1 = [token['1'] for token in instance]

            saliency_distance = [token[predicted_cls] - token[remaining_cls[0]] for token in instance]
            saliency_distance = sum(saliency_distance)
            features_val.append([saliency_distance])

            if analyze_instance_level:
                pred_confidence = reg.predict(scaler.transform([[saliency_distance]]))
                pred_confidence_up = reg_upsample.predict(scaler.transform([[saliency_distance]]))
                absolute_error_up = abs(confidence - pred_confidence_up)
                mean_absolute_error_up = absolute_error_up * (1 / len(saliencies_val))

                absolute_error = abs(confidence - pred_confidence)
                mean_absolute_error_instance = absolute_error * (1 / len(saliencies_val))

                print(f'Confidence: {round(float(confidence), 2)}\n'
                      f'Predicted model confidence: {round(float(pred_confidence), 2)}\n'
                      f'Predicted model confidence (upsampled training): {round(float(pred_confidence_up), 2)}\n'
                      f'Predicted class = {predicted_cls}\n'
                      f'Heatmap display in separate window:\n\n')
                display_heatmap(salience_scores=normalize_scores(np.array(heatmap_saliency_0)),
                                salience_scores_2=normalize_scores(np.array(heatmap_saliency_1)),
                                title=f'Explanation for class 0',
                                title_2='Explanation for class 1',
                                normalized=True).show()

    features_val = scaler.transform(np.array(features_val))

    return {"MAE on train set for linear regression model": mean_absolute_error(y_train, reg.predict(features_train)),
            "MAE on train set for linear regression model (on upsampled data)": mean_absolute_error(up_y, reg_upsample.predict(up_features)),
            "MAE on validation set with trained linear regression model": mean_absolute_error(y_val, reg.predict(features_val)),
            "MAE on validation set with trained linear regression model (on upsampled data) ": mean_absolute_error(y_val, reg_upsample.predict(features_val))}
