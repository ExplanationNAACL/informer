from pipeline import Informer
from salience_basic_util import SEQ_TASK, NLI_TASK, CORPUS_LEVEL, INSTANCE_LEVEL
import random
import matplotlib.pyplot as plt
import numpy as np
from lit_nlp.examples.datasets import glue
from lit_nlp.examples.models import glue_models
from lit_nlp.components import lime_explainer
from informers_specs_util import SST2LIME
from salience_basic_util import generate_serialization_files, SEQ_TASK, NLI_TASK


def instance_level_demo():
    model = glue_models.SST2Model('models/sst2-tiny-bert')
    dataset = glue.SST2Data('validation')
    lime = lime_explainer.LIME()
    lit_wrapper = SST2LIME(model, dataset, lime)
    informer_obj = Informer(explainer=lit_wrapper.explanation_method,
                            dataset=lit_wrapper.get_dataset(),
                            predictor=lit_wrapper.predict_fn,
                            task=SEQ_TASK,
                            num_classes=2,
                            scores_path="serialized_data/SST-tiny-bert/lime/val.json",
                            preds_path="serialized_data/SST-tiny-bert/predictions_val.json",
                            ui=False
                            )
    cea_scores = informer_obj.confidence_explanation_agreement(instance=True, visual=True)
    print(f"*** A SAMPLE OF 10 RANDOM INSTANCES FROM THE INPUT DATASET ***\n")
    sample = random.sample(cea_scores, 10)
    for i in cea_scores[:10]:
        print(i)

    print(f"\n\n*** DISTRIBUTION OF CEA SCORES OVER ENTIRE DATASET (SEE POP-UP WINDOW) ***\n")
    '''counter = {}
    for i in range(-4, 5):
        counter[i] = 0
    for i in cea_scores:
        counter[i['CEA score']] += 1'''
    hist = Informer.summary_statistics_from_cea(cea_scores)
    counter = hist.chart(8, low=-4, high=4, title="DISTRIBUTION OF CEA SCORES OVER THE ENTIRE DATASET")
    '''plt.bar(x=counter.keys(), height=counter.values())
    plt.xticks(np.arange(-4, 5))
    plt.title("DISTRIBUTION OF CEA SCORES OVER THE ENTIRE DATASET")
    plt.show()'''
    
    print(f"\n\n*** INSTANCES IN DATASET WHERE EXPLANATION AND MODEL CONFIDENCE AGREE ***\n")
    for i in cea_scores:
        if i['CEA score'] == 0:
            print(i['sentence(s)'])


def corpus_level_demo():
    print("*** LEARNED LINEAR REGRESSION MODEL FOR EXPLANATION CONFIDENCE VERSUS MODEL CONFIDENCE  ***\n\n")

    lin_reg_output = Informer.confidence_explanation_agreement_serialized(
            scores_path="serialized_data/SST-tiny-bert/lime/val.json",
            preds_path="serialized_data/SST-tiny-bert/predictions_val.json",
            instance=False, visual=True)

    for k, v in lin_reg_output.items():
        print(k, "|", v)


def main():
    # outputs a list of CEA scores, model confidence for each text in the dataset, distribution histogram
    # of CEA scores over an entire dataset, and instances where CEA = 0
    instance_level_demo()

    # outputs a scatterplot in a separate window, then prints values in standard output
    #corpus_level_demo()


if __name__ == '__main__':
    main()
