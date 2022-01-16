"""
This file is to demonstrate what types of wrappers a user would need to make to use their custom explainer,
dataset, and model, with our Informers library. In this case, our custom explainer, dataset, and model is from LIT.

- This file contains the Wrapper functions for the LIT explainers, models, and dataset to adhere to the Informers Specs
as described in our documentation.

- See main function for demos on how to use these wrapper functions to interact with LIT.
"""
from lit_nlp.examples.datasets import glue
from lit_nlp.examples.models import glue_models
from lit_nlp.components import gradient_maps
from lit_nlp.components import lime_explainer
import copy
from lit_nlp.api import components as lit_components
from typing import List, Dict
import numpy as np
from salience_basic_util import generate_serialization_files, SEQ_TASK, NLI_TASK


class LITSST2Wrapper:
    def __init__(self,
                 model: glue_models,
                 dataset: glue.SST2Data,
                 explainer: lit_components.Interpreter):
        self.model = model
        self.dataset = dataset
        self.explainer = explainer
        self.num_classes = 2

    def get_dataset(self) -> List[Dict[str, str]]:
        """
        Returns dataset in the format asked by Informers specs
        [{'sentence': 'I am happy ...', 'label': '0'}, {}, ...]
        """
        return self.dataset.examples

    def explanation_method(self, input_text: Dict[str, str]) -> List[Dict[str, float]]:
        """
        Returns explanations for each possible class in the form of salience scores for
        each token.
        Input:
            - Takes in dict in format: {'sentence': 'That movie was interesting', 'label': 0}
        Output:
            - Returns a list of dictionaries mapping token to salience score. Number of dictionaries equal
              number of classes (which is 2).
        """
        pass

    def predict_fn(self, input_text: Dict[str, str]) -> np.ndarray:
        """
        Takes in input text and labels and outputs probability distribution
        over number of classes (in this case, 2)

        Input:
            - One dictionary in the form {'sentence': 'A rollercoaster of emotions', 'label': '0'}.
                representing the input text for the model. Labels are optional.
        Output:
            - A 1D numpy array object representing the probabilities over all possible classes.

        TODO: Make a type scheme or something instead of manually checking if input is formatted correctly

        """
        #  Check input formatting
        assert len(list(input_text.keys())) == self.num_classes and \
               'sentence' in list(input_text.keys()) and \
               'label' in list(input_text.keys())

        #  Perform model inference on input text
        output = self.model.predict([input_text])
        probs = np.array([o['probas'] for o in output][0])

        #  Check output formatting and correctness
        assert probs.shape == (self.num_classes,)
        assert round(sum(probs)) == 1
        return probs


class SST2LIME(LITSST2Wrapper):
    """
    A class to provide the wrapper functions for the LIT implementation
    of model, dataset, and explainer to adhere to the Informer specifications.

    Specs:                explanation_method: Callable[[List[Dict[str, str]]], List[Dict[str, float]]],
                          predict_fn: Callable[[List[Dict[str, str]]], np.ndarray],
                          dataset: List[Dist[str, str]]

    Currently only supports LIT's LIME explainer.
    """

    def __init__(self,
                 model: glue_models.SST2Model,
                 dataset: glue.SST2Data,
                 explainer: lime_explainer.LIME):
        super().__init__(model, dataset, explainer)

    def explanation_method(self, input_text: Dict[str, str]) -> List[Dict[str, float]]:
        """
        Returns explanations for each possible class in the form of salience scores for
        each token.
        Input:
            - Takes in dict in format: {'sentence': 'That movie was interesting', 'label': 0}
        Output:
            - Returns a list of dictionaries mapping token to salience score. Number of dictionaries equal
              number of classes (which is 2).
        """
        explanations = []
        config = self.explainer.config_spec()
        config[lime_explainer.KERNEL_WIDTH_KEY] = config[lime_explainer.KERNEL_WIDTH_KEY].default
        config[lime_explainer.MASK_KEY] = config[lime_explainer.MASK_KEY].default
        config[lime_explainer.NUM_SAMPLES_KEY] = config[lime_explainer.NUM_SAMPLES_KEY].default
        config[lime_explainer.SEED_KEY] = config[lime_explainer.SEED_KEY].default

        for i in range(self.num_classes):
            config[lime_explainer.CLASS_KEY] = str(i)
            output = self.explainer.run([input_text], self.model, dataset=self.dataset, config=config)
            scores = output[0]['sentence'].salience
            tokens = input_text['sentence'].split()  # this is the default LIME tokenizer
            assert len(tokens) == len(scores)
            exp = {}
            for token, score in zip(tokens, scores):
                exp[token] = score
            explanations.append(exp)
        return explanations


class SST2GradNorm(LITSST2Wrapper):
    """
    A class to provide the wrapper functions for the LIT implementation
    of model, dataset, and explainer to adhere to the Informer specifications.
    LIT's Gradient L2 Norm explainer.

    Specs:                explanation_method: Callable[[List[Dict[str, str]]], List[Dict[str, float]]],
                          predict_fn: Callable[[List[Dict[str, str]]], np.ndarray],
                          dataset: List[Dist[str, str]]

    Currently only supports prediction class explanations.

    """

    def __init__(self,
                 model: glue_models.SST2Model,
                 dataset: glue.SST2Data,
                 explainer: gradient_maps.GradientNorm):
        super().__init__(model, dataset, explainer)

    def explanation_method(self, input_text: Dict[str, str]) -> List[Dict[str, float]]:
        """
        Returns explanations for each possible class in the form of salience scores for
        each token. The first set of explanation is actually for predicted class, but
        the second set is just a placeholder. Will have to do this for now.
        TODO: Ask LIT developer team about specifying class for gradient-based lit_components.Interpreter

        Input:
            - Takes in dict in format: {'sentence': 'That movie was interesting', 'label': 0}
        Output:
            - Returns a list of dictionaries mapping token to salience score. Number of dictionaries equal
              number of classes (which is 2).
        """
        explanations = []
        output = self.explainer.run([input_text], self.model, dataset=self.dataset)
        scores = output[0]['token_grad_sentence'].salience
        tokens = output[0]['token_grad_sentence'].salience
        assert len(tokens) == len(scores)
        exp = {}
        for token, score in zip(tokens, scores):
            exp[token] = score
        explanations.append(exp)
        explanations.append({})  # placeholder
        return explanations


class LITNLIWrapper:
    def __init__(self,
                 model: glue_models,
                 dataset: glue.MNLIData,
                 explainer: lit_components.Interpreter):
        self.model = model
        self.dataset = dataset
        self.explainer = explainer
        self.num_classes = 3
        #  GLUE labeling scheme is inconsistent. MNLI doesn't label by index
        self.classes = ['entailment', 'neutral', 'contradiction']

    def get_dataset(self) -> List[Dict[str, str]]:
        """
        Returns dataset in the format asked by Informers specs
        [{'premise': 'I am happy ...', 'hypothesis': 'I am sad ...', 'label': '2'}, {}, ...]
        """
        examples = []
        for example in self.dataset.examples:
            reformatted = copy.deepcopy(example)
            reformatted['label'] = str(self.classes.index(example['label']))
            examples.append(reformatted)
        return examples

    def explanation_method(self, input_text: Dict[str, str]) -> List[Dict[str, Dict[str, float]]]:
        """
        Returns explanations for each possible class in the form of salience scores for
        each token.
        Input:
            - Takes in dict in format:
            {'premise': 'That movie was interesting', 'hypothesis': 'A boring movie', 'label': 2}
        Output:
            - Returns a list of dictionaries mapping premise and hypothesis to dictionaries mapping token to
              salience score. Number of dictionaries equal number of classes (which is 3).
            [{'premise': {'a': 1.2, 'b': 1.3}, 'hypothesis': {'c': 1.3, 'd': 1.4},
             {'premise': {'a': 1.5, 'b': -6.3}, 'hypothesis': {'c': 7.3, 'd': 1.8},
             {'premise': {'a': 1.5, 'b': 9.3}, 'hypothesis': {'c': 0.3, 'd': 1.0}]
        """
        pass

    def predict_fn(self, input_text: Dict[str, str]) -> np.ndarray:
        """
        Takes in input text and labels and outputs probability distribution
        over number of classes (in this case, 2)

        Input:
            - One dictionary in the form
                {'premise': 'That movie was interesting', 'hypothesis': 'A boring movie', 'label': 2}
                representing the input text for the model. Labels are optional.
        Output:
            - A 1D numpy array object representing the probabilities over all possible classes.

        TODO: Make a type scheme or something instead of manually checking if input is formatted correctly

        """
        #  Check input formatting
        assert len(list(input_text.keys())) == self.num_classes and \
               'premise' in list(input_text.keys()) and \
               'hypothesis' in list(input_text.keys()) and \
               'label' in list(input_text.keys())

        #  Perform model inference on input text
        input_text_change = copy.deepcopy(input_text)
        input_text_change['label'] = self.classes[int(input_text['label'])]  # need string label instead of index
        output = self.model.predict([input_text])
        probs = np.array([o['probas'] for o in output][0])

        #  Check output formatting and correctness
        assert probs.shape == (self.num_classes,)
        assert round(sum(probs)) == 1
        return probs


class MNLILIME(LITNLIWrapper):
    def __init__(self,
                 model: glue_models,
                 dataset: glue.MNLIData,
                 explainer: lime_explainer.LIME):
        super().__init__(model, dataset, explainer)

    def explanation_method(self, input_text: Dict[str, str]) -> List[Dict[str, Dict[str, float]]]:
        """
        Returns explanations for each possible class in the form of salience scores for
        each token.
        Input:
            - Takes in dict in format:
            {'premise': 'That movie was interesting', 'hypothesis': 'A boring movie', 'label': 2}
        Output:
            - Returns a list of dictionaries mapping premise and hypothesis to dictionaries mapping token to
              salience score. Number of dictionaries equal number of classes (which is 3).
            [{'premise': {'a': 1.2, 'b': 1.3}, 'hypothesis': {'c': 1.3, 'd': 1.4},
             {'premise': {'a': 1.5, 'b': -6.3}, 'hypothesis': {'c': 7.3, 'd': 1.8},
             {'premise': {'a': 1.5, 'b': 9.3}, 'hypothesis': {'c': 0.3, 'd': 1.0}]
        """
        explanations = []
        config = self.explainer.config_spec()
        config[lime_explainer.KERNEL_WIDTH_KEY] = config[lime_explainer.KERNEL_WIDTH_KEY].default
        config[lime_explainer.MASK_KEY] = config[lime_explainer.MASK_KEY].default
        config[lime_explainer.NUM_SAMPLES_KEY] = config[lime_explainer.NUM_SAMPLES_KEY].default
        config[lime_explainer.SEED_KEY] = config[lime_explainer.SEED_KEY].default

        for i in range(self.num_classes):
            config[lime_explainer.CLASS_KEY] = str(i)
            output = self.explainer.run([input_text], self.model, dataset=self.dataset, config=config)
            # extract tokens and salience scores
            scores_premise = output[0]['premise'].salience
            tokens_premise = output[0]['premise'].tokens

            scores_hypothesis = output[0]['hypothesis'].salience
            tokens_hypothesis = output[0]['hypothesis'].tokens

            assert len(tokens_hypothesis) == len(scores_hypothesis)
            assert len(tokens_premise) == len(scores_premise)

            exp_premise = {}
            for token, score in zip(tokens_premise, scores_premise):
                exp_premise[token] = score
            exp_hypothesis = {}
            for token, score in zip(tokens_hypothesis, scores_hypothesis):
                exp_hypothesis[token] = score
            explanations.append({'premise': exp_premise, 'hypothesis': exp_hypothesis})
        return explanations


def main():
    # Stanford Sentiment Treebank (binary)

    # How to use lit_wrapper to generate explanation
    model = glue_models.SST2Model('models/sst2-tiny-bert')
    dataset = glue.SST2Data('validation')
    lime = lime_explainer.LIME()
    lit_wrapper = SST2LIME(model, dataset, lime)
    dataset = lit_wrapper.get_dataset()[:50]
    #for class_explanation in lit_wrapper.explanation_method(dataset[0]):
        #print(class_explanation)
    # How to serialize explanation and model outputs
    files = generate_serialization_files(task=SEQ_TASK,
                                         num_classes=2,
                                         explainer=lit_wrapper.explanation_method,
                                         predict_fn=lit_wrapper.predict_fn,
                                         dataset=dataset[:2],
                                         scores_path="./scores_sst.json",
                                         preds_path="./preds_sst.json")
    print(files)
    exit()

    # Multi-genre Natural Language Inference

    # How to use lit_wrapper to generate explanation
    model = glue_models.MNLIModel('models/mbert-mnli')
    dataset = glue.MNLIData('validation_matched')
    lime = lime_explainer.LIME()
    lit_wrapper = MNLILIME(model, dataset, lime)
    dataset = lit_wrapper.get_dataset()
    # How to serialize explanation and model outputs
    files = generate_serialization_files(task=NLI_TASK,
                                         num_classes=3,
                                         explainer=lit_wrapper.explanation_method,
                                         predict_fn=lit_wrapper.predict_fn,
                                         dataset=dataset[:2],
                                         scores_path="./scores_nli.json",
                                         preds_path="./preds_nli.json")
    print(files)

    for class_explanation in lit_wrapper.explanation_method(dataset[0]):
        print(class_explanation)


if __name__ == '__main__':
    main()
