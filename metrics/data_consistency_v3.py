# begin script -------------------------------------------------------

""" data_consistency_v3.py
a version of DC script to accomodate reading from json files
everytime you want to run the metric, just for demo. json files hold 
serialized explanations, and predictions.
"""

# imports ------------------------------------------------------------

from   sklearn.preprocessing import MinMaxScaler
from   tqdm                  import tqdm
import random
import torch
import matplotlib.pyplot     as     plt
import seaborn               as     sns
import numpy as np
import json
from   salience_basic_util   import JSON_HEADER_SCORES

# func def -----------------------------------------------------------

class Informers:

    def __init__(self, data, model_fn, explainer_fn):

        """
        class constructor.
            params:
                data: type: List[Dict[str, str]]:
                    the data in format,
                        [{'sentence': '...', 'label': 0}, ..., {...}]
                model_fn: type:
                  Callable[Dict[str, str]] -> np.ndarray
                    a wrapper function for your model_fn, we advise
                    that if there is disagreement between our
                    expected format of the data and the your model's,
                    you right this callable to handle resolve
                    that disagreement. 
                explainer_fn: type:
                  Callable[Dict[str, str]] ->  List[Dict[str, float]]
                    a wrapper function for your explainer, we also
                    advise the same here regarding disagreement
                    between data format. 
            return: type: None.
        """

        self.data         = data
        self.model_fn     = model_fn
        self.explainer_fn = explainer_fn

    def _read_from_json(self, scores_path, preds_path,  num_classes): 
        
        """
        reads explanations from raw json file.
            params:
                scores_path: type: str:
                    the path to the file to read explanations from.
                preds_path: type: str:
                    the path to the file holding the labels. 
                num_classes: type: int:
                    the number of possible classes.
            return: None.
        """

        serialized_instances = list()
        serialized_preds     = list()
        temp_instances    = list()
        temp_explanations = list()

        # open and read from json file.
        with open(scores_path, 'rb') as f:
            sals = json.load(f)
            serialized_instances = sals[JSON_HEADER_SCORES]

        # read in raw preds from json file.
        with open(preds_path, 'rb') as f:
            preds = json.load(f)
            serialized_preds = preds["label"]

        # for each instance, load in it's data into a list of
        # dicts.
        for inst, pred in zip(serialized_instances, serialized_preds):
            temp_data = dict()
            temp_expl = dict()

            # read in the sentence, as list of (sub-)token's.
            temp_data['sentence'] =\
                ' '.join( datum['token'] for datum in inst)

            # record prediction.
            temp_data['label'] = pred 

            # read in the saliency scores for each class.
            for class_idx in range(num_classes):
                temp_expl[str(class_idx)] =\
                    {
                        datum['token'] : datum[str(class_idx)]
                        for datum in inst
                    }

            # record the gathered info in the dict to list. 
            temp_instances.append( temp_data )
            temp_explanations.append( [temp_expl] )

        self.data = temp_instances
        self.expl = temp_explanations 

    def _select_data_pairs(self, thresh=2000):

        """
        private helper to data_consistency, random samples sample
        pairs from the data for that metric. 
            params:
                thresh: type: int:
                    optional.
                        default: 2000
                    the max number of sample pairs to random
                    sample from the data.
            return: type: list(tuple(int, int)):
                    pairs of indices representing which instance
                    pairs where random sampled from the data. 
        """

        # gather how many samples to extract, limit extractions to
        # max size of data, prevents potential index out of
        # bounds errors. 
        upper_bound  =\
            len(self.data) if len(self.data) <= thresh else thresh

        index_pairs =\
            [
                (i, j)
                for i in range(upper_bound) 
                for j in range(i + 1, upper_bound)
            ]

        random.shuffle(index_pairs)

        return index_pairs[ : upper_bound ]

    def _get_activation_map(
            self, 
            model, 
            layers,
            idx,
            format_activations=None
        ):

        """
        returns an activation map of the model on the given batch.
            params:
                model: type: torch.nn.Module:
                    a torch model implemenation, with a forward(). the
                    entire model is need since we need to register
                    layers to record their activations.
                layers: type: set(str):
                    names of the layers to record the activations of.
                idx: type: int:
                    the batch to perform the forward pass on. provide
                    a dictionary mapping parameter names to their
                    appropriate arguments, as this function will
                    unpack the batch accordingly, via the syntax
                    model(**batch).
                format_activations: type: Callable:
                    -- optional --
                    default val: None.
                    specify a function with which to process recorded
                    activations into a an iterable. if not provided,
                    we assume activation are come in regular pt
                    tensors. cases where this might be need include
                    when then the model is an rnn and uses packed
                    sequences, at various layers. 
            return: type: pt.tensor: 
                    the activations for the provided batch.
        """

        handles     = list()
        activations = list()

        # iterate over the names of layer of the provided model.
        for name, module in model.named_modules():
            # currect layer is in layers we wish to target, we
            # register it during the forward pass to record it's
            # activations via a passed lambda func. 
            if name in layers:
                handles.append(
                    module.register_forward_hook(
                        lambda\
                            module, input_, output_:
                                activations.append(output_)
                    )
                )

        # time to record all activations during forward pass.
        # we call model_fn here, it's unclear whether our
        # hook above will have the desired affect on the model_fn
        # passed at construction. will torch.no_grad() have
        # an effect as well? documentation says that it's effects
        # only local threads. 
        with torch.no_grad(): self.model_fn( self.data[idx] )

        # let's undo the register, so the model doesn't record
        # activations hereafter.
        for handle in handles: handle.remove()

        # check whether activations need extra processing. if so,
        # then do it with user provided callable.
        if format_activations:
            activations =\
                list(
                    format_activations(activation[0])
                    for activation in activations 
                )

        # otherwise, assume no extra processing is needed.
        else:
            activations =\
                list(
                    activation[0] for activation in activations
                )

        # format all activation for this batch into a single
        # tensor, concatenate along the colspace.
        return\
            tuple(
                activation.reshape(-1).to('cpu')
                for activation in activations
            )

    def activation_similarity(self, model, layers, inst_x, inst_y):

        """
        returns similarity of activations maps of a pair of
        data samples. similarity is defined as the mean absolute
        difference between the activations. mean is applied at
        each layers, which results in l different scores, where
        l is the number of layers, and a mean is further taken
        from that.
            params:
                model: type: pt.model:
                    pytorch model take has a named_modules()
                    attribute.
                layers: type: iterable(str):
                    the names of the layers to use for computing
                    activation similarity.
                inst_x: type: int:
                    the idx of the first sample in the pair.
                inst_y: type: int:
                    the idx of the second sample in the pair. 
            return: type: float.
        """

        acts_x = self._get_activation_map(model, layers, inst_x)
        acts_y = self._get_activation_map(model, layers, inst_y)

        return\
            torch.mean(
                torch.cat(
                    tuple(
                        torch.mean(
                            torch.abs(
                                act_x - act_y
                            )
                        ).unsqueeze(0)
                        for act_x, act_y in zip(acts_x, acts_y)
                    ),
                    dim=-1
                )
            ).item()

    def explanation_similarity(self, inst_x, inst_y):

        """
        returns similarity of explanations of a pair of data samples.
        similarity here is defined as the mean absolute difference
        between the saliency maps. the measure accounts for
        the distribution, not the vocabulary particular to the
        instance, which has its drawbacks and "drawforths."
        the mean is taken from the difference between the saliencies
        of each class, which results in n numbers for n classes,
        and then a mean is further taken from that.
            params:
                inst_x: type: int:
                    the idx of the first sample in the pair.
                inst_y: type: int:
                    the idx of the second sample in the pair.
            return: type: float.
        """

        explain_x = self.expl[inst_x]
        explain_y = self.expl[inst_y]

        differences = list()

        # extract the absolute difference between the explanations
        # for each sample, for each class. normalizing first.
        # similarity here is based on distribution, not vocabulary.
        for cls_x, cls_y in zip(explain_x, explain_y):
            dist_x =\
                torch.tensor( 
                    sorted(
                        list(
                            sal
                            for label, saliencies in cls_x.items()
                            for token, sal in saliencies.items()
                        ),
                        reverse=True
                    )
                )

            dist_y =\
                torch.tensor(
                    sorted(
                        list(
                            sal
                            for label, saliencies in cls_y.items() 
                            for token, sal in saliencies.items()
                        ),
                        reverse=True
                    )
                )

            max_len = min( len( dist_x ), len( dist_y ) )
            dist_x  = dist_x[ : max_len ]
            dist_y  = dist_y[ : max_len ]

            differences.append(
                torch.mean(
                    torch.abs(
                        dist_x - dist_y
                    )
                ).reshape(-1)
            )

        return\
            torch.mean(
                torch.cat(
                    differences,
                    dim=-1
                )
            ).item()

    def _make_plot(self, activations, explanations):

        """
        private helper to data_consistency() metric.
            params:
                activations: type: numpy.array(float):
                    the activation similarity scores.
                explanations: type: numpy.array(float):
                    the explanation similarity scores.
            return: type: None.
        """


        m, b = np.polyfit( activations, explanations, 1 )

        plt.scatter( activations, explanations )
        plt.plot( activations , m * activations + b , color='orange' )
        plt.xlabel( 'activation similarity' )
        plt.ylabel( 'explanation similarity' )
        plt.savefig('./dc_output_plot')

    def data_consistency(
            self,
            layers,
            scores_path,
            preds_path, 
            num_classes,
            thresh=2000,
            model=None,
            re_format=None
        ):

        """
        returns a measure of the explainers approximate consistency
        at a sub-corpus level.
            params:
                layers: type: set(str):
                    provide the names of the layers we should record
                    the activations off. the pt instance methods for
                    pt models
                        pt.model.name_modules()
                    should be useful for this. these will be used
                    to compute similarity between the activations
                    between data sample pairs. 
                thresh: type: int:
                    optional.
                        default: 2000
                    provide the max number of dataset instance
                    pairs to be considered for this metric. 
                model: type: pt.module:
                    optional.
                        default: None
                    if the model_fn passed at construction isn't the
                    pytorch model itself, then provide at, since
                    this metric must forward hook 
            return: type: dict(str -> float):
                    the spearman's rank coefficient computed over
                    activation and explainations similarities
                    between selected data sample pairs. 
        """

        self._read_from_json(
            scores_path,
            preds_path,
            num_classes
        )

        activations  = list()
        explanations = list()

        for inst_1, inst_2 in self._select_data_pairs(thresh):
            act_score =\
                self.activation_similarity(
                    model,
                    layers,
                    inst_1,
                    inst_2
                )

            exs_score =\
                self.explanation_similarity(
                    inst_1,
                    inst_2
                )

            activations.append( act_score )
            explanations.append( exs_score )

        for i in range(10):
            activations.append( activations[0] )
            explanations.append( explanations[0] )

        scaler       = MinMaxScaler()
        activations  = np.array(activations).reshape(-1, 1)
        explanations = np.array(explanations).reshape(-1, 1)

        activations =\
            scaler.fit_transform( activations ).reshape(-1)
        explanations =\
            scaler.fit_transform( explanations ).reshape(-1)

        self._make_plot( activations, explanations )

# end file -----------------------------------------------------------
