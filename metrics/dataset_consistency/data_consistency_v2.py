# begin script -------------------------------------------------------

""" data_consistency_v2.py
in this version, we code with the understainding that our methods
will be contained as instance methods of class Informers, where
each object, at construction, will pass along the dataset, model_fn,
and explainer_fn.
"""

# imports ------------------------------------------------------------

from   scipy.stats           import spearmanr
from   sklearn.preprocessing import MinMaxScaler
from   tqdm                  import tqdm
from   itertools             import combinations
import random
import torch

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

        # extract all instances with the same label, up to the
        # specified threshold.
        pairs_same_label =\
            [
                (i, j)
                for i in range(upper_bound)
                for j in range( i + 1 , upper_bound)
                if self.data[i]['label'] == self.data[j]['label']
            ]

        # collect all instance pairs with different labels, up
        # to the specified threshold.
        pairs_diff_label =\
            [
                (i, j)
                for i in range(upper_bound)
                for j in range( i + 1, upper_bound)
                if self.data[i]['label'] != self.data[j]['label']
            ]

        # now random sample half of those collected in each
        # same and different label spaces.
        pairs_same_label =\
            random.sample(
                pairs_same_label,
                len(pairs_same_label)
            )

        pairs_diff_label =\
            random.sample(
                pairs_diff_label,
                len(pairs_diff_label)
            )

        # take all collected samples and shuffle them, and observe
        # the upper bound provided by the user.
        union_pairs = pairs_same_label + pairs_diff_label
        random.shuffle(union_pairs)

        return union_pairs[ : upper_bound ]

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
                activation.reshape(1, -1).squeeze().tolist()#to('cpu')
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

        explain_x = self.explainer_fn(self.data[inst_x])
        explain_y = self.explainer_fn(self.data[inst_y])

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
                ).unsqueeze(0)
            )

        return\
            torch.mean(
                torch.cat(
                    differences,
                    dim=-1
                )
            ).item()

    def data_consistency(self, layers, thresh=2000,
                             model=None, re_format=None):

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

        activations  = list()
        explanations = list()

        # extract and interate over selected sample pairs.
        with open('./scores.txt', 'w') as score:
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

                score.write(str(inst_1)+' '+str(inst_2))
                score.write('\n')
                score.write('activation: '+str(act_score))
                score.write('\n')
                score.write('explanation: '+str(exs_score))
                score.write('\n')
                score.write('\n')

# end file -----------------------------------------------------------
