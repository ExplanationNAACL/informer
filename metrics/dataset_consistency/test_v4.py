#---------------------------------------------------------------------

""" test_v4.py
script testing version 2 of data consistency.
"""

#---------------------------------------------------------------------

from   data_loader      import get_collate_fn, get_dataset
from   transformers     import BertTokenizer, BertConfig,\
                               BertForSequenceClassification
import torch
from   data_consistency_v2\
                        import * 
from   functools        import partial
from   tqdm             import tqdm
from   lime.lime_text   import LimeTextExplainer
import numpy as np

# data processing ----------------------------------------------------

device = 'cuda:0'

# load data. 
data =\
    get_dataset(
        './data/tweet/',
        'tweet'
    )

# re-format data in the way informer class expects. 
data =\
    list(
        {'sentence': datum[0], 'label': datum[1]}
        for datum in data
    )

tokenizer = BertTokenizer.from_pretrained( 'bert-base-uncased' )
pad_token = tokenizer.vocab['[PAD]']

# get collate function for data set. 
collate_fn =\
    partial(
        get_collate_fn('tweet', 'trans'),
        tokenizer=tokenizer,
        device=device,
        return_attention_masks=True,
        pad_to_max_length=True
    )
    
# model processing and loading ---------------------------------------

# load in model from checkpoint, and model tokenizer.
checkpoint  = torch.load( './data/models/bert.pt' )

bert_config =\
    BertConfig.from_pretrained(
        'bert-base-uncased',
        num_labels=3
    )

model =\
    BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        config=bert_config
    ).to(device)

model.load_state_dict(checkpoint['model'])

# explainer loading and processing -----------------------------------

# init explainer method. 
explainer = LimeTextExplainer()

# define layers the activations of which we want to record.
layers =\
    {
        'bert.encoder.layer.0',
        'bert.encoder.layer.1', 
        'bert.encoder.layer.2',
        'bert.encoder.layer.3',
        'bert.encoder.layer.4',
        'bert.encoder.layer.5',
        'bert.encoder.layer.6',
        'bert.encoder.layer.7',
        'bert.encoder.layer.8',
        'bert.encoder.layer.9',
        'bert.encoder.layer.10',
        'bert.encoder.layer.11',
        'classifier'
    }

# pre-forward data processing ----------------------------------------

def model_format(batch):

    batch = collate_fn( [batch] )
    return\
        {
            'input_ids'      : batch[0],
            'attention_mask' : batch[1],
            'labels'         : batch[2]
        }

def explainer_format(datum):

    formatted =\
        ' '.join(
            str(i)
            for i in datum['input_ids'].squeeze().tolist()
        )

    return formatted

# model_fn def -------------------------------------------------------

def model_fn(input_text):

    batch  = tuple( val for _, val in input_text.items() )
    batch  = model_format( batch )
    output = model(**batch)

    return output['logits'].detach().cpu().numpy()
    
# model_fn for explainer def -----------------------------------------

def model_wrapper(input_strs, device=device):
    batch_size = 16
    batches    = set()
    outputs    = list()

    # list of strs, each of which is a str of input_ids of generated
    # perturbations inputs in the neighborhood of the original.
    # size of this list is d.
    input_strs =\
        list(
            [int(i) for i in _str.split()]
            for _str in input_strs
        )

    # collect batches.
    for i in range(0, len(input_strs), batch_size):
        # get batch, and max len input in batch.
        batch = input_strs[ i : i + batch_size ]
        max_len = max(len(d) for d in batch)
        
        # convert batch of type list to tensor.
        # shape: [batch_size, max_len_input]
        batches.add(
            torch.tensor(
                list(
                    d + ( [ pad_token ] * (max_len - len(d)) )
                    for d in batch
                ),
                dtype=torch.long
            ).to(device)
        )

    # store all outputs in list.
    for b in batches:
        outputs.append(model(b)['logits'].detach().cpu())

    # concat all logit outputs into a single tensor, along rowspace.
    # convert to numpy array before return. 
    return torch.cat( outputs, dim=-2 ).numpy()

# explainer_fn -------------------------------------------------------

def explainer_fn(input_str, labels=[0, 1, 2]):

    batch = tuple( val for _, val in input_str.items() )
    batch = model_format(batch)
    input_str = explainer_format(batch)

    explanations =\
        explainer.explain_instance(
            input_str,
            model_wrapper,
            num_features=len(batch['input_ids'].squeeze().tolist()),
            labels=labels
        )

    explainations =\
        list( 
            {
                label:
                {
                    tok : sal
                    for tok, sal in explanations.as_list(label=label)
                }
            }
            for label in labels
        )

    return explainations

# procedure ----------------------------------------------------------

# the code here generates and serializes explanations.
# comment out if you don't want this.
info = Informers( data, model_fn, explainer_fn )
scores = list()
import json

for point in tqdm(range(len(info.data[ : 1000]))):
    scores.append(
        {
            'index'      : point,
            'sentence'   : info.data[point]['sentence'],
            'label'      : info.data[point]['label'],
            'explanation': explainer_fn(info.data[point])
        }
    )

with open('data_tweet_sentiment.json', 'w') as outfile:
    json.dump(scores, outfile, indent=4)

# the code here will samples pairs of the data and generate
# activation similarity scores.
for inst_1, inst_2 in info._select_data_pairs():
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

# end file -----------------------------------------------------------
