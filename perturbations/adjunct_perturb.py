# begin script -------------------------------------------------------

""" perturbations.py
script for extracting adjuncts from textual data through use of
a dependecy parser.
"""

# imports ------------------------------------------------------------

import spacy

# variables ----------------------------------------------------------

# use spacy's dependecy parser.
nlp = spacy.load('en_core_web_sm')

# the adjuncts to target by default.
default_adjunct_labels =\
    {
        "advmod",
        "amod",
        "advcl",
        "acl",
        "appos",
        "nounmod",
        "npmod",
        "nummod",
        "poss",
        "prep",
        "quantmod",
        "relcl"
    }

# func def -----------------------------------------------------------

def remove_adjuncts(data, adjunct_labels=None): 

    """
    method for extracting adjuncts from an interable of sentences.
    sentences are expected to be of type string.
        params:
            iterable: type: iter(str):
                an iterbale of strings, each string being a
                sentence.
            adjunct_labels: type: set(str):
                optional; default val: None.
                specify the adjuncts of interest, those to
                remove from the original sentence. if not provided
                all adjuncts are removed by default.
        return: type: list(str):
                the sentences with their adjuncts removed.
    """

    # for recording derived data.
    new_data = list()

    # check whether particular labels where provided, to
    # to target specific adjuncts in removal.
    if not adjunct_labels:
        adjunct_labels = default_adjunct_labels

    # traverse the data and remove adjuncts in each.
    for sent in data:
        parse    = nlp(sent)
        new_sent = list()
        idxs     = set()

        # locate all adjunctive phrases.
        for tok in parse:
            # check whether current tokens was labels as an
            # adjunct we want to target.
            if tok.dep_ in adjunct_labels:
                # extract indices of modifier and it's dependents,
                # so we know where to delete later on.
                idxs.update(
                    t.i for t in tok.subtree
                )

        # remove the detected adjuntival phrases,
        # updating the new sent.
        new_data.append(
            ' '.join(
                tok.text
                for i, tok in enumerate(parse) if i not in idxs
            )
        )

    return new_data

# end script ---------------------------------------------------------
