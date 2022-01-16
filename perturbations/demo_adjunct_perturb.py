# begin script -------------------------------------------------------

""" demo.py
file testing adjunct removal in adjunct_pertrub.py.
"""

# imports ------------------------------------------------------------

from perturbations              import *
from lit_nlp.examples.datasets  import glue

# test ---------------------------------------------------------------

data =\
    [
        sample['sentence']
        for sample in glue.SST2Data('validation').examples
    ]

for sent, new_sent in zip(data, remove_adjuncts(data)):
    print(); print(sent); print(new_sent); print()

# end script ---------------------------------------------------------
