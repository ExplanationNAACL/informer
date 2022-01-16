from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import gensim.downloader
import random
import nltk
from numpy import dot
from numpy.linalg import norm
from nltk.corpus import wordnet
import pickle
import numpy as np
import sys

NUM_VARIATIONS = 0

#with open('glove_embeddings.pickle_2', 'wb') as f:
#    model = gensim.downloader.load('glove-twitter-100')
#    pickle.dump(model, f)

#with open('glove_embeddings.pickle_2', 'rb') as f:
#    model = pickle.load(f)
model = gensim.downloader.load('glove-twitter-100')
# Only certain parts of speech should be perturbed
replaceable_pos = ["JJ", "JJR", "JJS", "RB", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]


# Cosine similarity
def cos_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))


# Create the cross product list of perturbations
def combinations(items, predictor, focus_cls, max_needed):
    #global NUM_VARIATIONS
    if len(items) == 0:
        return [[]]
    curr_combos = combinations(items[1:], predictor, focus_cls, max_needed)
    #print(f"curr_combos is of len {len(curr_combos)} and looks like so: {curr_combos}")
    #print(f"Items = {items}")
    new_combos = []
    for item in items[0]:
        for combo in curr_combos:
            element = [item] + combo
            new_combos.append(element)
            instance = {"sentence": " ".join(element), "label": None}
            new_combos.append(element)
            #if np.argmax(predictor(instance)) == focus_cls and NUM_VARIATIONS < max_needed:
                #new_combos.append(element)
                #NUM_VARIATIONS += 1
            #elif NUM_VARIATIONS == max_needed:
            #    return new_combos
    #print(NUM_VARIATIONS)
    return new_combos


def perturb(sentence, depth, banned_words, tokenizer, predictor, focus_cls, max_needed):
    text = tokenizer(sentence)

    # Get parts of speech
    pos_tags = nltk.pos_tag(text)
    pos_dict = {}
    for item in pos_tags:
        pos_dict[item[0]] = item[1]

    syn_list = [[word] for word in text]

    for i, item in enumerate(syn_list):
        cur_word = item[0]
        if cur_word in banned_words:
            continue
        try:
            word_embedding = model[cur_word.lower()]
        except:
            word_embedding = None

        if pos_dict[cur_word] in replaceable_pos:
            possible_synonyms = []
            wordnet_syns = wordnet.synsets(cur_word)

            for syn in wordnet_syns:
                for l in syn.lemmas():
                    possible_synonyms.append(l.name())

            possible_synonym_vectors = []
            for syn in possible_synonyms:
                try:
                    possible_synonym_vectors.append(model[syn])
                except:
                    possible_synonym_vectors.append(None)

            # Some words do not have embeddings
            usable_synonyms = []
            for syn, vector in zip(possible_synonyms, possible_synonym_vectors):
                if vector is not None:
                    usable_synonyms.append(syn)

            # Sort by embedding similarity - these are static embeddings - dynamic would be extremely slow
            usable_synonyms.sort(key=lambda x: cos_sim(model[x], word_embedding))
            syn_list[i].extend(usable_synonyms[:depth])

    #print(syn_list)

    combos = combinations(syn_list, predictor, focus_cls, max_needed+1) # because the first combination consists of the original tokens
    outs = [" ".join(combo) for combo in combos][:10]#[:max_needed+1]
    print(f"Perturbed data: {outs}")
    return outs
