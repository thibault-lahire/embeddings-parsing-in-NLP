import string
import random
from operator import itemgetter
import re
import get as gt


def delete(w, loc):
    """ 
        Deletes the character in the word w at the position loc
    """
    return w[:loc] + w[loc + 1:]


def insert(w, c, loc):
    """ 
        Inserts the character c in the word w at the position loc
    """
    return w[:loc] + c + w[loc:]


def substitute(w, c, loc):
    """ 
        Subtitutes the character c in the word w at the position loc
    """
    return w[:loc] + c + w[loc + 1:]


def generate_candidates(w, charset=string.ascii_lowercase + "àâôéèëêïîçùœ"):
    """
        Generates possible candidates at distance 1 of the unknown word w

    Args:
        w: word not recognized
        charset: alphabet

    Returns:
        list of possible candidates
    """
    
    possibilities = []
    for i, _ in enumerate(w):
        possibilities += [delete(w, i)]
        for char in charset:
            possibilities += [insert(w, char, i)]
            possibilities += [substitute(w, char, i)]
    return possibilities
    


def case_normalizer(word, dictionary):
     """ In case the word is not available in the vocabulary,
     we can try multiple case normalizing procedure.
     We consider the best substitute to be the one with the lowest index,
     which is equivalent to the most frequent alternative."""
     w = word
     lower = (dictionary.get(w.lower(), 1e12), w.lower())
     upper = (dictionary.get(w.upper(), 1e12), w.upper())
     title = (dictionary.get(w.title(), 1e12), w.title())
     results = [lower, upper, title]
     results.sort()
     index, w = results[0]
     if index != 1e12:
         return w
     return word


def normalize(word, word_id, DIGITS=re.compile("[0-9]", re.UNICODE)):
    """ Find the closest alternative in case the word is OOV."""
    if not word in word_id:
        word = DIGITS.sub("#", word)
    if not word in word_id:
        word = case_normalizer(word, word_id)

    if not word in word_id:
        return None
    return word


def l2_nearest(embeddings, word_index, m=5):
    """Sorts words according to their Euclidean distance.
       To use cosine distance, embeddings has to be normalized so that their l2 norm is 1."""

    e = embeddings[word_index]
    distances = (((embeddings - e) ** 2).sum(axis=1) ** 0.5)
    sorted_distances = sorted(enumerate(distances), key=itemgetter(1))
    return zip(*sorted_distances[:m])


def knn(word, embeddings, word_id, id_word):
    word = normalize(word, word_id)
    if not word:
        return [id_word[0]]
    word_index = word_id[word]
    indices, distances = l2_nearest(embeddings, word_index)
    neighbors = [id_word[idx] for idx in indices]
    return neighbors





def get_neighbor(word, vocabulary, embeddings):
    """ 
        Gets most likely neighbor for the input word

    Args:
        word (str): word for which to find neighbors
        vocabulary (list): vocabulary available
        embeddings (list): embeddings associated to the vocabulary available

    Returns
        (str) the most likely word
    """
    vocabulary = set(vocabulary)
    word2idx = {w: i for (i, w) in enumerate(vocabulary)}
    idx2word = dict(enumerate(vocabulary))
    
    # get embedding neighbors
    embedding_candidates = set(knn(word, embeddings, word2idx, idx2word))

    # get levenshtein neighbors
    candidates = set(generate_candidates(word))
    levenshtein_candidates = candidates.intersection(vocabulary)
    
    # get neighbor
    for embedding_candidate in embedding_candidates:
        if embedding_candidate in levenshtein_candidates:
            return embedding_candidate
    if levenshtein_candidates:
        return random.choice(list(levenshtein_candidates))
    else:
        return list(embedding_candidates)[0]



if __name__=='__main__':
    vocabulary, embeddings = gt.compute_polyglot_words_embeddings()
    word_test = 'constituTion'
    res = get_neighbor(word_test, vocabulary, embeddings)
    print('The unknown word "{}" has been transformed into "{}"'.format(word_test, res))
    print()
    word_test = 'anarchite'
    res = get_neighbor(word_test, vocabulary, embeddings)
    print('The unknown word "{}" has been transformed into "{}"'.format(word_test, res))
    print()
    word_test = 'voitufre'
    res = get_neighbor(word_test, vocabulary, embeddings)
    print('The unknown word "{}" has been transformed into "{}"'.format(word_test, res))
    print()
    word_test = 'voistufre'
    res = get_neighbor(word_test, vocabulary, embeddings)
    print('The unknown word "{}" has been transformed into "{}"'.format(word_test, res))    
    


