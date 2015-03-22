import pydsm
import pydsm.similarity
from scipy.stats import spearmanr
from pkg_resources import resource_stream
import pickle
import os

def synonym_test(matrix, synonym_test, sim_func=pydsm.similarity.cos):
    """
    Evaluate DSM using a synonym test.

    :param matrix: A DSM matrix.
    :param synonym_test: A dictionary where the key is the word in focus, 
                           and the value is a list of possible word choices. 
                           The first word in the dict is the correct choice.
    :param sim_func: The similarity function to use for evaluation.
    :return: Accuracy of synonym test.
    """
    correct = []
    incorrect = []
    unknown_focus_words = []
    unknown_synonyms = []

    for focus_word, candidates in synonym_test.items():
        if focus_word not in matrix.word2row:
            unknown_focus_words.append(focus_word)
            continue

        known_words = [w for w in candidates if w in matrix.word2row]
        
        unknown_words = [w for w in candidates if w not in matrix.word2row]
        if candidates[0] in unknown_words:
            unknown_synonyms.append(focus_word)
            continue

        word_sims = sim_func(matrix[focus_word], matrix[known_words], assure_consistency=False).transpose().sort(ascending=False)
        if word_sims.row2word[0] == candidates[0]:
            correct.append(focus_word)
        else:
            incorrect.append(focus_word)
    

    accuracy = len(correct) / len(synonym_test)
    print("Evaluation report")
    print("Accuracy: {}".format(accuracy))
    print("Number of words: {}".format(len(synonym_test)))
    print("Correct words: {}".format(correct))
    print("Incorrect words: {}".format(incorrect))
    print("Unknown words: {}".format(unknown_focus_words))
    print("Unknown correct synonym: {}".format(unknown_synonyms))

    return accuracy


def simlex(matrix, sim_func=pydsm.similarity.cos):
    """
    Evaluate DSM using simlex-999 evaluation test [1].
    
    :param matrix: A DSM matrix.
    :param sim_func: The similarity function to use for evaluation.
    
    :return: Spearman correlation coefficient.

    [1] SimLex-999: Evaluating Semantic Models with (Genuine) Similarity Estimation. 2014. 
        Felix Hill, Roi Reichart and Anna Korhonen. Preprint pubslished on arXiv. arXiv:1408.3456
    """
    wordpair_sims = pickle.load(resource_stream(__name__, os.path.join('resources', 'simlex.pickle')))
    simlex_vals = []
    sim_vals = []
    skipped = []
    for (w1, w2), value in wordpair_sims.items():
        if w1 not in matrix.word2row or w2 not in matrix.word2row:
            skipped.append((w1, w2))
            continue

        sim_vals.append(sim_func(matrix[w1], matrix[w2])[0,0])
        simlex_vals.append(value)

    spearman = spearmanr(simlex_vals, sim_vals)
    print("Evaluation report")
    print("Spearman correlation: {}".format(spearman[0]))
    print("P-value: {}".format(spearman[1]))
    print("Skipped the following word pairs: {}".format(skipped ))
    return spearman[0]


def toefl(matrix, sim_func=pydsm.similarity.cos):
    """
    Evaluate DSM using TOEFL synonym test [1].

    :param matrix: A DSM matrix.
    :param sim_func: The similarity function to use for evaluation.

    :return: Accuracy of TOEFL test.

    [1] http://aclweb.org/aclwiki/index.php?title=TOEFL_Synonym_Questions_%28State_of_the_art%29
    """
    synonym_dict = pickle.load(resource_stream(__name__, os.path.join('resources', 'toefl.pickle')))
    return synonym_test(matrix, synonym_dict, sim_func=sim_func)

