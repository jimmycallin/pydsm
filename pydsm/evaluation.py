import pydsm

def synonym_test(matrix, focus_word, word_list, correct):
    known_words = [w for w in word_list if w in matrix.row2word]
    unknown_words = [w for w in word_list if w not in matrix.row2word]
    word_sims = pydsm.similarity.cos(matrix[known_words], matrix[focus_word]).sort(ascending=False)
    
    print("Focus word: {}".format(focus_word))
    print("Word list: {}".format(word_list))
    print("Correct word: {}".format(correct))
    print("Guessed word: {}".format(word_sims.row2word[0]))
    print("Similarity scores: \n{}".format(word_sims))