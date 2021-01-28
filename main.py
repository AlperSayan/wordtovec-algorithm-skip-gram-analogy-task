import WordToVec.WordToVecParameter as WvP
import WordToVec.NeuralNetwork as NN
import Corpus.Corpus as Corpus
import pickle
import numpy as np
import random
import time

from MorphologicalAnalysis.FsmMorphologicalAnalyzer import FsmMorphologicalAnalyzer

from MorphologicalDisambiguation.DisambiguatedWord import DisambiguatedWord
from MorphologicalDisambiguation.DisambiguationCorpus import DisambiguationCorpus
from MorphologicalDisambiguation.HmmDisambiguation import HmmDisambiguation


# because of long training times i saved necessary objects as pickle objects

def main():


    fsm = FsmMorphologicalAnalyzer("./turkish_dictionary.txt", "./turkish_misspellings.txt",
                                   "./turkish_finite_state_machine.xml")

    dis_corpus = DisambiguationCorpus("./penntreebank.txt")
    algorithm = HmmDisambiguation()
    algorithm.train(dis_corpus)
    save_obj(algorithm, 'trained_algorithm')
    algorithm = load_obj('trained_algorithm')
    print('algorithm part done')

    corpus = Corpus.Corpus('./corpus.txt')
    preprocessed_corpus = preprocess_corpus(corpus, fsm, algorithm)
    save_obj(preprocessed_corpus, 'preprocessed_corpus')


    preprocessed_corpus = load_obj('preprocessed_corpus')
    print('preprocessing corpus done and corpus is loaded')

    wvp = WvP.WordToVecParameter()
    # for skipgram
    wvp.setCbow(False)
    neuralNetwork = NN.NeuralNetwork(preprocessed_corpus, wvp)
    dictionary = neuralNetwork.train()
    save_obj(dictionary, 'dict_object')
    dictionary = load_obj('dict_object')
    print('dict is loaded')

    words = get_words_as_list(dictionary)
    save_obj(words, 'words_list')
    words = load_obj('words_list')
    print('words loaded')

    word_counts = get_word_counts(preprocessed_corpus, words)
    save_obj(word_counts, 'word_counts_list')
    word_counts = load_obj('word_counts_list')
    print('counts are loaded')

    vects = get_vects_as_list(dictionary)
    save_obj(vects, 'vect_list')
    vects = load_obj('vect_list')
    print('vects are loaded')




    # select 10 random words and get closest and furthest words for the word
    # with random words it gives bad results
    random_words_for_cases, ind_cases = get_random_n_words(words, n=10)
    distances_table = get_distances_table(ind_cases, vects)
    print('distances table for the cases is prepared')

    closest_words_for_cases_table = get_closest_n_words(distances_table, ind_cases, words, n=3)
    print_case_part(random_words_for_cases, closest_words_for_cases_table)


    # Getting highest frequency indices and getting closest words and printing the result
    # for high frequency words there seems to be a correlation between words

    highest_freq_words, ind = get_highest_freq_words(word_counts, words, n=10)
    distances_table = get_distances_table(ind, vects)
    print('distances table for the highest frequency words is prepared')

    closest_words_table = get_closest_n_words(distances_table, ind, words, n=10)
    print_closest_words(closest_words_table, highest_freq_words)


def preprocess_corpus(corpus, fsm, algorithm):

    preprocessed_corpus = Corpus.Corpus()
    for i in range(corpus.sentenceCount()):
        try:
            sentence_analysis = fsm.robustMorphologicalAnalysis(corpus.getSentence(i))
            fsm_parses = algorithm.disambiguate(sentence_analysis)
            if fsm_parses is None:
                print('fsm parse was None')
                continue
            sentence = Corpus.Sentence()
            for j in range(corpus.getSentence(i).wordCount()):
                words = str(fsm_parses[j].getWord())
                word = words.split(' ')[0]
                word = Corpus.Word(word)
                sentence.addWord(word)

            preprocessed_corpus.addSentence(sentence)

        except:
            pass

    return preprocessed_corpus


def get_closest_n_words(distances_table, ind, words, n):

    closest_words_table = []
    for i in range(len(ind)):
        distances_from_the_word = distances_table[i]
        inds = np.argpartition(distances_from_the_word, range(n+1))[1:n+1]  # closest word is itself
        closest_words = []
        for j in range(len(inds)):
            closest_words.append(words[inds[j]])
        closest_words_table.append(closest_words)

    return closest_words_table


def get_distances_table(inds, vects):

    distances_table = np.zeros([len(inds), len(vects)])
    for i in range(len(distances_table)):
        item = vects[inds[i]]
        dists = np.zeros(len(vects))
        for j in range(len(vects)):
            compared_to = vects[j]
            dists[j] = np.linalg.norm(item - compared_to)

        distances_table[i] = dists

    return distances_table



def print_case_part(random_words_for_cases, closest_words_for_cases_table):

    for i in range(len(random_words_for_cases)):
        print('case ' + str(i) + ':')
        closest_words_table_ = closest_words_for_cases_table[i]
        print('closest words to: {' + random_words_for_cases[i] + '} with smallest value being first:', end=" {")
        for j in range(len(closest_words_table_)):
            print(closest_words_table_[j], end=" ")

        print(end="}")
        print()
        print()


def get_random_n_words(words, n):

    ind = []
    new_words = []
    for i in range(n):
        rd = random.randrange(0, len(words))
        new_words.append(words[rd])
        ind.append(rd)

    return new_words, ind


def get_vects_as_list(dictionary):

    vects = []
    for i in range(dictionary.size()):
        vects_vals = []
        for j in range(dictionary.words[i].getVector().size()):
            vects_vals.append(dictionary.words[i].getVector().getValue(j))
        vects.append(vects_vals)

    vects = np.array(vects)

    return vects


def get_words_as_list(dictionary):

    words = []
    for i in range(dictionary.size()):
        word = dictionary.words[i].getName()
        words.append(word)

    return words


def print_closest_words(closest_words_table, highest_freq_words):

    for i in range(len(closest_words_table)):
        closest_words_table_ = closest_words_table[i]
        print('closest words to: {' + highest_freq_words[i] + '} with smallest value being first:', end=" {")
        for j in range(len(closest_words_table_)):
            print(closest_words_table_[j], end=" ")

        print(end="}")
        print()






def get_highest_freq_words(word_counts, words, n):

    word_counts = punctiation_counts_0(words, word_counts)
    ind = np.argpartition(word_counts, -n)[-n:]
    highest_freq_words = []
    for i in range(n):
        highest_freq_words.append(words[ind[i]])

    return highest_freq_words, ind


def punctiation_counts_0(words, word_counts):

    index_dot = find_index_of_the_word(words, '.')
    index_tire = find_index_of_the_word(words, '-')
    index_semi = find_index_of_the_word(words, ',')
    index_indet = find_index_of_the_word(words, '"')
    index_indent = find_index_of_the_word(words, "'")
    index_lp = find_index_of_the_word(words, "(")
    index_rp = find_index_of_the_word(words, ")")
    index_scolon = find_index_of_the_word(words, ":")

    word_counts[index_dot] = 0
    word_counts[index_tire] = 0
    word_counts[index_semi] = 0
    word_counts[index_indet] = 0
    word_counts[index_indent] = 0
    word_counts[index_lp] = 0
    word_counts[index_rp] = 0
    word_counts[index_scolon] = 0



    return word_counts


def get_word_counts(corpus, words):

    word_counts = np.zeros(corpus.wordCount())
    start_time = time.time()

    for i in range(corpus.sentenceCount()):
        sentence = corpus.getSentence(i)
        for j in range(sentence.wordCount()):
            word = sentence.getWord(j).getName()
            index = find_index_of_the_word(words, word)
            word_counts[index] = word_counts[index] + 1
        if i % 1000 == 0:
            print(i)
            print("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()

    return word_counts


def find_index_of_the_word(words, word):
    index = -1
    for i in range(len(words)):
        if words[i] == word:
            index = i
            break

    return index


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)



if __name__ == '__main__':
    main()
