
import json
import os
import numpy as np
import torch
from functools import reduce
import itertools
import time


import aoareader.Constants
from aoareader.Dict import Dict as Vocabulary

from nltk.tokenize import word_tokenize

from sys import argv

data_path = 'data/cnn/'
data_filenames = {
        'train': 'train.txt',
        'valid': 'dev.txt',
        'test': 'test.txt'
        }
vocab_file = os.path.join(data_path, 'vocab.json')
dict_file = os.path.join(data_path, 'dict.pt')

def tokenize(sentence):
    return [s.strip().lower() for s in word_tokenize(sentence) if s.strip()]


def parse_stories(lines, with_answer=True):
    stories = []
    story = []
    for line in lines:
        line = line.strip()
        if not line:
            story = []
        else:
            _, line = line.split(' ', 1)
            if line:
                if '\t' in line:  # query line
                    answer = ''
                    if with_answer:
                        q, answer, _, candidates = line.split('\t')
                        answer = answer.lower()
                    else:
                        q, _, candidates = line.split('\t')
                    q = tokenize(q)

                    # use the first 10
                    candidates = [cand.lower() for cand in candidates.split('|')[:10]]
                    stories.append((story, q, answer, candidates))
                else:
                    story.append(tokenize(line))
    return stories


def get_stories(story_lines, with_answer=True):
    stories = parse_stories(story_lines, with_answer=with_answer)
    flatten = lambda story: reduce(lambda x, y: x + y, story)
    stories = [(flatten(story), q, a, candidates) for story, q, a, candidates in stories]
    return stories
def get_stories_cnn(in_file):
    documents = []
    questions = []
    answers = []
    candidates=[]
    num_examples = 0
    with open(in_file, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            question = line.strip().lower()
            answer = f.readline().strip()
            document = f.readline().strip().lower()

            q_words = question.split(' ')
            d_words = document.split(' ')
            assert answer in d_words

            candidate = set()
            for word in d_words + q_words:
                if (word.startswith('@entity')):
                    candidate.add(word.lower())
            assert answer in candidate
            question=question.split(' ')
            document = document.split(' ')
            questions.append(question)
            answers.append(answer)
            documents.append(document)
            candidates.append(list(candidate))
            num_examples += 1

            f.readline()
            # if (max_example is not None) and (num_examples >= max_example):
            #     break

    print('#Examples: %d' % len(documents))
    return ([(documents[i], questions[i], answers[i], candidates[i]) for i in range(len(documents))])


def vectorize_stories(stories, vocab : Vocabulary):
    X = []
    Q = []
    C = []
    A = []

    for s, q, a, c in stories:
        x = vocab.convert2idx(s)
        xq = vocab.convert2idx(q)
        xc = vocab.convert2idx(c)
        X.append(x)
        Q.append(xq)
        C.append(xc)
        A.append(vocab.getIdx(a))

    X = X
    Q = Q
    C = C
    A = torch.LongTensor(A)
    return X, Q, A, C


def build_dict(stories):
    if os.path.isfile(vocab_file):
        with open(vocab_file, "r") as vf:
            word2idx = json.load(vf)
    else:

        vocab = sorted(set(itertools.chain(*(story + q + [answer] + candidates
                                             for story, q, answer, candidates in stories))))
        vocab_size = len(vocab) + 2     # pad, unk
        print('Vocab size:', vocab_size)
        word2idx = dict((w, i + 2) for i,w in enumerate(vocab))
        word2idx[aoareader.Constants.UNK_WORD] = 1
        word2idx[aoareader.Constants.PAD_WORD] = 0

        with open(vocab_file, "w") as vf:
            json.dump(word2idx, vf)

    return Vocabulary(word2idx)


def main():

    print('Preparing process dataset ...')
    train_filename = os.path.join(data_path, data_filenames['train'])
    valid_filename = os.path.join(data_path, data_filenames['valid'])
    test_filename = os.path.join(data_path, data_filenames['test'])

    train_stories = get_stories_cnn("data/cnn/train.txt")
    valid_stories = get_stories_cnn("data/cnn/dev.txt")
    test_stories = get_stories_cnn("data/cnn/test.txt")

    print('Preparing build dictionary ...')
    vocab_dict = build_dict(train_stories + valid_stories + test_stories)

    print('Preparing training, validation, testing ...')
    train = {}
    valid = {}
    test = {}

    train_data=vectorize_stories(train_stories,vocab_dict)
    print("finish making train_data")
    valid_data = vectorize_stories(valid_stories, vocab_dict)
    print("finish making valid_data")
    test_data = vectorize_stories(test_stories, vocab_dict)
    print("finish making test_data")
    train['documents'], train['querys'], train['answers'], train['candidates'] = train_data
    valid['documents'], valid['querys'], valid['answers'], valid['candidates'] = valid_data
    test['documents'], test['querys'], test['answers'], test['candidates'] = test_data


    print('Saving data to \'' + data_path + '\'...')
    torch.save(vocab_dict, dict_file)
    torch.save(train, train_filename + '.pt')
    torch.save(valid, valid_filename + '.pt')
    torch.save(test, test_filename + '.pt')

if __name__ == '__main__':
    main()
