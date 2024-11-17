### This demonstrates how the true answers for a given query are similar
### to that query. Change qInd, in range [0, 999], to test different 
### queries.

import numpy as np

vocab = np.load('./data/20news/vocab.npy')
print("vocab:", vocab.shape)

words = np.load('./data/20news/words.npy', allow_pickle=True)
print("words: ", words.shape)

dataset = np.load('./data/20news/dataset.npy', allow_pickle=True, encoding='bytes')
print("dataset:", dataset.shape)

queries = np.load('./data/20news/queries.npy', allow_pickle=True)
print("queries:", queries.shape)


answers = np.load('./data/20news/answers.npy')
print("answers:", answers.shape)


qInd = 444 # CHANGE THIS VALUE
post = queries[qInd]

print("query", qInd)
for token in post:
    print(words[token][1], end = " ")
print()


dataset_answ = answers[0][qInd]
response = dataset[dataset_answ]

print("datapoint", dataset_answ)
for token in response:
    print(words[token][1], end = " ")
print()