import json
import tflearn
import tensorflow as tf
import nltk
import numpy as np
import pickle
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
words=[]
docs=[]
labels=[]
docs_x=[]
docs_y=[]
file=open('intents_n.json','rb')
data=json.load(file)
try:
    with open("data.pickle","rb") as f:
        words,labels,training,output=pickle.load(f)
except:
    for d in data['intents']:
        for p in d['patterns']:
            wrds=nltk.tokenize.word_tokenize(p)
            words.extend(wrds)
            docs.append(p)
            docs_x.append(wrds)
            docs_y.append(d["tag"])
        if d['tag'] not in labels:
            labels.append(d['tag'])
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    print(words)
    labels = sorted(labels)
    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        print(x,doc)

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = np.array(training)
    output = np.array(output)
    with open("data.pickle","wb") as f:
        pickle.dump((words,labels,training,output),f)
tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')
model.load('model.tflearn')

    
