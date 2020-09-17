import tflearn
import json
import numpy as np
import tensorflow as tf 
import pickle
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer=LancasterStemmer()
data=json.load(open('intents_n.json'))
words,labels,training,output=pickle.load(open('data.pickle','rb'))
tf.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)
model.load('model.tflearn')
def bw(s,words):
    bag=[0 for _ in range(len(words))]
    s_w=nltk.word_tokenize(s)
    s_w=[stemmer.stem(word.lower()) for word in s_w]
    for s in s_w:
        for i,w in enumerate(words):
            if w==s:
                bag[i]=1
    return np.array(bag)
def chat():
    print("Hello User I am your Assistant")
    while True:
        inp=input('You: ')
        if inp.lower()=='quit':
            break
        result=model.predict([bw(inp,words)])
        r_i=np.argmax(result)
        tag=labels[r_i]
        for t in data["intents"]:
            if t['tag']==tag:
                res=t['responses']
        print(random.choice(res))
chat()