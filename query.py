#Author: Grace Gibson
#Date: 7/31/2018
#Main method for POSTGRESQL text analysis project

#import interface to connect to postgres
hostname = 'localhost'
username = 'postgres'
password = 'aw3s0me!'
database = 'notable'

import psycopg2
print "Using psycopg2..."
myConnection = psycopg2.connect(host=hostname, user=username, password=password, dbname=database )

import nltk
from nltk.tokenize import word_tokenize
#nltk.download('averaged_perceptron_tagger')
from POSTagger import transform_to_dataset
from POSTagger import train_classifier

#gaussian mixture model imports
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
#set up port to connect#
#s = socket.socket()
#host = 'localhost'
#port = 5432
#s.connect((host, port))
    
#s.recvfrom()
#s.close()


#run query on database
#converts customer note from tuple into list
def doQuery( conn ):
    done = []
    try:
        cur = conn.cursor()
        cur.execute("SELECT text FROM notes")
    
        for text in cur.fetchall():
            test = list(text)
            tokened = tokenizer(test)
            done.append(tokened)
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
         print(error)
    finally:
        conn.close()
    return done

tokens = doQuery(myConnection)

#tokenize data from query
def tokenizer(lst):
    tokens = []
    for x in lst:
        line = str(x)
        
        #gets rid of ERROR('ascii' codec can't decode byte 0xf0 in position 5: ordinal not in range(128))
        try:
            value = unicode(line, "ascii")
        except UnicodeError:
            value = unicode(line, "utf-8")
        else:
            pass
        token = nltk.word_tokenize(value)
        tokens.append(token)
    return tokens

#tags tokens with respective POS
#def posTag(tokens):
 #   tagged = []
  #  for x in tokens:
   #     tag = nltk.pos_tag(x)
    #    tagged.append(tag)
    #return tagged

def testModel(tagged):
    cutoff = cutoff = int(.80 * len(tagged))
    train_sents = tagged[:cutoff]
    test_sents = tagged[cutoff:]
    X, y = transform_to_dataset(tagged)
    
    #Training the classifier
    train_classifier(test_sents)
    return test_sents


#GMM
#could not convert string to float: NNP
gmm = GaussianMixture(n_components=4).fit(testModel(tokens))
#labels = gmm.predict(X)
#plt.scatter(X[:,0], X[:1], c = labels, s=40, cmap = 'viridis')

#close port access
myConnection.close()
