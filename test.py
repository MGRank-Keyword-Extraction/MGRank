from __future__ import unicode_literals, print_function
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from itertools import chain
import nltk
import glob
import string
import os
import re
import math
import numpy as np
from nltk import sent_tokenize
from nltk import RegexpParser
from nltk import pos_tag
import networkx as nx
from scipy import spatial
from nltk.tokenize import sent_tokenize
import numpy
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
import spacy
nlp = spacy.load("en_core_web_sm")
from spacy.lang.en import English # updated
from nltk.tokenize import word_tokenize 


#https://skeptric.com/making-words-singular/
SINGULAR_UNINFLECTED = ['gas', 'asbestos', 'womens', 'childrens', 'sales', 'physics']

SINGULAR_SUFFIX = [
    ('people', 'person'),
    ('men', 'man'),
    ('wives', 'wife'),
    ('menus', 'menu'),
    ('us', 'us'),
    ('ss', 'ss'),
    ('is', 'is'),
    ("'s", "'s"),
    ('ies', 'y'),
    ('ies', 'y'),
    ('es', 'e'),
    ('s', '')
]
  
def nounPhraseExt(text):

    text = text.lower()
    candidateKeyphrases = []
    candidateWords = [] 
    pattern = r"NP:{<JJ.*>?<NN.*>+}"   
    tokenized_text = nltk.tokenize.word_tokenize(text)
    postoks = nltk.tag.pos_tag(tokenized_text)
    parser = nltk.RegexpParser(pattern)
    chunks = parser.parse(postoks)

    for subtree in chunks.subtrees():
      if subtree.label()=='NP':

        adayPh = ' '.join((e[0] for e in list(subtree)))
             
        remove = string.punctuation
        remove = remove.replace("-", "")    
        adayPh.translate(str.maketrans('', '', string.punctuation))
        if "-" not in adayPh:
          adayPh = re.sub(r'[^\w]', ' ', adayPh)
        adayPh = adayPh.strip()
        adayPh = re.sub(' +', ' ', adayPh)

        if(len(adayPh.split())<4) and len(adayPh)>=2:
          candidateWords.extend(adayPh.split())
          candidateKeyphrases.append(adayPh)
        
    candidateKeyphrases = [i for i in candidateKeyphrases if i]  
    candidateKeyphrases = list(set(candidateKeyphrases))
    
    return candidateKeyphrases, candidateWords


def singularize_word(word):
    for ending in SINGULAR_UNINFLECTED:
        if word.lower().endswith(ending):
            return word
    for suffix, singular_suffix in SINGULAR_SUFFIX:
        if word.endswith(suffix):
            return word[:-len(suffix)] + singular_suffix
    return word

def spaceProblemSol(text):
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\t", " ", text)
    text = re.sub(r"  ", " ", text)
    
    return text


def positionWeight(candidateWords):
    pWDict = {}
    for wordL in candidateWords:
      index = candidateWords.index(wordL)
      pWDict[wordL] = 1/(index+1)
        
    return pWDict


def edgeWeightsForward(G,wordsResult):
  
  list1 = []
  list2 = []
  list3 = []

  for u,v,data in G.edges(data=True):
        list1.append(u)
        list1.append(v)
        list1.append(data['indiceF'])
        list1.append(data['weight'])
        list1.append(data['indiceB'])
        list2.append(list1)
        list1 = []
        list3.append(data['indiceF'])
  indice = []
  listEdgeWeight = []
  for i in range(0,len(set(list3))):
        indice = []
        tempList = []
        for l1 in list2:
        
            if l1[2] is i:
                indice.append(l1[3])
                tempList.append(l1)
        if len(indice) is 0:
            continue
        ekok = np.lcm.reduce(indice, dtype='int64')
        
        newList = [ekok * (1 / x) for x in indice]
        newList = [x/sum(newList) for x in newList]
   
        for t in range(0,len(tempList)):
            tempList[t].append(newList[t])
        
        listEdgeWeight.append(tempList)
  return listEdgeWeight   

def graphEdgeUpdation(ay1,G):

    G.remove_edges_from(list(G.edges()))
    for a1 in ay1:
        for i1 in range(0,len(a1)):
          G.add_edge(a1[i1][0],a1[i1][1],weight=0,indiceF=a1[i1][2],weightedF=a1[i1][5])
                                                
def forwardGain(G):

    dictRankForFwNode = {}
    for node1 in G.__iter__():
         dictRankForFwNode[node1] = 0.0
       
    for node1 in G.__iter__():
        for nb in G.neighbors(node1):
            for i in range (0, len(G.get_edge_data(node1, nb))): 
                temp = dictRankForFwNode[nb]
                dictRankForFwNode[nb] = G.get_edge_data(node1, nb)[i]['weightedF']*G.nodes[node1]['weight'] + temp
               
    return dictRankForFwNode 


def readStep(text):
    singPlCorp = {}
    tokenSize = []
    nlp = English()
    nlp.add_pipe('sentencizer')
    nlp2 = spacy.load("en_core_web_sm")
    processed_text = nlp2(text)
    lemma_tags = {"NNS", "NNPS"}
    singText = ""
    statePunc = False

    for token in processed_text:
       lemma = token.text
       if token.tag_ in lemma_tags and token.text != "-":

        lemma = singularize_word(token.text)
        singPlCorp[lemma.lower()] = token.text.lower()
        if statePunc == True:
            singText = singText + lemma
            statePunc = False
            continue
        singText = singText + " "  + lemma
       elif token.pos_ is "PUNCT" or token.text is "-":
        singText = singText + lemma
        if token.text is "-":
          statePunc = True
       else:
        if statePunc == True:
          singText = singText + lemma
          statePunc = False
          continue
        singText = singText + " "  + lemma

    text = spaceProblemSol(singText)
    text = text.lower()
    
    candidateKeyphrases,candidateWords = nounPhraseExt(text)
 
    return candidateKeyphrases,candidateWords, singPlCorp


def ranking(wordsResult,candidateKeyphrases,pwCorp,singPlCorp):

    G = nx.MultiGraph()
    isEmpty = []
    indFic = 0
    for w in wordsResult:
        if w not in isEmpty:
            isEmpty.append(w)
            G.add_node(w, weight = pwCorp[w])
        indFic = indFic + 1

    for i in range(0,len(wordsResult)):
        for j in range(i+1,len(wordsResult)):
            if wordsResult[i] != wordsResult[j]:
                G.add_edge(wordsResult[i], wordsResult[j], weight=abs(j-i), indiceF = i, indiceB = j)     

    ay1 = edgeWeightsForward(G,wordsResult)
    
    graphEdgeUpdation(ay1,G)     
    

    finalNode = nx.pagerank(G, alpha=0.85, weight='weightedF',personalization = pwCorp)

    finalNodeSorted = dict(sorted(finalNode.items(), key=lambda item: item[1], reverse=True))
        
    dictFinalNodeSorted = {}
    for a,b in finalNodeSorted.items():
        dictFinalNodeSorted[a] = b

    dictionaryEvaluation = {}

    for n in candidateKeyphrases:

        valuableVal = 0 
        valuableStr = "" 
        
        a = n.split()  
        
        for a1 in a:       
            if a1 in dictFinalNodeSorted: 
                if valuableVal is 0:
                    valuableVal = valuableVal + dictFinalNodeSorted[a1]
                    if a1 in singPlCorp:
                      valuableStr = valuableStr + " " + singPlCorp[a1].lower()
                    else:
                      valuableStr = valuableStr + " " + a1

                else:
                    valuableVal = valuableVal + dictFinalNodeSorted[a1]
                    if a1 in singPlCorp:
                      valuableStr = valuableStr + " " + singPlCorp[a1].lower()
                    else:
                      valuableStr = valuableStr + " " + a1

                    
        
        dictionaryEvaluation[valuableStr] = valuableVal


    finalSet = []
    keywordSize = 10
    for key, value in sorted(dictionaryEvaluation.items(), key=lambda item: item[1], reverse=True):
        if len(dictionaryEvaluation) > keywordSize:
            if len(finalSet) < keywordSize:
                finalSet.append(key)
        else:
            if len(finalSet) < len(dictionaryEvaluation): 
                finalSet.append(key)

    finalSet = [s.strip() for s in finalSet]            

    for f in finalSet:
      print(f)           
              
class MGRank():
  self.text
  candidateKeyphrases,candidateWords, singPlCorp = readStep(text)
  pwCorp = positionWeight(candidateWords)
  ranking(candidateWords,candidateKeyphrases,pwCorp,singPlCorp)

text = ''' Feedback effects between similarity and social influence in online communities A fundamental open question in the analysis of social networks is to understand the interplay between similarity and social ties. People are similar to their neighbors in a social network for two distinct reasons: first, they grow to resemble their current friends due to social influence; and second, they tend to form new links to others who are already like them, a process often termed selection by sociologists. While both factors are present in everyday social processes, they are in tension: social influence can push systems toward uniformity of behavior, while selection can lead to fragmentation. As such, it is important to understand the relative effects of these forces, and this has been a challenge due to the difficulty of isolating and quantifying them in real settings.'''
MGRank(text)
