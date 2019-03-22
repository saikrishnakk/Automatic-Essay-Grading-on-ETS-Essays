#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 09:59:45 2018

@author: kksaikrishna
"""
from __future__ import print_function
import nltk
import scipy.stats
from scipy import spatial
from enchant.checker import SpellChecker
import spacy
import pandas as pd
import numpy  as np
import sys
import re
import glob
#from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.corpus import wordnet

results_file_path = '../output/results.txt'
test_csv_file_path = '../input/testing/index.csv'
test_file_path = '../input/testing/essays/'
test_delimiter = ';'

train_file_path = '../input/training/essays/'
train_csv_file_path = '../input/training/index.csv'
train_delimiter = ';'

trained_values_path = 'resources/trained_values_temp.txt'
word2vec_path = 'resources/GoogleNews-vectors-negative300.bin.gz'
#print('Loading Word2Vec... It might take quite some time, depending on the speed of your machine')
#word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
#print('Loaded Word2Vec')

stops = set(stopwords.words('english'))
en_nlp = spacy.load('en')

male_path = 'resources/male.txt'
female_path = 'resources/female.txt'

#Utils
def removeText(mainText,subText):
    mainText = mainText[len(subText):-1]
    return mainText
    
def retFileNamesAsListFolder(path,ext='txt'):
        path += '/*.'+ ext
        files = glob.glob(path)    
        listOfFiles = [];
        for file in files:
            listOfFiles.append(file)        
        return listOfFiles

def retDataFromCsv(indexPath,delim):
    data = pd.read_csv(indexPath,delimiter = delim)
    fileNames = data.filename.tolist()
    topics = data.prompt.tolist()
    for topic in topics:
        topic = removeText(topic,'Do you agree or disagree with the following statement?')
    return fileNames,topics

def returnFileContentAsStr(path):
    with open(path, 'r') as myfile:
        data = myfile.read()
        return data
    
def readFileAsFloat(path):
    with open(path, 'r') as f:
        floats = map(float, f)
    return floats

def writeResult(output,path):
    with open(path,'a+') as f:   #************** Change it to w+ ************
        for content in output:
            f.write(content)   
            
def getOutputString(fileName,scoreA,scoreB,scoreC1,scoreC2,scoreC3,scoreD1,finalScore,highOrLow):  
    ret = ''
    ret = fileName + ';' + str(np.round(scoreA,2)) + ';' + str(np.round(scoreB,2)) + ';' 
    ret += str(np.round(scoreC1,2)) + ';' + str(np.round(scoreC2,2)) + ';' + str(np.round(scoreC3,2)) + ';'
    ret += str(np.round(scoreD1,2))
    ret += ';' + str(np.round(finalScore,2)) + ';' + highOrLow + '\n'
    return ret

def calculateScore(mean,sd,value,res = 4,offset = 1):
    prob = scipy.stats.norm.cdf(x=value,loc=mean,scale=sd)
    return (prob * res) + offset

def isHighOrLow(val):
    if val >= 2.5:
        return 'high'
    else:
        return 'low'
    
        
def getWordEmbedding(text1,text2):
    vocabulary = dict()
    inverse_vocabulary = ['<unk>']
    words1 = nltk.word_tokenize(text1)
    for word in words1:
#        if word in stops and word not in word2vec.vocab:
#            continue
        if word not in vocabulary:
            vocabulary[word] = len(inverse_vocabulary)
            inverse_vocabulary.append(word)
    
    words2 = nltk.word_tokenize(text2)
    for word in words2:
#        if word in stops and word not in word2vec.vocab:
#            continue
        if word not in vocabulary:
            vocabulary[word] = len(inverse_vocabulary)
            inverse_vocabulary.append(word)
            
    embedding_dim = 300
    embeddings1 = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
    embeddings1[0] = 0
    
    for word in words1:
    #    if word in word2vec.vocab:
        embeddings1[vocabulary[word]] = 2.22#word2vec.word_vec(word)
            
    embeddings2 = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
    embeddings2[0] = 0
    
    for word in words2:
#        if word in word2vec.vocab:
        embeddings2[vocabulary[word]] = 2.22#word2vec.word_vec(word)
    
    return embeddings1,embeddings2

def calcCosineDist(vector1,vector2):
    result = spatial.distance.cosine(vector1, vector2)
    return result
    
#    dp = np.dot(vector1,vector2)
#    a = np.linalg.norm(vector1)
#    b = np.linalg.norm(vector2)
#    val = dp/(a*b)
#    return val

def getSynonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return synonyms

def appendSynonymsToText(text):
    retList = []
    words = nltk.word_tokenize(text)
    for word in words:
        retList.append(word)
        retList.append(getSynonyms(word))
    retStr = ' '.join(str(token) for token in retList)
    return retStr
        

class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='=',
                 output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        print('\r' + self.fmt % args, file=self.output, end='')

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)

class Counter:
    def preProcessData(self,file_content):
        self.file_content = file_content
        self.sent_tokens = nltk.sent_tokenize(file_content)
        self.word_tokens = []
        self.pos_tokens = []
        self.spacy_tokens = []
        for sent in self.sent_tokens:
            words = nltk.word_tokenize(sent)
            self.word_tokens.append(words)
            
            posTags = nltk.pos_tag(words)
            self.pos_tokens.append(posTags)
            
            uSent = unicode(sent, "utf-8")
            spacyOut = en_nlp(uSent)
            self.spacy_tokens.append(spacyOut)
    
    
    def retCharCount(self):
        return len(self.file_content)
    
    def retWordCount(self):
        wc = 0
        for words in self.word_tokens:
            wc += len(words)
        return wc
    
    def retSentCount(self):
        return len(self.sent_tokens)
    
    def retSpErrCount(self):
        chkr = SpellChecker('en_US','en_GB')
        chkr.set_text(self.file_content)
        count = 0
        for err in chkr:
            count += 1
        return count
    
    def retSvAgrCount(self):
        count = 0
        setSNoun = ['NN','NNP']
        setPNoun = ['NNS','NNPS']
        setSPrp = ['he','she','it']
        setPPrp = ['i','you','we','they']
        for i in range(len(self.sent_tokens)):
            for j in range(len(self.spacy_tokens[i])):
                if(self.spacy_tokens[i][j].dep_ == u'nsubj'):
                    for k in range(j,len(self.spacy_tokens[i])):
                        if(self.spacy_tokens[i][k].pos_ == u'VERB'):
                            if(self.spacy_tokens[i][k].tag_ == 'VBZ'):
                                if(self.spacy_tokens[i][j].tag_ in setSNoun or self.spacy_tokens[i][j].text.lower() in setSPrp):
                                    count += 1
                            elif(self.spacy_tokens[i][k].tag_ == 'VBP'):
                                if(self.spacy_tokens[i][j].tag_ in setPNoun or self.spacy_tokens[i][j].text.lower() in setPPrp):
                                    count += 1
                            break
        return count          
    
    def retVTAgrCount(self):
        count = 0
        present_tense_verbs={"VBP", "VBZ", "VB", "VBG"}
        past_tense_verbs={"VBD", "VBN"}
        for i in range(len(self.sent_tokens)):
            count_present=0
            count_past=0
            verb_tags = [tag[1] for tag in self.pos_tokens[i] if tag[1] in present_tense_verbs or tag[1] in past_tense_verbs]
            for v in verb_tags:
                if(v in present_tense_verbs):
                    count_present = count_present + 1;
                if(v in past_tense_verbs):
                    count_past = count_past + 1;
            if(count_present==0 and count_past>0):
                count += 1
            elif(count_present>0 and count_past==0):
                count += 1
        return count 
    
    def retVerbCount(self):
        count = 0
        for i in range(len(self.sent_tokens)):
            verb_postags =["VBP", "VBZ", "VB", "VBG","VBD","VBN"]
            for tag in self.pos_tokens[i]:
                if(tag[1] in verb_postags):
                    count = count + 1     
        return count
    
    def retSubjCount(self):
        count = 0
        for i in range(len(self.sent_tokens)):
            for j in range(len(self.spacy_tokens[i])):
                if(self.spacy_tokens[i][j].dep_ == u'nsubj'):
                    count += 1
                    break
        return count
    
    def retCapCount(self):
        count = 0
        for i in range(len(self.sent_tokens)):
            if(self.sent_tokens[i][0].isupper() and self.sent_tokens[i][-1]=='.'):
                count += 1
        return count
    


######To be Structured#########

from nltk.corpus import names
import random

def gender_features(word):
    return {'last_letter': word[-1]}


labeled_names = ([(name, 'male') for name in names.words()] + [(name, 'female') for name in names.words('female.txt')])
random.shuffle(labeled_names)
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)

def textcoherency(file_content):
    sent_tokens = nltk.sent_tokenize(file_content)
    malePRP = ["he","his"]
    femalePRP = ["she","her"]
    pluralPRP = ["they","them","their"]
    totalSingPrp = ["he","his","she","her"]
    errorcount = 0
    #for sentence in sent_tokens:
    if(len(sent_tokens) > 2):
        twosent = sent_tokens[0] + sent_tokens[1]
        gender = ''
        for i in range(2 , len(sent_tokens)):
            word_tokens = nltk.word_tokenize(sent_tokens[i])
            pos_tokens = nltk.pos_tag(word_tokens)
            #print(pos_tokens)
            for pos in pos_tokens:
                prev_tokens = nltk.word_tokenize(twosent)
                prev_pos_tokens = nltk.pos_tag(prev_tokens)
                if(pos[1] == 'PRP' or pos[1] == 'PRP$' and pos[0].lower() in totalSingPrp):
                    #print("Yaay PRP", pos[0])
                    for pos_prev in prev_pos_tokens:
                        if(pos_prev[1] == "NNP" or pos_prev[1] == "NNPS"):
                            #print("Yaay NNP", pos_prev[0])
                            gender = classifier.classify(gender_features(pos_prev[0]))
                            #print("gender" , gender)
                            if(gender == "male" and pos[0].lower() in malePRP):
                                #print("gender mismatch")
                                errorcount = errorcount + 1
                            if(gender == "female" and pos[0].lower() in femalePRP):
                                #print("gender mismatch")
                                errorcount = errorcount + 1
                if(pos[1] == 'PRP' or pos[1] == 'PRP$' and pos[0].lower() in pluralPRP):
                    for pos_prev in prev_pos_tokens:
                        if(pos_prev[1] == "NNPS"):
                            errorcount = errorcount + 1
                            
            #find all word Noun tags in prev sentence and look for singular/plural and gender
            twosent = sent_tokens[i-1] + sent_tokens[i]
    return errorcount

def distribution(file_content):
    score = 0
    noun_list = ["NN","NNS","NNP","NNPS"]
    verb_list = ["VBP", "VBZ", "VB", "VBG","VBD","VBN"]
    adv_list = ["RB","RBR","RBS"]
    adj_list = ["JJ","JJR","JJS"]   
    sent_tokens = nltk.sent_tokenize(file_content)
    for sentence in sent_tokens:
        word_tokens = nltk.word_tokenize(sentence)
        pos_tokens = nltk.pos_tag(word_tokens) 
        list_of_noun = [tag[1] for tag in pos_tokens if tag[1] in noun_list]
        list_of_verbs = [tag[1] for tag in pos_tokens if tag[1] in verb_list]
        list_of_adv = [tag[1] for tag in pos_tokens if tag[1] in adv_list]
        list_of_adj = [tag[1] for tag in pos_tokens if tag[1] in adj_list]
        if(list_of_noun and list_of_verbs and list_of_adj and list_of_adv):
            score = score + 4
        elif(list_of_noun and list_of_verbs and list_of_adv):
            score = score + 3
        elif(list_of_noun and list_of_verbs and list_of_adj):
            score = score + 3
        elif(list_of_noun and list_of_verbs):
            score = score + 2 
    return score

def nsubjcheck(file_content):
    count = 0
    sent_tokens = nltk.sent_tokenize(file_content)
    for sentence in sent_tokens:
        uSent = unicode(sentence, "utf-8")
        doc = en_nlp(uSent)
        root = ""
        listofverb = []
        for token in doc:
            if(token.dep_ == "nsubj"):
                listofverb.append(token.head.text)
            if(token.dep_ == "ROOT" and token.head.pos_ == "VERB"):
                root = token.text
        if(root in listofverb):
            count = count +1
        for verb in listofverb:
            if(verb != root):
                for token in doc:
                    if(token.text == verb):
                        if(token.head.text == root):
                            count = count + 1
    return count

def dobjcheck(file_content):
    count = 0
    sent_tokens = nltk.sent_tokenize(file_content)
    for sentence in sent_tokens:
        uSent = unicode(sentence, "utf-8")
        doc = en_nlp(uSent)
        root = ""
        listofverb = []
        for token in doc:
            if(token.dep_ == "dobj"):
                listofverb.append(token.head.text)
            if(token.dep_ == "ROOT" and token.head.pos_ == "VERB"):
                root = token.text
        if(root in listofverb):
            count = count +1
        for verb in listofverb:
            if(verb != root):
                for token in doc:
                    if(token.text == verb):
                        if(token.head.text == root):
                            count = count + 1
    return count


def checkforrootverb(file_content):
    count = 0
    sent_tokens = nltk.sent_tokenize(file_content)
    for sentence in sent_tokens:
        uSent = unicode(sentence, "utf-8")
        doc = en_nlp(uSent)
        for token in doc:
            if(token.dep_ == "ROOT" and token.head.pos_ == "VERB"):
                count = count + 1
                
    return count 



##################################################################################################################


class AutomaticEssayGrader:
    def train(self):
        print('Training Started')
        listFileNames,listPrompt = retDataFromCsv(train_csv_file_path,train_delimiter)
        objCounter = Counter()
        listCharCount = []
        listWordCount = []
        listSentCount = []
        listSpErrCount = []
        listVerbCount = []
        listSvAgrCount = []
        listVTAgrCount = []
        
        listSubjCount = []
        listSubjRoot = []
        listObjRoot = []
        listRoot = []
        listCapCount = []
        
        listDist = []
        listTC = []
        
#        listCosineDist = []
        
        
        
        listFinalScore = []
        
        progressTrain = ProgressBar(len(listFileNames), fmt=ProgressBar.FULL)
        
        for files,topic in zip(listFileNames,listPrompt):
            progressTrain.current += 1
            progressTrain()
            path = train_file_path + files
            file_content = returnFileContentAsStr(path)
            objCounter.preProcessData(file_content)
            listCharCount.append(objCounter.retCharCount())
            listWordCount.append(objCounter.retWordCount())
            listSentCount.append(objCounter.retSentCount())
            listSpErrCount.append(objCounter.retSpErrCount())
            listSvAgrCount.append(objCounter.retSvAgrCount())
            listVTAgrCount.append(objCounter.retVTAgrCount())
            listVerbCount.append(objCounter.retVerbCount())
            
            listSubjCount.append(objCounter.retSubjCount())
            listSubjRoot.append(nsubjcheck(file_content))
            listObjRoot.append(dobjcheck(file_content))
            listRoot.append(checkforrootverb(file_content))
            listCapCount.append(objCounter.retCapCount())
            
            listDist.append(distribution(file_content))
            listTC.append(textcoherency(file_content))

#            extended_prompt = appendSynonymsToText(topic)
#            file_embed,prompt_embed = getWordEmbedding(file_content,extended_prompt)
#            listCosineDist.append(calcCosineDist(file_embed,prompt_embed))
            
        
        m_char = np.mean(listCharCount)
        sd_char = np.std(listCharCount)
        
        m_word = np.mean(listWordCount)
        sd_word = np.std(listWordCount)
        
        m_sent = np.mean(listSentCount)
        sd_sent = np.std(listSentCount)
        
        m_sp = np.mean(listSpErrCount)
        sd_sp = np.std(listSpErrCount)
        
        m_sv = np.mean(listSvAgrCount)
        sd_sv = np.std(listSvAgrCount)
        
        m_vt = np.mean(listVTAgrCount)
        sd_vt = np.std(listVTAgrCount)
        
        m_verb = np.mean(listVerbCount)
        sd_verb = np.std(listVerbCount)
        
        m_sub = np.mean(listSubjCount)
        sd_sub = np.std(listSubjCount)
        
        m_cap = np.mean(listCapCount)
        sd_cap = np.std(listCapCount)
        
#        m_cos = np.mean(listCosineDist)
#        sd_cos = np.std(listCosineDist)
        
        m_sr = np.mean(listSubjRoot)
        sd_sr = np.std(listSubjRoot)
        
        m_or = np.mean(listObjRoot)
        sd_or = np.std(listObjRoot)
        
        m_rt = np.mean(listRoot)
        sd_rt = np.std(listRoot)
        
        m_d = np.mean(listDist)
        sd_d = np.std(listDist)
        
        m_tc = np.mean(listTC)
        sd_tc = np.std(listTC)
        
        for i in range(len(listFileNames)):
            scoreA_1 = calculateScore(m_char,sd_char,listCharCount[i])
            scoreA_2 = calculateScore(m_word,sd_word,listWordCount[i])
            scoreA_3 = calculateScore(m_sent,sd_sent,listSentCount[i])
            scoreA = 0.1*scoreA_1 + 0.2*scoreA_2 + 0.7*scoreA_3
            
            scoreB = calculateScore(m_sp,sd_sp,listSpErrCount[i],4,0)
            
            scoreCi_1 = calculateScore(m_sv,sd_sv,listSvAgrCount[i])
            scoreCi_2 = calculateScore(m_vt,sd_vt,listVTAgrCount[i])
            scoreCi = 0.5*scoreCi_1 + 0.5*scoreCi_2
            
            scoreCii = calculateScore(m_verb,sd_verb,listVerbCount[i])
            
            scoreCiii_1 = calculateScore(m_sub,sd_sub,listSubjCount[i])
            scoreCiii_2 = calculateScore(m_cap,sd_cap,listCapCount[i])
            scoreCiii_3 = calculateScore(m_sr,sd_sr,listSubjRoot[i])
            scoreCiii_4 = calculateScore(m_or,sd_or,listObjRoot[i])
            scoreCiii_5 = calculateScore(m_rt,sd_rt,listRoot[i])
            scoreCiii = 0.2*scoreCiii_1 + 0.2*scoreCiii_2 + 0.2*scoreCiii_3 + 0.2*scoreCiii_4 + 0.2*scoreCiii_5
            
            scoreDi_1 = calculateScore(m_d,sd_d,listDist[i])
            scoreDi_2 = calculateScore(m_tc,sd_tc,listTC[i])
            scoreDi = 0.9*scoreDi_1 + 0.1*scoreDi_2
            
#            scoreDii = calculateScore(m_cos,sd_cos,listCosineDist[i])
            
            finalScore = 3*(scoreA) - 2(scoreB) + 0.5(scoreCi) + 2*scoreCii + 3*(scoreCiii) + 1*(scoreDi) 
            
            listFinalScore.append(finalScore)
            
        
        m_fs = np.mean(listFinalScore)
        sd_fs = np.std(listFinalScore)
        
        trained_out = str(m_char) + '\n' + str(sd_char) + '\n'
        trained_out += str(m_word) + '\n' + str(sd_word) + '\n'
        trained_out += str(m_sent) + '\n' + str(sd_sent) + '\n'
        trained_out += str(m_sp) + '\n' + str(sd_sp) + '\n'
        trained_out += str(m_sv) + '\n' + str(sd_sv) + '\n'
        trained_out += str(m_vt) + '\n' + str(sd_vt) + '\n'
        trained_out += str(m_verb) + '\n' + str(sd_verb) + '\n'
        trained_out += str(m_sub) + '\n' + str(sd_sub) + '\n'
        trained_out += str(m_cap) + '\n' + str(sd_cap) + '\n'  
        trained_out += str(m_sr) + '\n' + str(sd_sr) + '\n'   
        trained_out += str(m_or) + '\n' + str(sd_or) + '\n'   
        trained_out += str(m_rt) + '\n' + str(sd_rt) + '\n'    
        trained_out += str(m_d) + '\n' + str(sd_d) + '\n'   
        trained_out += str(m_tc) + '\n' + str(sd_tc) + '\n'
        trained_out += str(0) + '\n' + str(0) + '\n' 
        trained_out += str(m_fs) + '\n' + str(sd_fs) + '\n'   
            
        writeResult(list(trained_out),trained_values_path)
        print('\nTraining Complete')
        
    def test(self):
        print('\nTesting Started')
        tv = readFileAsFloat(trained_values_path)
        lowScore = 0
        highScore = 0
        output = []
        objCounter = Counter()

        listFileNames,listPrompt = retDataFromCsv(train_csv_file_path,train_delimiter)

        progressTest = ProgressBar(len(listFileNames), fmt=ProgressBar.FULL)
        for files,topic in zip(listFileNames,listPrompt):
            progressTest.current += 1
            progressTest()
            path = test_file_path + files
            file_content = returnFileContentAsStr(path)
            
            objCounter.preProcessData(file_content)
            
            scoreA_1 = calculateScore(tv[0],tv[1],objCounter.retCharCount())
            scoreA_2 = calculateScore(tv[2],tv[3],objCounter.retWordCount())
            scoreA_3 = calculateScore(tv[4],tv[5],objCounter.retSentCount())
            scoreA = 0.1*scoreA_1 + 0.2*scoreA_2 + 0.7*scoreA_3
            
            scoreB = calculateScore(tv[6],tv[7],objCounter.retSpErrCount(),4,0)
            
            scoreCi_1 = calculateScore(tv[8],tv[9],objCounter.retSvAgrCount())
            scoreCi_2 = calculateScore(tv[10],tv[11],objCounter.retVTAgrCount())
            scoreCi = 0.5*scoreCi_1 + 0.5*scoreCi_2
            
            scoreCii = calculateScore(tv[12],tv[13],objCounter.retVerbCount())
          
            
            scoreCiii_1 = calculateScore(tv[14],tv[15],objCounter.retSubjCount())
            scoreCiii_2 = calculateScore(tv[16],tv[17],objCounter.retCapCount())
            scoreCiii_3 = calculateScore(tv[18],tv[19],nsubjcheck(file_content))
            scoreCiii_4 = calculateScore(tv[20],tv[21],dobjcheck(file_content))
            scoreCiii_5 = calculateScore(tv[22],tv[23],checkforrootverb(file_content))
            scoreCiii = 0.2*scoreCiii_1 + 0.2*scoreCiii_2 + 0.2*scoreCiii_3 + 0.2*scoreCiii_4 + 0.2*scoreCiii_5
            
            
            scoreDi_1 = calculateScore(tv[24],tv[25],distribution(file_content))
            scoreDi_2 = calculateScore(tv[26],tv[27],textcoherency(file_content))
            scoreDi = 0.5*scoreDi_1 + 0.5*scoreDi_2
            
            
#            file_embed = getWordEmbedding(file_content)
#            extended_prompt = appendSynonymsToText(topic)
#            prompt_embed = getWordEmbedding(extended_prompt)
#            cosDist = calcCosineDist(file_embed,prompt_embed)
#            scoreDii = calculateScore(tv[28],tv[29],cosDist)
            
            total = 3*(scoreA) - 2(scoreB) + 0.5(scoreCi) + 2*scoreCii + 3*(scoreCiii) + 1*(scoreDi) #+ 3*(scoreDii)
            
            finalScore = calculateScore(tv[30],tv[31],total)
            highOrLow = isHighOrLow(finalScore)
            output.append(getOutputString(files,scoreA,scoreB,scoreCi,scoreCii,scoreCiii,scoreDi,finalScore,highOrLow))
            if highOrLow == 'high':
                highScore += 1
            elif highOrLow == 'low':
                lowScore += 1
        writeResult(output,results_file_path)
        print('\nTesting Complete')
        print('Number of Essays graded High:',highScore)
        print('Number of Essays graded Low:', lowScore)
        return [highScore,lowScore]

objEssayGrader = AutomaticEssayGrader()

if len(sys.argv) != 2:
    print('Invalid Arguments. \nExecute\n      nltkdemo.py -h\nto know more.')
elif sys.argv[1] == '-train':
    objEssayGrader.train()
elif sys.argv[1] == '-test':
    objEssayGrader.test()
elif sys.argv[1] == '-both':
    objEssayGrader.train()
    objEssayGrader.test()
elif sys.argv[1] == '-h' or sys.argv[1] == '--help':
    print('\n\n*********Automatic Essay Grader*********\n\n')
    print('Format:   nltkdemo.py <task>')
    print('\n<task>\n-train : To train with new set of data \n-test : To run the test with pre-computed values\n-both : To run the training and then test based on that\n-h or --help : For Help')
    print('\nExample: nltkdemo.py -test\nnltkdemo.py -both\n\n\n\n')
    print('\n\n****************************************\n\n')
else:
    print('Invalid Arguments. Execute\n nltkdemo.py -h\n to know more...')
    
    
#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
##import numpy as np
##import nltk
#
#
#def getDict(file_content):
#    tokens = word_tokenize(file_content)
#    words = [w.lower() for w in tokens]
#    print(words)
#
#    porter = nltk.PorterStemmer()
#    stemmed_tokens = [porter.stem(t) for t in words]
#    print(stemmed_tokens)
#
#    stop_words = set(stopwords.words('english'))
#    final_tokens = [w for w in stemmed_tokens if w not in stop_words]
#    print(final_tokens)
#
#    count = nltk.defaultdict(int)
#    for tok in final_tokens:
#        count[tok] += 1
#    
#    print(count)
#    return count



#vocabulary = dict()
#inverse_vocabulary = ['<unk>']
#EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin.gz'
#word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
#
#print('loaded')
#
#stops = set(stopwords.words('english'))
#
#listFileNames = retFileNamesAsListFolder('/Users/kksaikrishna/Desktop/Coursework/NLP/Project/skalah2_mrampr2/input/training/good_essay')
#for files in listFileNames:
#    path = files
#    file_content = returnFileContentAsStr(path)
#    words = nltk.word_tokenize(file_content)
#    for word in words:
#        if word in stops and word not in word2vec.vocab:
#            continue
#        if word not in vocabulary:
#            vocabulary[word] = len(inverse_vocabulary)
#            inverse_vocabulary.append(word)
#
#    embedding_dim = 300
#    embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
#    embeddings[0] = 0
#    
#    for word, index in vocabulary.items():
#        if word in word2vec.vocab:
#            embeddings[index] = word2vec.word_vec(word)
#    
#    print(embeddings)
#
#dp = np.dot(embeddings[1],embeddings[2])
#a = np.linalg.norm(embeddings[1])
#b = np.linalg.norm(embeddings[2])
#
#
#val = dp/(a*b)
#
#print(embeddings[1],embeddings[2],dp,a,b,val)













    

#dict1 = getDict("Hello world!")
#dict2 = getDict("World is a great place to live in!")
    
#vocab = dict()
#inv_vocab = ['<unk>'] 
#
#all_words = []
#for key in dict1:
#    all_words.append(key)
#for key in dict2:
#    all_words.append(key)
#    
#v1 = np.zeros(len(all_words),dtype = np.int)
#v2 = np.zeros(len(all_words),dtype = np.int)
#
#for i in range(len(all_words)):
#    v1[i] = dict1.get(all_words[i],0)
#    v2[i] = dict2.get(all_words[i],0)
#
#dp = np.dot(v1,v2)
#a = np.linalg.norm(v1)
#b = np.linalg.norm(v2)
#
#
#val = dp/(a*b)
#
#print(v1,v2,dp,a,b,val)
    












        
#listFileNames = returnFileNamesAsList('/Users/kksaikrishna/Desktop/Coursework/NLP/Project/skalah2_mrampr2/input/training/essays')
#objCounter = Counter()   
#listS = []
#for files in listFileNames:
#            path = files
#            file_content = returnFileContentAsStr(path)
#            listS.append(objCounter.frag(file_content))
#            print(listS[-1:])
#            
###writeResult(str(listS),'../output/resultsMSubjlow.txt')
#mean = np.mean(listS)
#std = np.std(listS)
#print(mean)
#print(std)
#metricsVS = Metrics(mean,std)
#
#listFileNames = returnFileNamesAsList('/Users/kksaikrishna/Desktop/Coursework/NLP/Project/skalah2_mrampr2/input/training/good_essay')
#listS = []
#highscore = 0
#lowscore = 0
#sum1 = 0
#for files in listFileNames:
#    path = files
#    file_content = returnFileContentAsStr(path)
#    count = objCounter.frag(file_content)   
#    sum1 += count         
#    pSpellErr = metricsVS.getProbVal(count)
#    score = pSpellErr * 5
#    if score>=2.5:
#        highscore+=1
#    else:
#        lowscore+=1
#
#print(highscore,lowscore,sum1)
#
#listFileNames = returnFileNamesAsList('/Users/kksaikrishna/Desktop/Coursework/NLP/Project/skalah2_mrampr2/input/training/bad_essay')
#listS = []
#highscore = 0
#lowscore = 0
#sum1 = 0
#for files in listFileNames:
#    path = files
#    file_content = returnFileContentAsStr(path)
#    count = objCounter.frag(file_content)  
#    sum1 += count          
#    pSpellErr = metricsVS.getProbVal(count)
#    score = pSpellErr * 5
#    if score>=2.5:
#        highscore+=1
#    else:
#        lowscore+=1
#
#print(highscore,lowscore,sum1)
            
            

#print(sent_tokens[i])
#                print(output['sentences'][0]['parse'])
#                writeResult(str('\n' + sent_tokens[i]),'../output/resultsMSubjlow.txt')
#                writeResult(str('\n' + output['sentences'][0]['parse']),'../output/resultsMSubjlow.txt')
#        writeResult(str('\n\n\n\n\n\n\nNew FIle:\n\n'),'../output/resultsMSubjlow.txt')
#        print('\n\n\n\n\n\n\n\n\n\n\n\n')                                                               








