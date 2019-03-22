#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 09:59:45 2018

@author: kksaikrishna
"""
import nltk
import scipy.stats
import glob
from enchant.checker import SpellChecker
import spacy

pathAllFiles = '/Users/kksaikrishna/Desktop/Coursework/NLP/Project/essays_dataset/essays'
pathHighFiles = '/Users/kksaikrishna/Desktop/Coursework/NLP/Project/essays_dataset/good_essay'
pathLowFiles = '/Users/kksaikrishna/Desktop/Coursework/NLP/Project/essays_dataset/bad_essay'
pathTestFile = '/Users/kksaikrishna/Desktop/Coursework/NLP/Project/essays_dataset/essays/339615.txt'
charCountMean = 1531.21
charCountStd = 628.874745796
wordCountMean = 295.68
wordCountStd = 121.345777018
sentCountMean = 13.9
sentCountStd = 6.74907401056
spErrCountMean = 9.34
spErrCountStd = 8.03768623423
svCountMean = 6.08
svCountStd = 4.02661147865
mvCountMean = 14.24
mvCountStd = 6.76331279773
scoreMean = 8.033841562932526
scoreStd = 6.0925116204070155

#Utils
def returnFileNamesAsList(path,ext='txt'):
        path += '/*.'+ ext
        files = glob.glob(path)    
        listOfFiles = [];
        for file in files:
            listOfFiles.append(file)        
        return listOfFiles

def returnFileContentAsStr(path):
    with open(path, 'r') as myfile:
        data = myfile.read()
        return data
    
#Normal Distribution
class Metrics:
    def __init__(self,lMean,lStd):
        self.mean = lMean
        self.sd = lStd    
    
    def getProbVal(self,value):
        return scipy.stats.norm.cdf(x=value,loc=self.mean,scale=self.sd)

#Natural Language ToolKit
class NLTKTools:
    def wordTokenize(self,file_content):
        tokens = nltk.word_tokenize(file_content)
        return tokens

    def sentTokenize(self,file_content):
        tokens = nltk.sent_tokenize(file_content)
        return tokens

    def posTagging(self,wordtokens):
        pos_tags = nltk.pos_tag(wordtokens)
        return pos_tags
    
    def spellCheck(self, text):
        listRet = []
        strErr = ''
        objChecker = SpellChecker('en_US')
        objChecker.set_text(text)
        for error in objChecker:
            strErr += error.word + ' '                        
        objChecker = SpellChecker('en_GB')
        objChecker.set_text(strErr)
        for error in objChecker:
            listRet.append(error.word)
        return listRet
    
    def isSvErr(self,s_tag,s_word,v_tag):
        setSNoun = ['NN','NNP']
        setPNoun = ['NNS','NNPS']
        setSPrp = ['he','she','it']
        setPPrp = ['i','you','we','they']
        ret = 0
        if(v_tag == 'VBZ'):
            if(s_tag in setSNoun or s_tag in setSPrp):
                ret = 1
        elif(v_tag == 'VBP'):
            if(s_tag in setPNoun or s_tag in setPPrp):
                ret = 1
        
        return ret
            
#Tokenize as Char, Word and Sentence
class Counter:
    def __init__(self):
        self.nltkTools = NLTKTools()
    def returnCharCount(self,file_content):
        return len(file_content)

    def returnWordCount(self,file_content):
        return len(self.nltkTools.wordTokenize(file_content))

    def returnSentCount(self,file_content):
        return len(self.nltkTools.sentTokenize(file_content)) - self.cleanSentCount(file_content)

    def cleanSentCount(self, file_content):
        prevWord = ''
        prevTag = ''
        errCount = 0
        posTokens = self.nltkTools.posTagging(self.nltkTools.wordTokenize(file_content))
        for tokens in posTokens:
            if(tokens[0][0].isupper() and tokens[0] != 'I' and tokens[1] != 'NNP'):
                if(prevWord != ''):
                    if(self.nltkTools.posTagging(prevWord[-1])[0][1] != '.' and prevTag != '``' and prevTag != '"'):
                        errCount += 1
            prevWord = tokens[0]
            prevTag = tokens[1]
        return errCount
    
    def returnSpellErrCount(self,file_content):
        spellErr = self.nltkTools.spellCheck(file_content)
        return len(spellErr)  
    
    def retMissVerbCount(self,file_content):
        tot_count = 0
        sent_tokens = self.nltkTools.sentTokenize(file_content)
        for sentence in sent_tokens:
            word_tokens = self.nltkTools.wordTokenize(sentence)
            pos_tokens = self.nltkTools.posTagging(word_tokens)
            listof_verbs=["VBP", "VBZ", "VB", "VBG","VBD","VBN"]
            verb_tags = [tag[1] for tag in pos_tokens if tag[1] in listof_verbs]
            if verb_tags:              
                tot_count = tot_count + 1
        return tot_count  
    
    def retSVErrCount(self,file_content):
        en_nlp = spacy.load('en')
        listSent = self.nltkTools.sentTokenize(file_content)
        errCount = 0
        for sent in listSent:
            uSent = unicode(sent, "utf-8")
            doc = en_nlp(uSent)
            for i in range(len(doc)):
                if(doc[i].dep_ == u'nsubj'):
                    for j in range(i,len(doc)):
                        if(doc[j].pos_ == u'VERB'):
                            s_tag = doc[i].tag_
                            s_word = doc[i].text
                            v_tag = doc[j].tag_
                            if(self.nltkTools.isSvErr(s_tag,s_word,v_tag)):
                                errCount +=1
                            break
        return errCount                        
                                    
class ScoreCalculator:    
    def __init__(self):
        self.counter = Counter()
        
    def LOS_Score(self,file_content):
        metricsChar = Metrics(charCountMean,charCountStd)
        metricsWord = Metrics(wordCountMean,wordCountStd)
        metricsSent = Metrics(sentCountMean,sentCountStd)
        pChar = metricsChar.getProbVal(self.counter.returnCharCount(file_content))
        pWord = metricsWord.getProbVal(self.counter.returnWordCount(file_content))
        pSent = metricsSent.getProbVal(self.counter.returnSentCount(file_content))
        value = (0.0*pChar) + (0.00*pWord) + (1.00*pSent)
        return self.scoreFromVal(value,5)
    
    def spell_Score(self,file_content):
        metricsSpell = Metrics(spErrCountMean,spErrCountStd)
        spellErrCount = self.counter.returnSpellErrCount(file_content)
        pSpellErr = metricsSpell.getProbVal(spellErrCount)
        return self.scoreFromVal(pSpellErr,4)
    
    def SVErr_score(self,file_content):
        metricsSV = Metrics(svCountMean,svCountStd)
        svErrCount =  self.counter.retSVErrCount(file_content)
        pSvErr = metricsSV.getProbVal(svErrCount)
        return self.scoreFromVal(pSvErr,5)
    
    def MissVerb_Score(self,file_content):
        metricsMV = Metrics(mvCountMean,mvCountStd)
        mvCount = self.counter.retMissVerbCount(file_content)
        pMVErr = metricsMV.getProbVal(mvCount)
        return self.scoreFromVal(pMVErr,5)
    
    def HighOrLow(self,score):
        metricsScore = Metrics(scoreMean,scoreStd)
        pScore = metricsScore.getProbVal(score)
        if(pScore >= 0.5):
            return 1
        else:
            return 0
        
    def scoreFromVal(self,value,resolution):
        score =  value * resolution
        return score



objScoreCalc = ScoreCalculator()
highScore = 0
lowScore = 0
listOfFiles = returnFileNamesAsList(pathLowFiles)
#listOfFiles = []
#listOfFiles.append(pathTestFile)
score = []
for files in listOfFiles:
    file_content = returnFileContentAsStr(files)
    scoreA =  objScoreCalc.LOS_Score(file_content)
    scoreB = objScoreCalc.spell_Score(file_content)
    scoreC1 = objScoreCalc.SVErr_score(file_content)
    scoreC2 = objScoreCalc.MissVerb_Score(file_content)
    score = (2*scoreA) - scoreB + scoreC1 + scoreC2
    highOrLow = objScoreCalc.HighOrLow(score)
#    print scoreA, scoreB, scoreC1, scoreC2, highOrLow
    if(highOrLow):
        highScore += 1
    else:
        lowScore += 1
print(lowScore)
print(highScore)
