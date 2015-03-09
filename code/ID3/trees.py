#-*- coding: utf-8 *-
import operator
from math import log

def createDataSet():
  dataSet = [
              [1, 1, 'yes'],
              [1, 1, 'yes'],
              [1, 0, 'no'],
              [0, 1, 'no'],
              [0, 1, 'no']
            ]
  labels  = ['no surfacing', 'flippers']
  return dataSet, labels

def calcShannonEnt(dataSet):
  numEntries = len(dataSet) # 데이터 집합에 있는 사례 개수
  labelCounts = {}
  # 가능한 모든 분류 항목에 대한 Dictionary 생성
  for featVec in dataSet:
    currentLabel = featVec[-1]
    if currentLabel not in labelCounts.keys():
      labelCounts[currentLabel] = 0
    labelCounts[currentLabel] += 1

  shannonEnt = 0.0
  for key in labelCounts:
    prob = float(labelCounts[key]) / numEntries
    shannonEnt -= prob * log(prob, 2)
  return shannonEnt

"""
  dataSet : 분할하고자 하는 데이터 집합,
  axis    : 분할하고자 하는 속성
  value   : 반할할 속성의 값
"""
def splitDataSet(dataSet, axis, value):
  # 새로운 리스트 생성
  retDataSet = []
  for featVec in dataSet:
    # 속성을 잘라내고 데이터 집합을 분할
    if featVec[axis] == value:
      reduceFeatVec = featVec[:axis]
      """
        a = [1, 2, 3]
        b = [4, 5, 6]
        a.append(b) : [1, 2, 3, [4, 5, 6]]
        a.extend(b) : [1, 2, 3, 4, 5, 6]
      """
      reduceFeatVec.extend(featVec[axis+1:])
      retDataSet.append(reduceFeatVec)
  return retDataSet

# 데이터 집합 분할하기
def chooseBestFeatureToSplit(dataSet):
  numFeatures = len(dataSet[0]) - 1 # 3
  baseEntropy = calcShannonEnt(dataSet)
  bestInfoGain = 0.0
  bestFeature = -1

  for i in range(numFeatures):
    featList = [example[i] for example in dataSet]
    # 분류 항목 표시에 대해 중복이 없는 리스트 생성
    uniqueVals = set(featList)
    newEntropy = 0.0

    # 각각의 분할을 위해 엔트로피 계산
    for value in uniqueVals:
      subDataSet = splitDataSet(dataSet, i, value)
      prob = len(subDataSet) / float(len(dataSet))
      newEntropy += prob * calcShannonEnt(subDataSet)

    infoGain = baseEntropy - newEntropy

    # 가장 큰 정보 이득 찾기
    if(infoGain > bestInfoGain):
      bestInfoGain = infoGain
      bestFeature = i

  return bestFeature

def createTree(dataSet, labels):
  classList = [example[-1] for example in dataSet]

  # 모든 분류 항목이 같을 때 멈춤
  if classList.count(classList[0]) == len(classList):
    return classList[0]

  # 속성이 더 이상 없을 때 가장 많은 수를 반환함
  if len(dataSet[0]) == 1:
    return majorityCnt(classList)

  bestFeat = chooseBestFeatureToSplit(dataSet)
  bestFeatLabel = labels[bestFeat]
  myTree = {
    bestFeatLabel: {}
  }
  del(labels[bestFeat])
  # 유일한 값의 리스트를 구함
  featValues = [example[bestFeat] for example in dataSet]
  uniqueVals = set(featValues)

  for value in uniqueVals:
    subLabels = labels[:]
    myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

  return myTree

def majorityCnt(classList):
  classCount = {}
  for vote in classList:
    if vote not in classCount.keys():
      classCount[vote] = 0
    classCount[vote] += 1
  # 정렬 - 값에 따른 내림차순 python3에서는 iteritems -> items로 변경 
  sortedClassCount = sorted(classCount.items로(), key=operator.itemgetter(1), reverse=True)

  # 가장 많이 가지고 있는 값의 label을 return 
  return sortedClassCount[0][0]


def classify(inputTree, featLabels, testVec):
  # 오류 inputTree.keys()[0] -> list(inputTree.keys())[0]
  firstStr = list(inputTree.keys())[0]
  secondDict = inputTree[firstStr]

  # first key의 index 추출 
  featIndex = featLabels.index(firstStr)

  for key in secondDict.keys():
    if testVec[featIndex] == key:
      if type(secondDict[key]).__name__ == 'dict':
        classLabel = classify(secondDict[key], featLabels, testVec)
      else:
        classLabel = secondDict[key]

  return classLabel

def storeTree(inputTree, filename):
  import pickle
  fw = open(filename, 'w')
  pickle.dump(inputTree, fw)
  fw.close()

def grabTree(filename):
  import pickle
  fr = open(filename)
  return pickel.load(fr)
