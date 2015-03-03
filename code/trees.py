from math import log

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

def createDataSet():
  dataSet = [[1, 1, 'yes'],
             [1, 1, 'yes'],
             [1, 0, 'no'],
             [1, 1, 'no'],
             [1, 1, 'no']]
  labels = ['no surfacing', 'flippers']
  return dataSet, labels


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