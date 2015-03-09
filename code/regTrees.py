#-*- coding: utf-8 -*-

from numpy import *

def loadDataSet(fileName):
  dataMat = []
  fr = open(fileName)
  for line in fr.readlines():
    curLine = line.strip().split('\t')
    # 모든 값을 float() 처리 
    fltLine = map(float, curLine)
    dataMat.append(fltLine)

  return dataMat

def binSplitDataSet(dataSet, feature, value):
  # nonzero: 다음 조건을 만족하는 index matrix return
  mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :][0]
  mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :][0]

  return mat0, mat1

# 단말 노드에 대한 모델을 생성한다
# chooseBestSplit()이 더이상 데이터를 분할해서는 안된다고 결정했을때
# 이 노드에 대한 모델(목적 변수에 대한 평균값)을 구하기 위해 호출된다. 
def regLeaf(dataSet):
  # mean: 평균 
  # [: , -1]: 마지막 열의 모든 행 값
  return mean(dataSet[:, -1])

# 오류를 평가한다
# 주어진 데이터 집합 내에서 목적 변수에 대한 전체 제곱 오류를 반환한다.
def regErr(dataSet):
  # var: 분산 구하는 함수
  # shape: 면적 구하는 함수 - reutrn: (행, 열)
  return var(dataSet[:, -1]) * shape(dataSet)[0]

# 분류 트리를 구축하는 함수 
# 데이터를 이진 분할하는 가장 좋은 방법을 찾는 것 
# 좋은 이진 분할을 찾을 수 없다면 -> None을 반환하고 단말 노드를 생성한다.
# 좋은 분할을 찾게 된다면 -> 이 속성의 번화와 분할을 위한 값을 반환한다.

# ops = 새로운 분할이 그만 생성되어야 하는 때를 함수에게 알린다. 
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
  tolS = ops[0] # 오류를 줄이기 위해 사용되는 오차 범위
  tolN = ops[1] # 분할에 포함되는 데이터 사례의 최소 개수

  # 모든 값이 동일하면 종료
  # .T: 행에 값을 matrix의 하나의 행의 배열로 변경 
  # .tolist(): matrix를 배열로 변경
  if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
    return None, leafType(dataSet)

  # 집합의 크기 측정: 행, 열
  m, n = shape(dataSet)

  # 이 데이터 집합의 오류를 측정 - 분할이 오류를 줄이는지를 확인하기 위해 오류에 대한 새로운 값과 비교하게 된다.
  S = errType(dataSet)


  bestS = inf
  bestIndex = 0
  bestValue = 0

  # 가장 좋은 분할을 찾는다 
  for featIndex in range(n - 1):
    # 오류가 난다면 set(dataSet[:, -1].T.tolist()[0])로 변경 하기
    for splitVal in set(dataSet[:, featIndex]):
      mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
      if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        continue
      newS = errType(mat0) + errType(mat1)

      # 오류가 작은 것이 가장 좋은 분할
      if newS < bestS:
        bestIndex = featIndex
        bestValue = splitVal
        bestS = newS

  # 데이터 집합을 분할하여 오류가 너무 작게 개선된다면, 분할을 하지 않고 단말 노드를 생성한다.
  if (S - bestS) < tolS:
    return None, leafType(dataSet)

  mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)

  # 분할이 작은 데이터 집합을 생성했다면 종료
  if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
    return None, leafType(dataSet)

  return bestIndex, bestValue

"""
  dataSet: 트리를 구축하기 위한 하나의 데이터 집합
  leafType: 노드를 생성하는 데 사용되는 기능
  errType: 데이터 집합에 대한 오류를 측정하기 위해 사용되는 기능
  ops: 트리를 생성하기 위한 것
"""
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
  feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
  if feat == None:
    return val

  retTree = {}
  retTree['spInd'] = feat
  retTree['spVal'] = val
  lSet, rSet = binSplitDataSet(dataSet, feat, val)
  retTree['left'] = createTree(lSet, leafType, errType, ops)
  retTree['rihgt'] = createTree(rSet, leafType, errType, ops)
  return retTree

def isTree(obj):
  return (type(obj).__name__ == 'dict')

# 재귀 함수로서 단말 노드와 마주칠 때까지 트리를 축소한다
def getMean(tree):
  if isTree(tree['right']):
    tree['right'] = getMean(tree['right'])
  if isTree(tree['left']):
    tree['left'] = getMean(tree['left'])
  return (tree['right'] + tree['left']) / 2.0

# tree: 가지치기를 위한 트리
# testData: 트리를 가지치기하는 데 필요한 데이터

def prune(tree, testData):
  # 검사데이터가 비어있는지 확인 
  if shape(testData)[0] == 0:
    return getMean(tree)


  if (isTree(tree['right']) or isTree(tree['left'])):
    lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])

  # 트리인지 아닌지 확인하기 위한 검사 트리면 가지치기를 재귀적으로 호출  
  if isTree(tree['left']):
    tree['left'] = prune(tree['left'], lSet)
  if isTree(tree['right']):
    tree['right'] = prune(tree['right'], rSet)

  # 트리가 존재하는지를 확인하기 위한 검사 
  # 모두 트리가 아니라면 이 가지들은 병합하는 것이 가능하다.

  if not isTree(tree['left']) and not isTree(tree['right']):
    # 데이터를 분할하고 오류를 측정한다 
    lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2))

    treeMean = (tree['left'] + tree['right']) / 2.0
    errorMerge = sum(power(testData[:, -1] - treeMean, 2))

    # 두가지 병합했을 때의 오류가 병합하지 않았을 때의 오류보다 적다면 가지들을 병합한다.
    if errorMerge < errorNoMerge:
      print("merging")
      return treeMean
    else:
      return tree
  else:
    return tree

# 데이터 집합을 목적 변수 Y와 독립 변수 X의 형식으로 만든다.
# 역행렬을 확인할 수 없다는 예외가 발생한다.
def linearSolve(dataSet):
  m, n = shape(dataSet)
  # ones((m, n): m x n 배열 만드는 함수 
  X = mat(ones((m, n)))
  Y = mat(ones((m, 1)))

  X[:, 1:n] = dataSet[:, 0:n-1]
  Y = dataSet[:, -1]
  xTx = X.T * X

  if linalg.det(xTx) == 0:
    raise NameError('This matrix is singular, cannot do inverse,\n try increasing the second value of ops')
  ws = xTx.I * (X.T * Y)
  return ws, X, Y

# 단말 노드에 대한 모델을 생성하는 데 사용되며, 더이상 데이터가 분할되지 않을 때까지 수행한다.
# regLeaf와 유사하다.
# ws: 회귀 계수
def modelLeaf(dataSet):
  ws, X, Y = linearSolve(dataSet) 
  return ws

# 주어진 데이터 집합에 대한 오류를 계산한다.
# regErr() 함수 대신에 사용될 것이다.
def modelErr(dataSet):
  ws, X, Y = linearSolve(dataSet)
  yHat = X * ws
  return sum(power(Y - yHat), 2)



