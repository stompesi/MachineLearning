#-*- coding: utf-8 -*-
import matplotlib.pyplot as plt

#파이썬은 기본적인 모든 변수가 전역 변수

# 상자와 화살표 형태 정의
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

# 화살표로 주석 그리기
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
  createPlot.ax1.annotate(nodeTxt, xy=parentPt,
                          xycoords='axes fraction',
                          xytext=centerPt, textcoords='axes fraction',
                          va="center", ha="center", bbox=nodeType,
                          arrowprops=arrow_args)

def createPlot():
  flg = plt.figure(1, facecolor='white')
  flg.clf()
  # 플롯
  createPlot.ax1 = plt.subplot(111, frameon=False)
  plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
  plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
  plt.show()

def getNumLeafs(myTree):
  numLeafs = 0
  firstStr = myTree.keys()[0]
  secondDict = myTree[firstStr]
  for key in secondDict.keys():
    if type(secondDict[key]).__name__ == 'dict':
      numLeafs += getNumLeafs(secondDict[key])
    else:
      numLeafs += 1

  return numLeafs

def getTreeDepth(myTree):
  maxDepth = 0
  firstStr = myTree.keys()[0]
  secondDict = myTree[firstStr]
  for key in secondDict.keys():
    if type(secondDict[key]).__name__ == 'dict':
      thisDepth = 1 + getTreeDepth(secondDict[key])
    else:
      thisDepth = 1
    if thisDepth > maxDepth:
      maxDepth = thisDepth

  return maxDepth

def retrieveTree(i):
  listOfTrees = [{'no surfacing' : {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                 {'no surfacing' : {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                ]

  return listOfTrees[i]

# 자식 노드와 부모 노드 사이에 텍스트 플롯하기
def plotMidText(cntrPt, parentPt, txtString):
  xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
  yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
  createPlot.ax1.text(xMid, yMid, txtString)

# 넓이와 높이 구하기
def plotTree(myTree, parentPt, nodeTxt):
  numLeafs = getNumLeafs(myTree)
  getTreeDepth(myTree)
  firstStr = myTree.keys()[0]
  cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
  plotMidText(cntrPt, parentPt, nodeTxt)
  plotNode(firstStr, cntrPt, parentPt, decisionNode)
  secondDict = myTree[firstStr]
  plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
  for key in secondDict.keys():
    if type(secondDict[key]).__name__ == 'dict':
      plotTree(secondDict[key], cntrPt, str(key))
    else:
      plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
      plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
      plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
  plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

# def createPlot(inTree):
#   fig = plt.figure(1, facecolor='white')
#   fig.clf()
#   axprops = dict(xticks=[], yticks=[])
#   createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
#   plotTree.totalW = float(getNumLeafs(inTree))
#   plotTree.totalD = float(getTreeDepth(inTree))
#   plotTree.xOff = -0.5 / plotTree.totalW
#   plotTree.yOff = 1.0
#   plotTree(inTree, (0.5, 1.0), '')
#   plt.show()



# 69부터 











