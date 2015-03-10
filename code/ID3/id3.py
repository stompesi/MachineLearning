#-*- coding: utf-8 *-
import operator
import math
from  xml.dom import  minidom
from xml.etree import ElementTree as ET

def train(training_data_set, xmldir):
  # 속성값 추가
  labels = training_data_set[0]
  data_set = training_data_set[1:]

  root = ET.Element('DecisionTree')
  tree = ET.ElementTree(root)

  my_tree = create_tree(data_set, labels, root)

  store_tree(my_tree, "model.txt")
  tree.write(xmldir)
  ET.dump(prettify(root))

  return True

def create_tree(data_set, labels, parent):
  class_list = [example[-1] for example in data_set]

  # 모든 분류 항목이 같을 때 멈춤
  if class_list.count(class_list[0]) == len(class_list):
    parent.text = class_list[0]
    return class_list[0]

  # 속성이 더 이상 없을 때 가장 많은 수를 반환함
  if len(data_set[0]) == 1:
    parent.text = majority_cnt(class_list)
    return parent.text

  best_feature = choose_best_feature_to_split(data_set)
  best_feature_label = labels[best_feature]
  my_tree = {
    best_feature_label: {}
  }
  del(labels[best_feature])
  # 유일한 값의 리스트를 구함
  feat_values = [data[best_feature] for data in data_set]
  unique_vals = set(feat_values)

  for value in unique_vals:
    sub_labels = labels[:]
    p = float(feat_values.count(value)) / (len(data_set))
    son = ET.SubElement(parent, best_feature_label, {'value': value, "flag": "m", "p": str(round(p, 3))})

    my_tree[best_feature_label][value] = create_tree(split_data_set(data_set, best_feature, value), sub_labels, son)

  return my_tree

def calc_entropy(data_set):
  num_element = len(data_set) # 데이터 집합에 있는 사례 개수
  label_counts = {}
  # 가능한 모든 분류 항목에 대한 Dictionary 생성
  for feature_vector in data_set:
    current_label = feature_vector[-1]
    if current_label not in label_counts.keys():
      label_counts[current_label] = 0
    label_counts[current_label] += 1

  entropy = 0.0
  for key in label_counts:
    prob = float(label_counts[key]) / num_element
    entropy -= prob * math.log(prob, 2)
  return entropy

"""
  data_set : 분할하고자 하는 데이터 집합,
  axis    : 분할하고자 하는 속성
  value   : 반할할 속성의 값
"""
def split_data_set(data_set, axis, value):
  # 새로운 리스트 생성
  ret_data_set = []
  for feature_vector in data_set:
    # 속성을 잘라내고 데이터 집합을 분할
    if feature_vector[axis] == value:
      reduce_feature_vector = feature_vector[:axis]
      """
        a = [1, 2, 3]
        b = [4, 5, 6]
        a.append(b) : [1, 2, 3, [4, 5, 6]]
        a.extend(b) : [1, 2, 3, 4, 5, 6]
      """
      reduce_feature_vector.extend(feature_vector[axis+1:])
      ret_data_set.append(reduce_feature_vector)
  return ret_data_set

# 데이터 집합 분할하기
def choose_best_feature_to_split(data_set):
  num_features = len(data_set[0]) - 1 # 3
  base_entropy = calc_entropy(data_set)
  best_info_gain = 0.0
  best_feature = -1

  for i in range(num_features):
    feat_list = [data[i] for data in data_set]
    # 분류 항목 표시에 대해 중복이 없는 리스트 생성
    unique_vals = set(feat_list)
    new_entropy = 0.0

    # 각각의 분할을 위해 엔트로피 계산
    for value in unique_vals:
      sub_data_set = split_data_set(data_set, i, value)
      prob = len(sub_data_set) / float(len(data_set))
      new_entropy += prob * calc_entropy(sub_data_set)

    info_gain = base_entropy - new_entropy

    # 가장 큰 정보 이득 찾기
    if(info_gain > best_info_gain):
      best_info_gain = info_gain
      best_feature = i

  return best_feature



def majority_cnt(class_list):
  class_count = {}
  for vote in class_list:
    if vote not in class_count.keys():
      class_count[vote] = 0
    class_count[vote] += 1
  # 정렬 - 값에 따른 내림차순 python3에서는 iteritems -> items로 변경
  sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)

  # 가장 많이 가지고 있는 값의 label을 return
  return sorted_class_count[0][0]


def classify(input_tree, feature_labels, test_vector):
  # 오류 input_tree.keys()[0] -> list(input_tree.keys())[0]
  class_label = ''
  first_str = list(input_tree.keys())[0]
  second_dict = input_tree[first_str]

  # first key의 index 추출
  feature_index = feature_labels.index(first_str)

  for key in second_dict.keys():
    if test_vector[feature_index] == key:
      if type(second_dict[key]).__name__ == 'dict':
        class_label = classify(second_dict[key], feature_labels, test_vector)
      else:
        class_label = second_dict[key]
  return class_label

def predict(filename, testing_data_set):
  my_tree = grab_tree(filename)
  feature_labels = testing_data_set[0]
  prediction = []

  for i in range(1,len(testing_data_set)):
    answer = classify(my_tree, feature_labels, testing_data_set[i])
    prediction.append(answer)
  return prediction


def prettify(elem, level=0):
  i = "\n" + level*"  "
  if len(elem):
    if not elem.text or not elem.text.strip():
      elem.text = i + "  "
    for e in elem:
      prettify(e, level+1)
    if not e.tail or not e.tail.strip():
      e.tail = i
  if level and (not elem.tail or not elem.tail.strip()):
    elem.tail = i
  return elem

def store_tree(inputTree, filename):
  import pickle
  fw = open(filename, 'wb')
  pickle.dump(inputTree, fw)
  fw.close()

def grab_tree(filename):
  import pickle
  fr = open(filename, 'rb')
  return pickle.load(fr)
