import pyC45, csv
if __name__=="__main__":
  run()

def run():
  training_obs = [] # 각 속성 값
  training_cat = [] # 분류 하고자하는 데이터

  with open('./data/training_set.csv', newline='') as f:
    reader = csv.reader(f)
    for line in reader:
      training_obs.append(line[:-1])
      training_cat.append(line[-1])

  pyC45.train(training_obs, training_cat, "DecisionTree.xml")

  #test the C45 decision tree
  test_data_set = []
  answer = []

  with open('./data/testing_set.csv', newline='') as f:
    reader = csv.reader(f)
    for line in reader:
      test_data_set.append(line[:-1])
      answer.append(line[-1])
  answer.pop(0)


  prediction = pyC45.predict("DecisionTree.xml", test_data_set)
  err = 0

  for i in range(len(answer)):
    if not answer[i] == prediction[i]:
      err = err + 1

  print ("error rate=", round(float(err) / len(prediction) * 100, 2), "%")
