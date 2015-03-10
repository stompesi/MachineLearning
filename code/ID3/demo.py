import id3, csv
if __name__=="__main__":

  training_data_set = []

  with open('./data/training_set.csv', newline='') as f:
    reader = csv.reader(f)
    for line in reader:
      training_data_set.append(line[:])

  id3.train(training_data_set, "DecisionTree.xml")

  test_data_set = []
  answer = []

  with open('./data/testing_set.csv', newline='') as f:
    reader = csv.reader(f)
    for line in reader:
      test_data_set.append(line[:])
      answer.append(line[-1])
  answer.pop(0)

  prediction = id3.predict("model.txt", test_data_set)
  err = 0

  for i in range(len(answer)):
    if not answer[i] == prediction[i]:
      err = err + 1

  print "error rate=", round(float(err) / len(prediction) * 100, 2), "%"

