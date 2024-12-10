import numpy as np
from ml import Activation, Loss, WeightInitializer, BiasInitializer, Model, Optimizer

print("loading data")
trainData = np.loadtxt("mnist/train.txt", delimiter=',', dtype=int)
xTrain = trainData[:, 1:] / 255.0
yTrain = np.zeros((trainData.shape[0], 10), dtype=int)
yTrain[np.arange(trainData.shape[0]), trainData[:, 0]] = 1
trainData = np.array([(xTrain[i], yTrain[i]) for i in range(xTrain.shape[0])], dtype=object)
testData = np.loadtxt("mnist/test.txt", delimiter=',', dtype=int)
xTest = testData[:, 1:] / 255.0
yTest = np.zeros((testData.shape[0], 10), dtype=int)
yTest[np.arange(testData.shape[0]), testData[:, 0]] = 1
testData = np.array([(xTest[i], yTest[i]) for i in range(xTest.shape[0])], dtype=object)

model = Model(
  [784, 128, 10],
  Activation.ReLu(),
  Activation.SoftMax(),
  WeightInitializer.He(),
  BiasInitializer.Constant(0.1))
loss = Loss.CrossEntropy()
optimizer = Optimizer.Adam(model, 0.0003, 0.9, 0.999, 1e-8, 0.0001)
model.initialize()

print("training")
numEpochs = 5
batchSize = 32
for epoch in range(numEpochs):
  print(f" epoch: {epoch + 1}/{numEpochs}")
  indices = np.random.permutation(len(trainData))
  trainDataShuffled = trainData[indices]
  for i in range(0, len(trainDataShuffled), batchSize):
    batch = trainDataShuffled[i:i + batchSize]
    for x, y in batch:
        model.backPropagate(x, y, loss)
    optimizer.step()
    model.zeroGradients()

print("evaluating")
numCorrectTrain = 0
numCorrectTest = 0
trainPredictions = np.argmax([model.feedForward(x) for x, _ in trainData], axis=1)
trainLabels = np.argmax([y for _, y in trainData], axis=1)
numCorrectTrain = np.sum(trainPredictions == trainLabels)
testPredictions = np.argmax([model.feedForward(x) for x, _ in testData], axis=1)
testLabels = np.argmax([y for _, y in testData], axis=1)
numCorrectTest = np.sum(testPredictions == testLabels)
print(f"training accuracy: {round(numCorrectTrain / len(trainData) * 100, 2)}%")
print(f"testing accuracy: {round(numCorrectTest / len(testData) * 100, 2)}%")
