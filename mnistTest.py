import numpy as np
from ml import Activation, Loss, WeightInitializer, BiasInitializer, Model, Optimizer

print("loading data")
trainingData = []
with open("mnist/train.txt", "r") as file:
  for line in file:
    line = line.strip().split(",")
    x = [int(pixel) / 255 for pixel in line[1:]]
    y = [0] * 10
    y[int(line[0])] = 1
    trainingData.append((x, y))
testingData = []
with open("mnist/test.txt", "r") as file:
  for line in file:
    line = line.strip().split(",")
    x = [int(pixel) / 255 for pixel in line[1:]]
    y = [0] * 10
    y[int(line[0])] = 1
    testingData.append((x, y))

model = Model(
  [784, 128, 10],
  Activation.ReLu(),
  Activation.SoftMax(),
  WeightInitializer.He(),
  BiasInitializer.Constant(0.1))
loss = Loss.CrossEntropy()
optimizer = Optimizer.Adam(model, 0.0003, 0.9, 0.999, 1e-8, 0.0001)

numEpochs = 5
batchSize = 32

print("training")
model.initialize()
for epoch in range(numEpochs):
  print(f" epoch: {epoch + 1}/{numEpochs}")
  np.random.shuffle(trainingData)
  for i in range(batchSize, len(trainingData), batchSize):
    batch = trainingData[i - batchSize : i]
    for j in range(0, batchSize):
      model.backPropagate(batch[j][0], batch[j][1], loss)
    optimizer.step()
    model.zeroGradients()

print("evaluating")
numCorrectTraining = 0
numCorrectTesting = 0
for i in range(len(trainingData)):
  numCorrectTraining += int(np.argmax(model.feedForward(trainingData[i][0])) == np.argmax(trainingData[i][1]))
for i in range(len(testingData)):
  numCorrectTesting += int(np.argmax(model.feedForward(testingData[i][0])) == np.argmax(testingData[i][1]))
print(f"training accuracy: {round(numCorrectTraining / len(trainingData) * 100, 2)}%")
print(f"testing accuracy: {round(numCorrectTesting / len(testingData) * 100, 2)}%")
