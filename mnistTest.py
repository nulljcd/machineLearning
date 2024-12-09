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
  BiasInitializer.Zero())

loss = Loss.CrossEntropy()

optimizer = Optimizer.Adam(
  model=model,
  eta=0.0003,
  beta1=0.9,
  beta2=0.999,
  epsilon=1e-8,
  weightDecay=0.0001)


print("training")

model.initialize()

for epoch in range(5):
  print(f"  epoch: {epoch + 1}/5")
  np.random.shuffle(trainingData)
  for i in range(len(trainingData)):
    model.backPropagate(trainingData[i][0], trainingData[i][1], loss=loss)
    optimizer.step()
    model.zeroGradients()


print("evaluating")
score = 0
for i in range(len(trainingData)):
  score += int(np.argmax(model.feedForward(trainingData[i][0])) == np.argmax(trainingData[i][1]))
print(f"training score: {round(score / len(trainingData) * 100, 2)}%")

score = 0
for i in range(len(testingData)):
  score += int(np.argmax(model.feedForward(testingData[i][0])) == np.argmax(testingData[i][1]))
print(f"testing score: {round(score / len(testingData) * 100, 2)}%")