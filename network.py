import math
import random
import matplotlib.pyplot as plt

class Activation:
  class ReLu:
    def compute(self, z):
      return [z[i] if z[i] > 0 else 0 for i in range(len(z))]
  
    def derivative(self, z):
      return [1 if z[i] > 0 else 0 for i in range(len(z))]

  class LeakyReLu:
    def __init__(self, eta):
      self.eta = eta

    def compute(self, z):
      return [z[i] if z[i] > 0 else z[i] * self.eta for i in range(len(z))]
  
    def derivative(self, z):
      return [1 if z[i] > 0 else self.eta for i in range(len(z))]

  class TanH:
    def compute(self, z):
      a = [0 for i in range(len(z))]
      for i in range(len(z)):
        e2 = math.exp(2 * z[i])
        a[i] = (e2 - 1) / (e2 + 1)
      return a

    def derivative(self, z):
      t = self.compute(z)
      return [1 - t[i] * t[i] for i in range(len(z))]

class Loss:
  class MeanSquaredError:
    def compute(self, x, y):
      loss = 0
      for i in range(len(x)):
        error = x[i] - y[i]
        loss += error ** 2
      return loss * 0.5

    def derivative(self, x, y):
      return [x[i] - y[i] for i in range(len(x))]

class Network:
  def __init__(self, layerSizes, activation):
    self.layerSizes = layerSizes
    self.activation = activation
    self.numLayers = len(self.layerSizes)
    self.weights = [[[random.normalvariate(0, 1) for k in range(self.layerSizes[l])] for j in range(self.layerSizes[l - 1])] for l in range(1, self.numLayers)]
    self.biases = [[0 for j in range(self.layerSizes[l])] for l in range(0, self.numLayers)]

  def feedForward(self, x):
    for l in range(1, self.numLayers):
      z = [0 for i in range(self.layerSizes[l])]
      for j in range(self.layerSizes[l]):
        z[j] = self.biases[l][j]
        for k in range(self.layerSizes[l - 1]):
          z[j] += x[k] * self.weights[l - 1][k][j]
      x = self.activation.compute(z)
    return x

  def backPropagate(self, x, y, loss):
    zs = [None]
    activations = [x]

    for l in range(1, self.numLayers):
      z = [0 for i in range(self.layerSizes[l])]
      for j in range(self.layerSizes[l]):
        z[j] = self.biases[l][j]
        for k in range(self.layerSizes[l - 1]):
          z[j] += x[k] * self.weights[l - 1][k][j]
      x = self.activation.compute(z)

      zs.append(z)
      activations.append(x)

    delta = [[0 for j in range(self.layerSizes[l])] for l in range(0, self.numLayers)]

    dLoss = loss.derivative(activations[self.numLayers - 1], y)
    sp = self.activation.derivative(zs[self.numLayers - 1])
    delta[self.numLayers - 1] = [dLoss[i] * sp[i] for i in range(self.layerSizes[self.numLayers - 1])]

    for l in range(self.numLayers - 2, 0, -1):
      sum = [0 for i in range(self.layerSizes[l])]
      for j in range(self.layerSizes[l]):
        for k in range(self.layerSizes[l + 1]):
          sum[j] += self.weights[l][j][k] * delta[l + 1][k]
      sp = self.activation.derivative(zs[l])
      delta[l] = [sum[i] * sp[i] for i in range(self.layerSizes[l])]

    nablaW = [[[0 for k in range(self.layerSizes[l])] for j in range(self.layerSizes[l - 1])] for l in range(1, self.numLayers)]
    nablaB = [[0 for j in range(self.layerSizes[l])] for l in range(0, self.numLayers)]

    for l in range(0, self.numLayers - 1):
      for j in range(self.layerSizes[l]):
        for k in range(self.layerSizes[l + 1]):
          nablaW[l][j][k] = activations[l][j] * delta[l + 1][k]
    for l in range(0, self.numLayers):
      for j in range(self.layerSizes[l]):
        nablaB[l][j] = delta[l][j]
    
    return (nablaW, nablaB)

  def train(self, loss, trainingData, epochs, eta):
    for i in range(epochs):
      for j in range(len(trainingData)):
        nablaW, nablaB = self.backPropagate(trainingData[j][0], trainingData[j][1], loss)

        for l in range(0, self.numLayers - 1):
          for j in range(self.layerSizes[l]):
            for k in range(self.layerSizes[l + 1]):
              self.weights[l][j][k] -= nablaW[l][j][k] * eta
        for l in range(0, self.numLayers):
          for j in range(self.layerSizes[l]):
            pass
            self.biases[l][j] -= nablaB[l][j] * eta



def shuffle(arr):
  n = len(arr)
  for i in range(n-1,0,-1):
    j = random.randint(0,i+1)
    arr[i],arr[j] = arr[j],arr[i]
  return arr

trainingData = [[[math.cos(i / 20 * math.tau) * 0.15 + (random.normalvariate(0, 0.1)), math.sin(i / 20 * math.tau) * 0.15 + (random.normalvariate(0, 0.1))], [-1]] for i in range(20)] + [[[math.cos(i/50 * math.tau) * 0.45 + (random.normalvariate(0, 0.1)), math.sin(i/50 * math.tau) * 0.45 + (random.normalvariate(0, 0.1))], [1]] for i in range(50)] + [[[math.cos(i / 80 * math.tau) * 0.75 + (random.normalvariate(0, 0.1)), math.sin(i / 80 * math.tau) * 0.75 + (random.normalvariate(0, 0.1))], [-1]] for i in range(80)]
trainingData = shuffle(trainingData)

network = Network((2, 8, 8, 1), Activation.TanH())
loss = Loss.MeanSquaredError()

network.train(loss, trainingData, 100, 0.05)

x0 = []
y0 = []
x1 = []
y1 = []

for i in range(80):
  for j in range(80):
    x = i / 40 - 1
    y = j / 40 - 1
    output = network.feedForward([x, y])[0]
    if output < 0:
      x0.append(x)
      y0.append(y)
    else:
      x1.append(x)
      y1.append(y)

x2 = []
y2 = []
x3 = []
y3 = []

for i in range(len(trainingData)):
  if (trainingData[i][1][0] < 0):
    x2.append(trainingData[i][0][0])
    y2.append(trainingData[i][0][1])
  else:
    x3.append(trainingData[i][0][0])
    y3.append(trainingData[i][0][1])

fig, ax = plt.subplots()
ax.plot(x0, y0, 'o')
ax.plot(x1, y1, 'o')
ax.plot(x2, y2, 'o')
ax.plot(x3, y3, 'o')
ax.set(xlim=(-1, 1), ylim=(-1, 1))
plt.show()
