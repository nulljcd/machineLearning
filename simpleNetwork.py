import math
import random

class Activation:
  class ReLu:
    def __call__(self, z):
      return [z[i] if z[i] > 0 else 0 for i in range(len(z))]
  
    def derivative(self, z):
      return [1 if z[i] > 0 else 0 for i in range(len(z))]

  class LeakyReLu:
    def __init__(self, alpha):
      self.alpha = alpha

    def __call__(self, z):
      return [z[i] if z[i] > 0 else z[i] * self.alpha for i in range(len(z))]
  
    def derivative(self, z):
      return [1 if z[i] > 0 else self.alpha for i in range(len(z))]

  class TanH:
    def __call__(self, z):
      a = []
      for i in range(len(z)):
        e2 = math.exp(2 * z[i])
        a.append((e2 - 1) / (e2 + 1))
      return a

    def derivative(self, z):
      t = self(z)
      return [1 - t[i] * t[i] for i in range(len(z))]

class Loss:
  class MeanSquaredError:
    def __call__(self, p, e):
      loss = 0
      for outputIndex in range(len(p)):
        delta = p[outputIndex] - e[outputIndex]
        loss += delta ** 2
      return loss * 0.5

    def derivative(self, p, e):
      return [p[i] - e[i] for i in range(len(p))]

class Network:
  def __init__(self, layerSizes, activation):
    self.layerSizes = layerSizes
    self.activation = activation
    self.numLayers = len(self.layerSizes)
    self.weights = [[[random.normalvariate(0, 1) for k in range(self.layerSizes[i])] for j in range(self.layerSizes[i - 1])] for i in range(1, self.numLayers)]
    self.biases = [[0 for j in range(self.layerSizes[i])] for i in range(1, self.numLayers)]

  def feedForward(self, x):
    for l in range(1, self.numLayers):
      z = [0 for i in range(self.layerSizes[l])]
      for j in range(self.layerSizes[l]):
        z[j] = self.biases[l - 1][j]
        for k in range(self.layerSizes[l - 1]):
          z[j] += x[k] * self.weights[l - 1][k][j]
      x = self.activation(z)
    return x

  def backPropagate(self, x, y, loss):
    # feed forward
    zs = []
    activations = [x]
    for l in range(1, self.numLayers):
      z = [0 for i in range(self.layerSizes[l])]
      for j in range(self.layerSizes[l]):
        z[j] = self.biases[l - 1][j]
        for k in range(self.layerSizes[l - 1]):
          z[j] += x[k] * self.weights[l - 1][k][j]
      zs.append(z)
      x = self.activation(z)
      activations.append(x)

    delta = [[0 for j in range(self.layerSizes[i])] for i in range(self.numLayers)]

    # calculate ouput loss
    dLoss = loss.derivative(activations[self.numLayers - 1], y)
    sp = self.activation.derivative(zs[self.numLayers - 2])
    delta[self.numLayers - 1] = [dLoss[i] * sp[i] for i in range(self.layerSizes[self.numLayers - 1])]

    # back propagate
    for l in range(self.numLayers - 2, 0, -1):
      sum = [0 for i in range(self.layerSizes[l])]
      for j in range(self.layerSizes[l]):
        for k in range(self.layerSizes[l + 1]):
          sum[j] += self.weights[l][j][k] * delta[l + 1][k]
      sp = self.activation.derivative(zs[l - 1])
      delta[l] = [sum[i] * sp[i] for i in range(self.layerSizes[l])]

    # calculate the gradients from delta
    gradientW = [[[0 for k in range(self.layerSizes[i])] for j in range(self.layerSizes[i - 1])] for i in range(1, self.numLayers)]
    gradientB = [[0 for j in range(self.layerSizes[i])] for i in range(1, self.numLayers)]
    for l in range(0, self.numLayers - 1):
      for j in range(self.layerSizes[l]):
        for k in range(self.layerSizes[l + 1]):
          gradientW[l][j][k] = activations[l][j] * delta[l + 1][k]
        gradientB[l - 1][j] = delta[l][j]

    return (gradientW, gradientB)

  def applyGradients(self, gradients, alpha):
    gradientW, gradientB = gradients
    for l in range(0, self.numLayers - 1):
      for j in range(self.layerSizes[l]):
        for k in range(self.layerSizes[l + 1]):
          self.weights[l][j][k] -= gradientW[l][j][k] * alpha
        self.biases[l - 1][j] -= gradientB[l - 1][j] * alpha



# simple test
trainingData = (
  ((0, 0), (.4, .6)),
  ((1, 0), (.6, .4)),
  ((0, 1), (.6, .4)),
  ((1, 1), (.4, .6)))

n = Network((2, 6, 2), Activation.TanH())

loss = Loss.MeanSquaredError()

print('training')
for i in range(800):
  totalLoss = 0
  for j in range(4):
    gradients = n.backPropagate(trainingData[j][0], trainingData[j][1], loss)
    n.applyGradients(gradients, 0.3)
  if i % 20 == 0:
    totalLoss = 0
    for i in range(4):
      outputs = n.feedForward(trainingData[i][0])
      totalLoss += loss(outputs, trainingData[i][1])
    totalLoss /= 4
    print(f'average loss: {totalLoss}')

print('evaluation')
totalLoss = 0
for i in range(4):
  outputs = n.feedForward(trainingData[i][0])
  totalLoss += loss(outputs, trainingData[i][1])
  print(f'input: {trainingData[i][0]} predicted: {outputs} actual: {trainingData[i][1]}')
totalLoss /= 4
print(f'average loss: {totalLoss}')