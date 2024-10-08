import math
import random

class Activation:
  class ReLu:
    def compute(self, z):
      return [z[i] if z[i] > 0 else 0 for i in range(len(z))]
  
    def derivative(self, z):
      return [1 if z[i] > 0 else 0 for i in range(len(z))]

  class LeakyReLu:
    def __init__(self, alpha):
      self.alpha = alpha

    def compute(self, z):
      return [z[i] if z[i] > 0 else z[i] * self.alpha for i in range(len(z))]
  
    def derivative(self, z):
      return [1 if z[i] > 0 else self.alpha for i in range(len(z))]

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

  class Sigmoid:
    def compute(self, z):
      return [1 / (1 + math.exp(-z[i])) for i in range(len(z))]

    def derivative(self, z):
      a = [1 / (1 + math.exp(-z[i])) for i in range(len(z))]
      return [a[i] * (1 - a[i]) for i in range(len(z))]

class Loss:
  class MeanSquaredError:
    def compute(self, p, e):
      loss = 0
      for i in range(len(p)):
        error = p[i] - e[i]
        loss += error ** 2
      return loss * 0.5

    def derivative(self, p, e):
      return [p[i] - e[i] for i in range(len(p))]

class Network:
  def __init__(self, layerSizes, activation):
    self.layerSizes = layerSizes
    self.activation = activation
    self.numLayers = len(self.layerSizes)
    self.weights = [[[random.normalvariate(0, 1) for k in range(self.layerSizes[l])] for j in range(self.layerSizes[l - 1])] for l in range(1, self.numLayers)]
    self.biases = [[0 for j in range(self.layerSizes[l])] for l in range(0, self.numLayers)]
    self.lossGradientW = [[[0 for k in range(self.layerSizes[l])] for j in range(self.layerSizes[l - 1])] for l in range(1, self.numLayers)]
    self.lossGradientB = [[0 for j in range(self.layerSizes[l])] for l in range(0, self.numLayers)]

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

    error = [[0 for j in range(self.layerSizes[l])] for l in range(0, self.numLayers)]

    dLoss = loss.derivative(activations[self.numLayers - 1], y)
    sp = self.activation.derivative(zs[self.numLayers - 1])
    error[self.numLayers - 1] = [dLoss[i] * sp[i] for i in range(self.layerSizes[self.numLayers - 1])]

    for l in range(self.numLayers - 2, 0, -1):
      sum = [0 for i in range(self.layerSizes[l])]
      for j in range(self.layerSizes[l]):
        for k in range(self.layerSizes[l + 1]):
          sum[j] += self.weights[l][j][k] * error[l + 1][k]
      sp = self.activation.derivative(zs[l])
      error[l] = [sum[i] * sp[i] for i in range(self.layerSizes[l])]

    for l in range(0, self.numLayers - 1):
      for j in range(self.layerSizes[l]):
        for k in range(self.layerSizes[l + 1]):
          self.lossGradientW[l][j][k] = activations[l][j] * error[l + 1][k]
    for l in range(0, self.numLayers):
      for j in range(self.layerSizes[l]):
        self.lossGradientB[l][j] = error[l][j]

  def applyGradients(self, alpha):
    for l in range(0, self.numLayers - 1):
      for j in range(self.layerSizes[l]):
        for k in range(self.layerSizes[l + 1]):
          self.weights[l][j][k] -= self.lossGradientW[l][j][k] * alpha
    for l in range(0, self.numLayers):
      for j in range(self.layerSizes[l]):
        self.biases[l][j] -= self.lossGradientB[l][j] * alpha

  def getParams(self):
    params = []
    for l in range(0, self.numLayers - 1):
      for j in range(self.layerSizes[l]):
        for k in range(self.layerSizes[l + 1]):
          params.append(self.weights[l][j][k])
    for l in range(0, self.numLayers):
      for j in range(self.layerSizes[l]):
        params.append(self.biases[l][j])
    return params

  def setParams(self, params):
    i = 0
    for l in range(0, self.numLayers - 1):
      for j in range(self.layerSizes[l]):
        for k in range(self.layerSizes[l + 1]):
          self.weights[l][j][k] = params[i]
          i += 1
    for l in range(0, self.numLayers):
      for j in range(self.layerSizes[l]):
        self.biases[l][j] = params[i]
        i += 1
    return params