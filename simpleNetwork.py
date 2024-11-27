import math
import random

class Activation:
  class TanH:
    def compute(self, z):
      a = [0 for j in range(len(z))]
      for j in range(len(z)):
        e2 = math.exp(2 * z[j])
        a[j] = (e2 - 1) / (e2 + 1)
      return a

    def derivative(self, z):
      t = self.compute(z)
      return [1 - t[j] * t[j] for j in range(len(z))]

  class SoftMax:
    def compute(self, z):
      expSum = 0
      for j in range(len(z)):
        expSum += math.exp(z[j])
      return [math.exp(z[j]) / expSum for j in range(len(z))]

    def derivative(self, z):
      expSum = 0
      for j in range(len(z)):
        expSum += math.exp(z[j])
      d = [0 for j in range(len(z))]
      for j in range(len(z)):
        exp = math.exp(z[j])
        d[j] = (exp * expSum - exp * exp) / (expSum * expSum)
      return d

class Cost:
  class MeanSquaredError:
    def compute(self, a, y):
      cost = 0
      for j in range(len(a)):
        error = a[j] - y[j]
        cost += error * error
      return cost * 0.5

    def derivative(self, a, y):
      return [a[j] - y[j] for j in range(len(a))]

class Network:
  def __init__(self, layerSizes, activation, outputActivation, cost):
    self.layerSizes = layerSizes
    self.activation = activation
    self.outputActivation = outputActivation
    self.cost = cost
    self.numLayers = len(self.layerSizes)
    self.weights = [[[random.normalvariate(0, 1) / math.sqrt(self.layerSizes[l - 1]) for j in range(self.layerSizes[l])] for k in range(self.layerSizes[l - 1])] for l in range(1, self.numLayers)]
    self.biases = [[0 for j in range(self.layerSizes[l])] for l in range(0, self.numLayers)]
    self.gradientW = [[[0 for j in range(self.layerSizes[l])] for k in range(self.layerSizes[l - 1])] for l in range(1, self.numLayers)]
    self.gradientB = [[0 for j in range(self.layerSizes[l])] for l in range(0, self.numLayers)]

  def feedForward(self, a):
    for l in range(1, self.numLayers):
      z = [0 for j in range(self.layerSizes[l])]
      for j in range(self.layerSizes[l]):
        z[j] = self.biases[l][j]
        for k in range(self.layerSizes[l - 1]):
          z[j] += a[k] * self.weights[l - 1][k][j]
      a = self.activation.compute(z) if l != self.numLayers - 1 else self.outputActivation.compute(z)
    return a

  def backPropagate(self, a, y):
    zs = [None]
    activations = [a]
    for l in range(1, self.numLayers):
      z = [0 for j in range(self.layerSizes[l])]
      for j in range(self.layerSizes[l]):
        z[j] = self.biases[l][j]
        for k in range(self.layerSizes[l - 1]):
          z[j] += a[k] * self.weights[l - 1][k][j]
      a = self.activation.compute(z) if l != self.numLayers - 1 else self.outputActivation.compute(z)
      zs.append(z)
      activations.append(a)
    error = [[0 for j in range(self.layerSizes[l])] for l in range(0, self.numLayers)]
    aPrime = self.outputActivation.derivative(zs[self.numLayers - 1])
    cPrime = self.cost.derivative(activations[self.numLayers - 1], y)
    error[self.numLayers - 1] = [aPrime[j] * cPrime[j] for j in range(self.layerSizes[self.numLayers - 1])]
    for l in range(self.numLayers - 1, 1, -1):
      sum = [0 for k in range(self.layerSizes[l - 1])]
      for k in range(self.layerSizes[l - 1]):
        for j in range(self.layerSizes[l]):
          sum[k] += self.weights[l - 1][k][j] * error[l][j]
      aPrime = self.activation.derivative(zs[l - 1])
      error[l - 1] = [sum[k] * aPrime[k] for k in range(self.layerSizes[l - 1])]
    for l in range(1, self.numLayers):
      for k in range(self.layerSizes[l - 1]):
        for j in range(self.layerSizes[l]):
          self.gradientW[l - 1][k][j] = activations[l - 1][k] * error[l][j]
    for l in range(0, self.numLayers):
      for j in range(self.layerSizes[l]):
        self.gradientB[l][j] = error[l][j]

  def applyGradients(self, eta):
    for l in range(1, self.numLayers):
      for k in range(self.layerSizes[l - 1]):
        for j in range(self.layerSizes[l]):
          self.weights[l - 1][k][j] -= self.gradientW[l - 1][k][j] * eta
    for l in range(0, self.numLayers):
      for j in range(self.layerSizes[l]):
        self.biases[l][j] -= self.gradientB[l][j] * eta



network = Network(
  (1, 1),
  Activation.TanH(),
  Activation.TanH(),
  Cost.MeanSquaredError())
