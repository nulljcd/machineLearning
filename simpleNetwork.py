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

class Cost:
  class MeanSquaredError:
    def compute(self, a, y):
      cost = 0

      for i in range(len(a)):
        error = a[i] - y[i]
        cost += error ** 2

      return cost * 0.5

    def derivative(self, a, y):
      return [a[i] - y[i] for i in range(len(a))]

class Network:
  def __init__(self, layerSizes, activation):
    self.layerSizes = layerSizes
    self.activation = activation
    self.numLayers = len(self.layerSizes)
    self.weights = [[[random.normalvariate(0, 1) / math.sqrt(self.layerSizes[l - 1]) for j in range(self.layerSizes[l])] for k in range(self.layerSizes[l - 1])] for l in range(1, self.numLayers)]
    self.biases = [[0 for j in range(self.layerSizes[l])] for l in range(0, self.numLayers)]

  def feedForward(self, a):
    for l in range(1, self.numLayers):
      z = [0 for i in range(self.layerSizes[l])]
      for j in range(self.layerSizes[l]):
        z[j] = self.biases[l][j]
        for k in range(self.layerSizes[l - 1]):
          z[j] += a[k] * self.weights[l - 1][k][j]
      a = self.activation.compute(z)
    
    return a

  def backPropagate(self, x, y, cost):
    # feed forward
    zs = [None]
    a = x
    activations = [a]

    for l in range(1, self.numLayers):
      z = [0 for i in range(self.layerSizes[l])]
      for j in range(self.layerSizes[l]):
        z[j] = self.biases[l][j]
        for k in range(self.layerSizes[l - 1]):
          z[j] += a[k] * self.weights[l - 1][k][j]
      a = self.activation.compute(z)

      zs.append(z)
      activations.append(a)

    # delta == (del c / del a last)
    delta = [[0 for j in range(self.layerSizes[l])] for l in range(0, self.numLayers)]

    # last layer delta
    # partial derivative -> del a / del z
    az = self.activation.derivative(zs[self.numLayers - 1])
    # partial derivative -> del c / del a
    ca = cost.derivative(activations[self.numLayers - 1], y)
    # chain rule -> (del a / del z) (del c / del a)
    delta[self.numLayers - 1] = [az[j] * ca[j] for j in range(self.layerSizes[self.numLayers - 1])]

    # back propagation
    for l in range(self.numLayers - 1, 1, -1):
      sum = [0 for i in range(self.layerSizes[l - 1])]
      for k in range(self.layerSizes[l - 1]):
        for j in range(self.layerSizes[l]):
          # partial derivative -> del z / del a last
          zal = self.weights[l - 1][k][j]
          sum[k] += zal * delta[l][j]
      # partial derivative -> del a / del z
      az = self.activation.derivative(zs[l - 1])
      # chain rule -> (del z / del a last) (del a / del z) (del c / del a)
      delta[l - 1] = [sum[k] * az[k] for k in range(self.layerSizes[l - 1])]

    gradientW = [[[0 for j in range(self.layerSizes[l])] for k in range(self.layerSizes[l - 1])] for l in range(1, self.numLayers)]
    gradientB = [[0 for j in range(self.layerSizes[l])] for l in range(0, self.numLayers)]

    for l in range(1, self.numLayers):
      for k in range(self.layerSizes[l - 1]):
        for j in range(self.layerSizes[l]):
          # partial derivative -> del z / del w
          zw = activations[l - 1][k]
          # del c / del w
          gradientW[l - 1][k][j] = zw * delta[l][j]
    for l in range(0, self.numLayers):
      for j in range(self.layerSizes[l]):
        # partial derivative -> del z / del b
        zb = 1
        # del c / del b
        gradientB[l][j] = zb * delta[l][j]
    
    return (gradientW, gradientB)
