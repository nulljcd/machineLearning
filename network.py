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

    class CrossEntropy:
      pass

class Network:
  def __init__(self, layerSizes, activation):
    self.layerSizes = layerSizes
    self.activation = activation
    self.numLayers = len(self.layerSizes)
    self.weights = [[[random.normalvariate(0, 1) for k in range(self.layerSizes[l])] for j in range(self.layerSizes[l - 1])] for l in range(1, self.numLayers)]
    self.biases = [[0 for j in range(self.layerSizes[l])] for l in range(0, self.numLayers)]
    self.nablaW = [[[0 for k in range(self.layerSizes[l])] for j in range(self.layerSizes[l - 1])] for l in range(1, self.numLayers)]
    self.nablaB = [[0 for j in range(self.layerSizes[l])] for l in range(0, self.numLayers)]

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

    for l in range(0, self.numLayers - 1):
      for j in range(self.layerSizes[l]):
        for k in range(self.layerSizes[l + 1]):
          self.nablaW[l][j][k] = activations[l][j] * delta[l + 1][k]
    for l in range(0, self.numLayers):
      for j in range(self.layerSizes[l]):
        self.nablaB[l][j] = delta[l][j]

  def applyGradients(self, alpha):
    for l in range(0, self.numLayers - 1):
      for j in range(self.layerSizes[l]):
        for k in range(self.layerSizes[l + 1]):
          self.weights[l][j][k] -= self.nablaW[l][j][k] * alpha
    for l in range(0, self.numLayers):
      for j in range(self.layerSizes[l]):
        self.biases[l][j] -= self.nablaB[l][j] * alpha

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














# x0 = [math.cos(i) * 0.8 + random.normalvariate(0, 0.1) for i in range(350)] + [random.normalvariate(0, 0.12) for i in range(50)]
# y0 = [math.sin(i) * 0.8 + random.normalvariate(0, 0.1) for i in range(350)] + [random.normalvariate(0, 0.12) for i in range(50)]
# x1 = [math.cos(i) * 0.45 + random.normalvariate(0, 0.08) for i in range(200)]
# y1 = [math.sin(i) * 0.45 + random.normalvariate(0, 0.08) for i in range(200)]

trainingDataCircle600pts = [[[math.cos(i) * 0.8 + random.normalvariate(0, 0.1), math.sin(i) * 0.8 + random.normalvariate(0, 0.1)], [-1]] for i in range(350)] + \
  [[[random.normalvariate(0, 0.1), random.normalvariate(0, 0.1)], [-1]] for i in range(50)] + \
  [[[math.cos(i) * 0.45 + random.normalvariate(0, 0.08), math.sin(i) * 0.45 + random.normalvariate(0, 0.08)], [1]] for i in range(200)]

network = Network((2, 26, 1), Activation.TanH())
loss = Loss.MeanSquaredError()

for i in range(1000):
  for j in range(600):
    network.backPropagate(trainingDataCircle600pts[j][0], trainingDataCircle600pts[j][1], loss)
    network.applyGradients(0.05)

x0 = []
y0 = []
x1 = []
y1 = []

for i in range(40):
  for j in range(40):
    x = i / 20 - 1
    y = j / 20 - 1
    output = network.feedForward([x, y])[0]
    if output < 0:
      x0.append(x)
      y0.append(y)
    else:
      x1.append(x)
      y1.append(y)

fig, ax = plt.subplots()
ax.plot(x0, y0, '.')
ax.plot(x1, y1, '.')
ax.set(xlim=(-1, 1), ylim=(-1, 1))
plt.show()
