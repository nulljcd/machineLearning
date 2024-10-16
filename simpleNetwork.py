import math
import random
import matplotlib.pyplot as plt

class Activation:
  class ReLu:
    def compute(self, z):
      return [z[j] if z[j] > 0 else 0 for j in range(len(z))]
  
    def derivative(self, z):
      return [1 if z[j] > 0 else 0 for j in range(len(z))]

  class LeakyReLu:
    def __init__(self, alpha):
      self.alpha = alpha

    def compute(self, z):
      return [z[j] if z[j] > 0 else z[j] * self.alpha for j in range(len(z))]
  
    def derivative(self, z):
      return [1 if z[j] > 0 else self.alpha for j in range(len(z))]

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
        cost += error ** 2

      return cost * 0.5

    def derivative(self, a, y):
      return [a[j] - y[j] for j in range(len(a))]

  class CrossEntropy:
    def __init__(self, epsilon):
      self.epsilon = epsilon

    def compute(self, a, y):
      cost = 0
      for j in range(len(a)):
        a[j] = max(min(a[j], 1 - self.epsilon), self.epsilon)
        cost -= y[j] * math.log(a[j] + self.epsilon)
      cost /= len(a)

      return cost

    def derivative(self, a, y):
      d = [0 for j in range(len(a))]
      for j in range(len(a)):
        if a[j] == 0 or a[j] == 1:
          d[j] = 0
        else:
          d[j] = (-a[j] + y[j]) / (a[j] * (a[j] - 1))

      return d
 
class Network:
  def __init__(self, layerSizes, activation, ouputActivation):
    self.layerSizes = layerSizes
    self.activation = activation
    self.ouputActivation = ouputActivation
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
      a = self.activation.compute(z) if l != self.numLayers - 1 else self.ouputActivation.compute(z)
    
    return a

  def backPropagate(self, x, y, cost):
    zs = [None]
    a = x
    activations = [a]

    for l in range(1, self.numLayers):
      z = [0 for i in range(self.layerSizes[l])]
      for j in range(self.layerSizes[l]):
        z[j] = self.biases[l][j]
        for k in range(self.layerSizes[l - 1]):
          z[j] += a[k] * self.weights[l - 1][k][j]
      a = self.activation.compute(z) if l != self.numLayers - 1 else self.ouputActivation.compute(z)

      zs.append(z)
      activations.append(a)

    delta = [[0 for j in range(self.layerSizes[l])] for l in range(0, self.numLayers)]

    az = self.ouputActivation.derivative(zs[self.numLayers - 1])
    ca = cost.derivative(activations[self.numLayers - 1], y)
    delta[self.numLayers - 1] = [az[j] * ca[j] for j in range(self.layerSizes[self.numLayers - 1])]

    for l in range(self.numLayers - 1, 1, -1):
      sum = [0 for i in range(self.layerSizes[l - 1])]
      for k in range(self.layerSizes[l - 1]):
        for j in range(self.layerSizes[l]):
          zal = self.weights[l - 1][k][j]
          sum[k] += zal * delta[l][j]
      az = self.activation.derivative(zs[l - 1])
      delta[l - 1] = [sum[k] * az[k] for k in range(self.layerSizes[l - 1])]

    nablaW = [[[0 for j in range(self.layerSizes[l])] for k in range(self.layerSizes[l - 1])] for l in range(1, self.numLayers)]
    nablaB = [[0 for j in range(self.layerSizes[l])] for l in range(0, self.numLayers)]

    for l in range(1, self.numLayers):
      for k in range(self.layerSizes[l - 1]):
        for j in range(self.layerSizes[l]):
          nablaW[l - 1][k][j] = activations[l - 1][k] * delta[l][j]
    for l in range(0, self.numLayers):
      for j in range(self.layerSizes[l]):
        nablaB[l][j] = delta[l][j]
    
    return (nablaW, nablaB)

  def applyGradients(self, gradients, eta):
    nablaW, nablaB = gradients

    for l in range(1, self.numLayers):
      for k in range(self.layerSizes[l - 1]):
        for j in range(self.layerSizes[l]):
          self.weights[l - 1][k][j] -= nablaW[l - 1][k][j] * eta
    for l in range(0, self.numLayers):
      for j in range(self.layerSizes[l]):
        self.biases[l][j] -= nablaB[l][j] * eta







def shuffle(array):
  for i in range(len(array)-2, 0, -1):
      j = random.randint(0, i + 1)
      array[i], array[j] = array[j], array[i]
  return array

trainingData = [[[math.cos(i / 100 * math.tau) * 0.8 + random.normalvariate(0, 0.1), math.sin(i / 100 * math.tau) * 0.8 + random.normalvariate(0, 0.1)], [0, 1]] for i in range(150)] +\
  [[[math.cos(i / 20 * math.tau) * 0.1 + random.normalvariate(0, 0.1), math.sin(i / 20 * math.tau) * 0.1 + random.normalvariate(0, 0.1)], [0, 1]] for i in range(50)] +\
  [[[math.cos(i / 20 * math.tau) * 0.45 + random.normalvariate(0, 0.1), math.sin(i / 20 * math.tau) * 0.45 + random.normalvariate(0, 0.1)], [1, 0]] for i in range(100)]

network = Network([2, 8, 8, 2], Activation.TanH(), Activation.SoftMax())
cost = Cost.MeanSquaredError()

for i in range(100):
  trainingData = shuffle(trainingData)
  for j in range(len(trainingData)):
    gradients = network.backPropagate(trainingData[j][0], trainingData[j][1], cost)
    network.applyGradients(gradients, 0.1)

fig, ax = plt.subplots()

x0 = []
y0 = []
x1 = []
y1 = []

for i in range(len(trainingData)):
  if (trainingData[i][1][0] == 0):
    x0.append(trainingData[i][0][0])
    y0.append(trainingData[i][0][1])
  else:
    x1.append(trainingData[i][0][0])
    y1.append(trainingData[i][0][1])

x2 = []
y2 = []
x3 = []
y3 = []

for i in range(60):
  for j in range(60):
    x = ((i + 0.5) / 30 - 1)
    y = ((j + 0.5) / 30 - 1)
    outputs = network.feedForward([x, y])
    if outputs[0] > outputs[1]:
      x2.append(x)
      y2.append(y)
    else:
      x3.append(x)
      y3.append(y)

ax.plot(x2, y2, 's')
ax.plot(x3, y3, 's')
ax.plot(x0, y0, '.')
ax.plot(x1, y1, '.')

ax.set(xlim=(-1, 1), ylim=(-1, 1))

plt.show()
