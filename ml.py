import numpy as np

class Activation:
  class ReLu:
    def compute(self, z):
      return np.maximum(0.0, z)

    def derivative(self, z):
      return (z > 0).astype(float)
    
  class LeakyReLu:
    def __init__(self, alpha):
      self.alpha = alpha

    def compute(self, z):
      return np.maximum(z * self.alpha, z)

    def derivative(self, z):
      return np.where(z > 0, 1.0, self.alpha)

  class SoftMax:
    def compute(self, z):
      expValues = np.exp(z - np.max(z))
      return expValues / np.sum(expValues)

    def derivative(self, z):
      expValues = np.exp(z - np.max(z))
      sumExp = np.sum(expValues)
      return (expValues * (sumExp - expValues)) / (sumExp ** 2)


class Loss:
  class MeanSquaredError:
    def compute(self, a, y):
      return 0.5 * np.sum((a - y) ** 2)

    def derivative(self, a, y):
      return a - y

  class CrossEntropy:
    def compute(self, a, y):
      a = np.clip(a, 1e-8, 1 - 1e-8)
      loss = -np.sum(y * np.log(a))
      return loss

    def derivative(self, a, y):
      return a - y


class WeightInitializer:
  class He:
    def set(self, weights, layerSizes):
      for l in range(1, len(layerSizes)):
        weights[l - 1] = np.random.normal(0, 1 / np.sqrt(layerSizes[l]), (layerSizes[l - 1], layerSizes[l]))

  class Glorot:
    def set(self, weights, layerSizes):
      for l in range(1, len(layerSizes)):
        weights[l - 1] = np.random.normal(0, 1 / np.sqrt(layerSizes[l - 1]), (layerSizes[l - 1], layerSizes[l]))


class BiasInitializer:
  class Zero:
    def set(self, biases, layerSizes):
      for l in range(1, len(layerSizes)):
        biases[l - 1] = np.zeros(layerSizes[l])

  class Constant:
    def __init__(self, value):
      self.value = value

    def set(self, biases, layerSizes):
      for l in range(1, len(layerSizes)):
        biases[l - 1] = np.ones(layerSizes[l]) * self.value


class Model:
  def __init__(self, layerSizes, activation, outputActivation, weightInitializer, biasInitializer):
    self.layerSizes = layerSizes
    self.activation = activation
    self.outputActivation = outputActivation
    self.weightInitializer = weightInitializer
    self.biasInitializer = biasInitializer

    self.numLayers = len(layerSizes)
    self.weights = [np.zeros((layerSizes[l - 1], layerSizes[l])) for l in range(1, self.numLayers)]
    self.biases = [np.zeros(layerSizes[l]) for l in range(1, self.numLayers)]
    self.gradientW = [np.zeros_like(w) for w in self.weights]
    self.gradientB = [np.zeros_like(b) for b in self.biases]

  def initialize(self):
    self.weightInitializer.set(self.weights, self.layerSizes)
    self.biasInitializer.set(self.biases, self.layerSizes)

  def feedForward(self, x):
    a = x
    for l in range(1, self.numLayers):
      z = np.dot(a, self.weights[l - 1]) + self.biases[l - 1]
      a = self.activation.compute(z) if l != self.numLayers - 1 else self.outputActivation.compute(z)
    return a

  def backPropagate(self, x, y, loss):
    a = x
    zs = [None] * self.numLayers
    activations = [None] * self.numLayers
    activations[0] = a
    for l in range(1, self.numLayers):
      z = np.dot(a, self.weights[l - 1]) + self.biases[l - 1]
      a = self.activation.compute(z) if l != self.numLayers - 1 else self.outputActivation.compute(z)
      zs[l] = z
      activations[l] = a
    error = [None] * self.numLayers
    error[self.numLayers - 1] = loss.derivative(activations[self.numLayers - 1], y) * self.outputActivation.derivative(zs[self.numLayers - 1])
    for l in range(self.numLayers - 2, 0, -1):
      error[l] = np.dot(error[l + 1], self.weights[l].T) * self.activation.derivative(zs[l])
    for l in range(1, self.numLayers):
      self.gradientW[l - 1] += np.outer(activations[l - 1], error[l])
      self.gradientB[l - 1] += error[l]

  def zeroGradients(self):
    self.gradientW = [np.zeros_like(w) for w in self.weights]
    self.gradientB = [np.zeros_like(b) for b in self.biases]


class Optimizer:
  class Adam:
    def __init__(self, model, eta, beta1, beta2, epsilon, weightDecay):
      self.model = model
      self.eta = eta
      self.beta1 = beta1
      self.beta2 = beta2
      self.epsilon = epsilon
      self.weightDecay = weightDecay

      self.mW = [np.zeros_like(w) for w in self.model.weights]
      self.vW = [np.zeros_like(w) for w in self.model.weights]
      self.mB = [np.zeros_like(b) for b in self.model.biases]
      self.vB = [np.zeros_like(b) for b in self.model.biases]
      self.t = 0

    def step(self):
      self.t += 1
      for l in range(1, self.model.numLayers):
        gradientW = self.model.gradientW[l - 1] + self.weightDecay * self.model.weights[l - 1]
        self.mW[l - 1] = self.beta1 * self.mW[l - 1] + (1 - self.beta1) * gradientW
        self.vW[l - 1] = self.beta2 * self.vW[l - 1] + (1 - self.beta2) * (gradientW ** 2)
        mHatW = self.mW[l - 1] / (1 - self.beta1 ** self.t)
        vHatW = self.vW[l - 1] / (1 - self.beta2 ** self.t)
        self.model.weights[l - 1] -= self.eta * mHatW / (np.sqrt(vHatW) + self.epsilon)
        gradientB = self.model.gradientB[l - 1]
        self.mB[l - 1] = self.beta1 * self.mB[l - 1] + (1 - self.beta1) * gradientB
        self.vB[l - 1] = self.beta2 * self.vB[l - 1] + (1 - self.beta2) * (gradientB ** 2)
        mHatB = self.mB[l - 1] / (1 - self.beta1 ** self.t)
        vHatB = self.vB[l - 1] / (1 - self.beta2 ** self.t)
        self.model.biases[l - 1] -= self.eta * mHatB / (np.sqrt(vHatB) + self.epsilon)
