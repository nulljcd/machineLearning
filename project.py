# %% [markdown]
# ### imports
# all imports
# Note: only uses native python imports

# %%
import math
import random

# %% [markdown]
# ### initializer
# used for initializing the hyper parameters before training
# - zero
# - one
# - random normal distribution

# %%
class Initializer:
  @staticmethod
  class Zero:
    def calculateValue(self):
      return 0
      
    def setValues(self, values):
      for i in range(len(values)):
        values[i] = self.calculateValue()

  @staticmethod
  class One:
    def calculateValue(self):
      return 1

    def setValues(self, values):
      for i in range(len(values)):
        values[i] = self.calculateValue()

  @staticmethod
  class RandomNormalDistribution:
    def __init__(self, mean, stddev):
      self.mean = mean
      self.stddev = stddev

    def calculateValue(self):
      return random.normalvariate(self.mean, self.stddev)

    def setValues(self, values):
      for i in range(len(values)):
        values[i] = self.calculateValue()

# %% [markdown]
# ### activation
# the activation function and derivative
# - linear
# - sigmoid
# - relu
# - leaky relu
# - tanh
# - silu
# - softmax

# %%
class Activation:
  @staticmethod
  class Linear:
    def calculate(self, inputs, index):
      return inputs[index]
      
    def derivative(self, inputs, index):
      return 1

  @staticmethod
  class Sigmoid:
    def calculate(self, inputs, index):
      return 1 / (1 + math.exp(-inputs[index]))

    def derivative(self, inputs, index):
      a = self.calculate(inputs, index)
      return a * (1 - a)

  @staticmethod
  class ReLu:
    def calculate(self, inputs, index):
      return inputs[index] if inputs[index] > 0 else 0

    def derivative(self, inputs, index):
      return 1 if inputs[index] > 0 else 0

  @staticmethod
  class LeakyReLu:
    def __init__(self, alpha):
      self.alpha = alpha

    def calculate(self, inputs, index):
      return inputs[index] if inputs[index] > 0 else inputs[index] * self.alpha

    def derivative(self, inputs, index):
      return 1 if inputs[index] > 0 else self.alpha
      
  @staticmethod
  class TanH:
    def calculate(self, inputs, index):
      e2 = math.exp(2 * inputs[index])
      return (e2 - 1) / (e2 + 1)

    def derivative(self, inputs, index):
      t = self.calculate(inputs, index)
      return 1 - t * t

  @staticmethod
  class SiLu:
    def calculate(self, inputs, index):
      return inputs[index] / (1 + math.exp(-inputs[index]))

    def derivative(self, inputs, index):
      sig = 1 / (1 + math.exp(-inputs[index]))
      return inputs[index] * sig * (1 - sig) + sig

  @staticmethod
  class SoftMax:
    def calculate(self, inputs, index):
      expSum = 0
      for inputIndex in range(len(inputs)):
        expSum += math.exp(inputs[inputIndex])
      return math.exp(inputs[index]) / expSum

    def derivative(self, inputs, index):
      expSum = 0
      for inputIndex in range(len(inputs)):
        expSum += Math.exp(inputs[inputIndex])
      exp = Math.exp(inputs[index])
      return (exp * expSum - exp * exp) / (expSum * expSum)

# %% [markdown]
# ### loss
# calculates how much error the output of a neural network has to the expected output
# - mean squared error

# %%
class Loss:
  @staticmethod
  class MeanSquaredError:
    def calculate(self, predictedOutputs, expectedOutputs):
      loss = 0
      for outputIndex in range(len(predictedOutputs)):
        error = predictedOutputs[outputIndex] - expectedOutputs[outputIndex]
        loss += error * error
      return loss * 0.5
      
    def derivative(self, predictedOutput, expectedOutput):
      return predictedOutput - expectedOutput

# %% [markdown]
# ### layer
# layers for the neural network
# - linear
# - dense

# %%
class Layer:
  @staticmethod
  class Linear:
    def __init__(self, numNeurons):
      self.numNeurons = numNeurons

      self.values = None
      self.numNeuronsIn = None

      self.isTrainable = False
      
    def build(self, numNeuronsIn):
      self.numNeuronsIn = numNeuronsIn
      self.values = [0.0 for i in range(self.numNeurons)]

    def initialize(self):
      pass

    def forward(self, inputs):
      for neuronOutIndex in range(self.numNeurons):
        self.values[neuronOutIndex] = inputs[neuronOutIndex]
      return self.values

  @staticmethod
  class Dense:
    def __init__(self, numNeurons, weightInitializer, biasInitializer, activation):
      self.numNeurons = numNeurons
      self.weightInitializer = weightInitializer
      self.biasInitializer = biasInitializer
      self.activation = activation

      self.values = None
      self.numNeuronsIn = None
      self.values = None
      self.numWeights = None
      self.numBiases = None
      self.weights = None
      self.biases = None

      self.isTrainable = True

    def build(self, numNeuronsIn):
      self.numNeuronsIn = numNeuronsIn
      self.values = [0.0 for i in range(self.numNeurons)]
      self.numWeights = self.numNeuronsIn * self.numNeurons
      self.numBiases = self.numNeurons
      self.weights = [0.0 for i in range(self.numWeights)]
      self.biases = [0.0 for i in range(self.numBiases)]

    def initialize(self):
      self.weightInitializer.setValues(self.weights)
      self.biasInitializer.setValues(self.biases)

    def forward(self, inputs):
      weightedInputs = [0.0 for i in range(self.numNeurons)]
      for neuronOutIndex in range(self.numNeurons):
        weightedInput = self.biases[neuronOutIndex]
        for neuronInIndex in range(self.numNeuronsIn):
          weightedInput += inputs[neuronInIndex] * self.weights[neuronInIndex + neuronOutIndex * self.numNeuronsIn]
        weightedInputs[neuronOutIndex] = weightedInput
      activations = [0.0 for i in range(self.numNeurons)]
      for neuronOutIndex in range(self.numNeurons):
        activations[neuronOutIndex] = self.activation.calculate(weightedInputs, neuronOutIndex)
        self.values[neuronOutIndex] = activations[neuronOutIndex]
      return self.values

# %% [markdown]
# ### model
# the neural network model
# - sequential

# %%
class Model:
  @staticmethod
  class Sequential:
    def __init__(self, layers):
      self.layers = layers
      
      self.numLayers = len(self.layers)

    def build(self):
      self.layers[0].build(self.layers[0].numNeurons)
      for layerIndex in range(len(self.layers) - 1):
        self.layers[layerIndex+1].build(self.layers[layerIndex].numNeurons)
        
    def initialize(self):
      for layerIndex in range(len(self.layers)):
        self.layers[layerIndex].initialize()

    def forward(self, inputs):
      outputs = inputs;
      for layerIndex in range(self.numLayers):
        outputs = self.layers[layerIndex].forward(outputs);
      return outputs;

# %% [markdown]
# ### main

# %%
model = Model.Sequential([
  Layer.Linear(
    2),
  Layer.Dense(
    2,
    Initializer.Zero(),
    Initializer.RandomNormalDistribution(0, 1),
    Activation.SoftMax())])

model.build()
model.initialize()

print(model.forward([random.random()*2-1 for i in range(2)]))


