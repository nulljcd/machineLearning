# %% [markdown]
# # machine learning project
# 

# %% [markdown]
# ## imports

# %%
import math
import random
import copy

# %% [markdown]
# ## library

# %% [markdown]
# ### initializer
# used for initializing the hyper parameters before training
# - zero
# - one
# - random normal distribution

# %%
class Initializer:
  class Zero:
    def calculateValue(self):
      return 0
      
    def setValues(self, values):
      for i in range(len(values)):
        values[i] = self.calculateValue()

  class One:
    def calculateValue(self):
      return 1

    def setValues(self, values):
      for i in range(len(values)):
        values[i] = self.calculateValue()

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
  class Linear:
    def calculate(self, inputs, index):
      return inputs[index]
      
    def derivative(self, inputs, index):
      return 1

  class Sigmoid:
    def calculate(self, inputs, index):
      return 1 / (1 + math.exp(-inputs[index]))

    def derivative(self, inputs, index):
      a = self.calculate(inputs, index)
      return a * (1 - a)

  class ReLu:
    def calculate(self, inputs, index):
      return inputs[index] if inputs[index] > 0 else 0

    def derivative(self, inputs, index):
      return 1 if inputs[index] > 0 else 0

  class LeakyReLu:
    def __init__(self, alpha):
      self.alpha = alpha

    def calculate(self, inputs, index):
      return inputs[index] if inputs[index] > 0 else inputs[index] * self.alpha

    def derivative(self, inputs, index):
      return 1 if inputs[index] > 0 else self.alpha
      
  class TanH:
    def calculate(self, inputs, index):
      e2 = math.exp(2 * inputs[index])
      return (e2 - 1) / (e2 + 1)

    def derivative(self, inputs, index):
      t = self.calculate(inputs, index)
      return 1 - t * t

  class SiLu:
    def calculate(self, inputs, index):
      return inputs[index] / (1 + math.exp(-inputs[index]))

    def derivative(self, inputs, index):
      sig = 1 / (1 + math.exp(-inputs[index]))
      return inputs[index] * sig * (1 - sig) + sig

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
# ### layer
# layers for the neural network
# - linear
# - dense

# %%
class Layer:
  class Linear:
    def __init__(self, numUnits):
      self.numUnits = numUnits

      self.values = None
      self.numInputUnits = None

      self.isTrainable = False
      
    def build(self, numInputUnits):
      self.numInputUnits = numInputUnits
      self.values = [0.0 for i in range(self.numUnits)]

    def initialize(self):
      pass

    def forward(self, inputs):
      for neuronOutIndex in range(self.numUnits):
        self.values[neuronOutIndex] = inputs[neuronOutIndex]
      return self.values

  class Dense:
    def __init__(self, numUnits, weightInitializer, biasInitializer, activation):
      self.numUnits = numUnits
      self.weightInitializer = weightInitializer
      self.biasInitializer = biasInitializer
      self.activation = activation

      self.values = None
      self.numInputUnits = None
      self.values = None
      self.numWeights = None
      self.numBiases = None
      self.weights = None
      self.biases = None

      self.isTrainable = True

    def build(self, numInputUnits):
      self.numInputUnits = numInputUnits
      self.values = [0.0 for i in range(self.numUnits)]
      self.numWeights = self.numInputUnits * self.numUnits
      self.numBiases = self.numUnits
      self.weights = [0.0 for i in range(self.numWeights)]
      self.biases = [0.0 for i in range(self.numBiases)]

    def initialize(self):
      self.weightInitializer.setValues(self.weights)
      self.biasInitializer.setValues(self.biases)

    def forward(self, inputs):
      weightedInputs = [0.0 for i in range(self.numUnits)]
      for neuronOutIndex in range(self.numUnits):
        weightedInput = self.biases[neuronOutIndex]
        for neuronInIndex in range(self.numInputUnits):
          weightedInput += inputs[neuronInIndex] * self.weights[neuronInIndex + neuronOutIndex * self.numInputUnits]
        weightedInputs[neuronOutIndex] = weightedInput
      for neuronOutIndex in range(self.numUnits):
        self.values[neuronOutIndex] = self.activation.calculate(weightedInputs, neuronOutIndex)
      return self.values

# %% [markdown]
# ### loss
# calculates how much error the output of a neural network has to the expected output
# - mean squared error

# %%
class Loss:
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
# ### model
# the neural network model
# - sequential

# %%
class Model:
  class Sequential:
    def __init__(self, layers):
      self.layers = layers
      
      self.numLayers = len(self.layers)

    def build(self):
      self.layers[0].build(self.layers[0].numUnits)
      for layerIndex in range(len(self.layers) - 1):
        self.layers[layerIndex+1].build(self.layers[layerIndex].numUnits)
        
    def initialize(self):
      for layerIndex in range(len(self.layers)):
        self.layers[layerIndex].initialize()

    def forward(self, inputs):
      outputs = inputs;
      for layerIndex in range(self.numLayers):
        outputs = self.layers[layerIndex].forward(outputs);
      return outputs;

# %% [markdown]
# ### training system
# systems to train neural networks
# - deep genetic algorithm

# %%
class TrainingSystem:
  class DeepGeneticAlgorithm:
    class MutateSystem:
      class AddativeRandomNormalDistribution:
        def __init__(self, mean, stddev):
          self.mean = mean
          self.stddev = stddev

        def mutate(self, value):
          return value + random.normalvariate(self.mean, self.stddev)
    
    def __init__(self, baseModel, numAgents, numParents, numElitist, weightMutateSystem, weightReplaceRate, biasMutateSystem, biasReplaceRate):
      self.baseModel = baseModel
      self.numAgents = numAgents
      self.numParents = numParents
      self.numElitist = numElitist
      self.weightMutateSystem = weightMutateSystem
      self.weightReplaceRate = weightReplaceRate
      self.biasMutateSystem = biasMutateSystem
      self.biasReplaceRate = biasReplaceRate

      self.agents = None
      self.agentLosses = None
      self.fittestAgentIndexes = None

    def build(self):
      self.agents = [None for i in range(self.numAgents)]
      self.agentLosses = [0.0 for i in range(self.numAgents)]
      self.fittestAgentIndexes = [0 for i in range(self.numAgents)]

      for agentIndex in range(self.numAgents):
        self.agents[agentIndex] = copy.deepcopy(self.baseModel)
        self.agents[agentIndex].build()
        self.fittestAgentIndexes[agentIndex] = agentIndex
      self.baseModel.build()

    def initialize(self):
      for agentIndex in range(self.numAgents):
        self.agents[agentIndex].initialize()

    def step(self):
      def sortFunction(index):
        return self.agentLosses[index]
      self.fittestAgentIndexes.sort(key=sortFunction)

      for layerIndex in range(self.baseModel.numLayers):
        if not self.baseModel.layers[layerIndex].isTrainable:
          continue

        parentWeights = [0.0 for i in range(self.baseModel.layers[layerIndex].numWeights * self.numParents)]
        parentBiases = [0.0 for i in range(self.baseModel.layers[layerIndex].numBiases * self.numParents)]
        parentWeightIndexOffset = 0
        parentBiasIndexOffset = 0

        for parentIndex in range(self.numParents):
          for weightIndex in range(self.baseModel.layers[layerIndex].numWeights):
            parentWeights[weightIndex + parentWeightIndexOffset] = self.agents[self.fittestAgentIndexes[parentIndex]].layers[layerIndex].weights[weightIndex]
          for biasIndex in range(self.baseModel.layers[layerIndex].numBiases):
            parentBiases[biasIndex + parentBiasIndexOffset] = self.agents[self.fittestAgentIndexes[parentIndex]].layers[layerIndex].biases[biasIndex]
          parentWeightIndexOffset += self.baseModel.layers[layerIndex].numWeights
          parentBiasIndexOffset += self.baseModel.layers[layerIndex].numBiases
        for agentIndex in range(self.numAgents - self.numElitist):
          for weightIndex in range(self.baseModel.layers[layerIndex].numWeights):
            agentLayerWeights = self.agents[self.fittestAgentIndexes[self.numAgents - agentIndex - 1]].layers[layerIndex].weights
            if (random.random() < self.weightReplaceRate):
              agentLayerWeights[weightIndex] = self.baseModel.layers[layerIndex].weightInitializer.calculateValue()
            else:
              agentLayerWeights[weightIndex] = parentWeights[weightIndex + int(random.random() * self.numParents) * self.baseModel.layers[layerIndex].numWeights]
              agentLayerWeights[weightIndex] = self.weightMutateSystem.mutate(agentLayerWeights[weightIndex])
          for biasIndex in range(self.baseModel.layers[layerIndex].numBiases):
            agentLayerBiases = self.agents[self.fittestAgentIndexes[self.numAgents - agentIndex - 1]].layers[layerIndex].biases
            if (random.random() < self.biasReplaceRate):
              agentLayerBiases[biasIndex] = self.baseModel.layers[layerIndex].biasInitializer.calculateValue()
            else:
              agentLayerBiases[biasIndex] = parentBiases[biasIndex + int(random.random() * self.numParents) * self.baseModel.layers[layerIndex].numBiases]
              agentLayerBiases[biasIndex] = self.biasMutateSystem.mutate(agentLayerBiases[biasIndex])
          


# %% [markdown]
# ## test
# most complicated binary decoder

# %%
baseModel = Model.Sequential([
  Layer.Linear(
    2),
  Layer.Dense(
    6,
    Initializer.RandomNormalDistribution(0, 1),
    Initializer.RandomNormalDistribution(0, 1),
    Activation.Sigmoid()),
  Layer.Dense(
    4,
    Initializer.RandomNormalDistribution(0, 1),
    Initializer.RandomNormalDistribution(0, 1),
    Activation.SoftMax())])

trainingSystem = TrainingSystem.DeepGeneticAlgorithm(
  baseModel,
  40, 4, 0,
  TrainingSystem.DeepGeneticAlgorithm.MutateSystem.AddativeRandomNormalDistribution(0, 0.5), 0,
  TrainingSystem.DeepGeneticAlgorithm.MutateSystem.AddativeRandomNormalDistribution(0, 0.5), 0)

lossFunction = Loss.MeanSquaredError()

trainingData = [
  [[0, 0], [1, 0, 0, 0]],
  [[1, 0], [0, 1, 0, 0]],
  [[0, 1], [0, 0, 1, 0]],
  [[1, 1], [0, 0, 0, 1]]]

# %%
def setup():
  print('---- setting up')
  trainingSystem.build()
  trainingSystem.initialize()

def train():
  print('---- training')
  for i in range(40):
    for agentIndex in range(40):
      totalLoss = 0
      for sampleIndex in range(4):
        outputs = trainingSystem.agents[agentIndex].forward(trainingData[sampleIndex][0])
        totalLoss += lossFunction.calculate(outputs, trainingData[sampleIndex][1])
      totalLoss /= 3
      trainingSystem.agentLosses[agentIndex] = totalLoss
    trainingSystem.step()
    if i % 10 == 0:
      print(f'  accuracy: {(1 - trainingSystem.agentLosses[trainingSystem.fittestAgentIndexes[0]]) * 100}%')

def test():
  print('---- testing')
  model = trainingSystem.agents[trainingSystem.fittestAgentIndexes[0]]
  for sampleIndex in range(4):
    outputs = model.forward(trainingData[sampleIndex][0])
    expectedOuputs = trainingData[sampleIndex][1]
    loss = lossFunction.calculate(outputs, expectedOuputs)
    print(f'-- case: {sampleIndex}')
    print(f'  expected outputs: {expectedOuputs}')
    print(f'  actual outputs: {outputs}')
    print(f'  accuracy: {(1-loss) * 100}%')

# %%
setup()
train()
test()


