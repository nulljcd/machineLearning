class Project {
  static DataType = class {
    static Float32 = "Float32";
    static Float64 = "Float64";
    static Uint16 = "Uint16";
  }

  static Utils = class {
    static deepClone(o) {
      // https://stackoverflow.com/questions/4459928/how-to-deep-clone-in-javascript
      var references = [];
      var cachedResults = [];
      function clone(obj) {
        if (typeof obj !== "object")
          return obj;
        var index = references.indexOf(obj);
        if (index !== -1)
          return cachedResults[index];
        references.push(obj);
        var result = Array.isArray(obj) ? [] :
          obj.constructor ? new obj.constructor() : {};
        cachedResults.push(result);
        for (var key in obj)
          if (obj.hasOwnProperty(key))
            result[key] = clone(obj[key]);
        return result;
      }
      return clone(o);
    }

    static randomNormalDistribution(mean, stddev) {
      // https://stackoverflow.com/questions/25582882/javascript-math-random-normal-distribution-gaussian-bell-curve
      return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random()) * stddev + mean;
    }
  }

  static Array = class {
    constructor(data, dataType = undefined) {
      if (typeof data != "object") {
        this.length = data;
        this.dataType = dataType;
        switch (this.dataType) {
          case Project.DataType.Float32:
            this.data = new Float32Array(this.length);
            break;
          case Project.DataType.Float64:
            this.data = new Float64Array(this.length);
            break;
          case Project.DataType.Uint16:
            this.data = new Uint16Array(this.length);
            break;
          default:
            this.data = new Array(this.length);
            break;
        }
      } else {
        this.length = data.length;
        this.dataType = dataType;
        switch (this.dataType) {
          case Project.DataType.Float32:
            this.data = new Float32Array(this.length);
            break;
          case Project.DataType.Float64:
            this.data = new Float64Array(this.length);
            break;
          case Project.DataType.Uint16:
            this.data = new Uint16Array(this.length);
            break;
          default:
            this.data = new Array(this.length);
            break;
        }
        for (let dataIndex = 0; dataIndex < this.length; dataIndex++)
          this.data[dataIndex] = data[dataIndex];
      }
    }

    get(index) {
      return this.data[index];
    }

    set(index, data) {
      this.data[index] = data;
    }
  }

  static Initializer = class {
    static Zero = class {
      calculateValue() {
        return 0;
      }

      setValues(values) {
        for (let valueIndex = 0; valueIndex < values.length; valueIndex++)
          values.set(valueIndex, this.calculateValue());
      }
    }

    static One = class {
      calculateValue() {
        return 1;
      }

      setValues(values) {
        for (let valueIndex = 0; valueIndex < values.length; valueIndex++)
          values.set(valueIndex, this.calculateValue());
      }
    }

    static RandomNormalDistribution = class {
      constructor(mean, stddev) {
        this.mean = mean;
        this.stddev = stddev;
      }

      calculateValue() {
        return Project.Utils.randomNormalDistribution(this.mean, this.stddev);
      }

      setValues(values) {
        for (let valueIndex = 0; valueIndex < values.length; valueIndex++)
          values.set(valueIndex, this.calculateValue());
      }
    }
  }

  static Activation = class {
    static Linear = class {
      calculate(inputs, index) {
        return inputs.get(index);
      }

      derivative(inputs, index) {
        return 1;
      }
    }

    static Sigmoid = class {
      calculate(inputs, index) {
        return 1 / (1 + Math.exp(-inputs.get(index)));
      }

      derivative(inputs, index) {
        let a = this.calculate(inputs, index);
        return a * (1 - a);
      }
    }

    static ReLu = class {
      calculate(inputs, index) {
        return Math.max(0, inputs.get(index));
      }

      derivative(inputs, index) {
        return inputs.get(index) > 0 ? 1 : 0;
      }
    }

    static LeakyReLu = class {
      constructor(alpha) {
        this.alpha = alpha;
      }

      calculate(inputs, index) {
        return inputs.get(index) > 0 ? inputs.get(index) : inputs.get(index) * this.alpha;
      }

      derivative(inputs, index) {
        return inputs.get(index) > 0 ? 1 : this.alpha;
      }
    }

    static TanH = class {
      calculate(inputs, index) {
        let e2 = Math.exp(2 * inputs.get(index));
        return (e2 - 1) / (e2 + 1);
      }

      derivative(inputs, index) {
        let t = this.calculate(inputs, index);
        return 1 - t * t;
      }
    }

    static SiLu = class {
      calculate(inputs, index) {
        return inputs.get(index) / (1 + Math.exp(-inputs.get(index)));
      }

      derivative() {
        let sig = 1 / (1 + Math.exp(-inputs.get(index)));
        return inputs.get(index) * sig * (1 - sig) + sig;
      }
    }

    static SoftMax = class {
      calculate(inputs, index) {
        let expSum = 0;
        for (let inputIndex = 0; inputIndex < inputs.length; inputIndex++)
          expSum += Math.exp(inputs.get(inputIndex));
        return Math.exp(inputs.get(index)) / expSum;
      }

      derivative(inputs, index) {
        let expSum = 0;
        for (let inputIndex = 0; inputIndex < inputs.length; inputIndex++)
          expSum += Math.exp(inputs.get(inputIndex));
        let ex = Math.exp(inputs.get(index));
        return (ex * expSum - ex * ex) / (expSum * expSum);
      }
    }
  }

  static Loss = class {
    static MeanSquaredError = class {
      calculate(predictedOutputs, expectedOutputs) {
        let loss = 0;
        for (let outputIndex = 0; outputIndex < predictedOutputs.length; outputIndex++) {
          let error = predictedOutputs.get(outputIndex) - expectedOutputs.get(outputIndex);
          loss += error * error;
        }
        return 0.5 * loss;
      }

      derivative(predictedOutput, expectedOutput) {
        return predictedOutput - expectedOutput;
      }
    }

    static CrossEntropy = class {
      calculate(predictedOutputs, expectedOutputs) {
        let loss = 0;
        for (let outputIndex = 0; outputIndex < predictedOutputs.length; outputIndex++) {
          let x = predictedOutputs.get(outputIndex);
          let y = expectedOutputs.get(outputIndex);
          let v = y == 1 ? -Math.log(x) : -Math.log(1 - x);
          loss += !isNaN(v) ? v : 0;
        }
        return loss;
      }

      derivative(predictedOutput, expectedOutput) {
        let x = predictedOutput;
        let y = expectedOutput;
        if (x == 0 || x == 1)
          return 0;
        return (-x + y) / (x * (x - 1));
      }
    }
  }

  static Layer = class {
    static Linear = class {
      constructor(numNeurons) {
        this.numNeurons = numNeurons;

        this.isTrainable = false;
      }

      build(numNeuronsIn) {
        this.numNeuronsIn = numNeuronsIn;
        this.values = new Project.Array(this.numNeurons, Project.DataType.Float64);
      }

      initialize() {

      }

      forward(inputs) {
        for (let neuronOutIndex = 0; neuronOutIndex < this.numNeurons; neuronOutIndex++)
          this.values.set(neuronOutIndex, inputs.get(neuronOutIndex));
        return this.values;
      }
    }

    static Dense = class {
      constructor(numNeurons, weightInitializer, biasInitializer, activation) {
        this.numNeurons = numNeurons;
        this.weightInitializer = weightInitializer;
        this.biasInitializer = biasInitializer;
        this.activation = activation;

        this.isTrainable = true;
      }

      build(numNeuronsIn) {
        this.numNeuronsIn = numNeuronsIn;
        this.numWeights = this.numNeuronsIn * this.numNeurons;
        this.numBiases = this.numNeurons;
        this.values = new Project.Array(this.numNeurons, Project.DataType.Float64);
        this.weights = new Project.Array(this.numWeights, Project.DataType.Float64);
        this.biases = new Project.Array(this.numBiases, Project.DataType.Float64);
      }

      initialize() {
        this.weightInitializer.setValues(this.weights);
        this.biasInitializer.setValues(this.biases);
      }

      forward(inputs) {
        let weightedInputs = new Project.Array(this.numNeurons, Project.DataType.Float64);
        for (let neuronOutIndex = 0; neuronOutIndex < this.numNeurons; neuronOutIndex++) {
          let weightedInput = this.biases.get(neuronOutIndex);
          for (let neuronInIndex = 0; neuronInIndex < this.numNeuronsIn; neuronInIndex++)
            weightedInput += inputs.get(neuronInIndex) * this.weights.get(neuronInIndex + neuronOutIndex * this.numNeuronsIn);
          weightedInputs.set(neuronOutIndex, weightedInput);
        }
        let activations = new Project.Array(this.numNeurons, Project.DataType.Float64);
        for (let neuronOutIndex = 0; neuronOutIndex < this.numNeurons; neuronOutIndex++) {
          activations.set(neuronOutIndex, this.activation.calculate(weightedInputs, neuronOutIndex));
          this.values.set(neuronOutIndex, activations.get(neuronOutIndex));
        }
        return this.values;
      }
    }
  }

  static Model = class {
    constructor(layers) {
      this.layers = layers;
    }

    build() {
      this.numLayers = this.layers.length;
      this.layers.get(0).build(this.layers.get(0).numNeurons);
      for (let layerIndex = 0; layerIndex < this.numLayers - 1; layerIndex++)
        this.layers.get(layerIndex + 1).build(this.layers.get(layerIndex).numNeurons);
    }

    initialize() {
      for (let layerIndex = 0; layerIndex < this.numLayers; layerIndex++) {
        this.layers.get(layerIndex).initialize();
      }
    }

    forward(inputs) {
      let outputs = inputs;
      for (let layerIndex = 0; layerIndex < this.numLayers; layerIndex++)
        outputs = this.layers.get(layerIndex).forward(outputs);
      return outputs;
    }

    getHyperParams() {
      let hyperParams = [];
      for (let layerIndex = 0; layerIndex < this.numLayers; layerIndex++) {
        let layer = this.layers.get(layerIndex);
        if (layer.isTrainable)
          for (let weightIndex = 0; weightIndex < layer.numWeights; weightIndex++)
            hyperParams.push(layer.weights.get(weightIndex));
      }
      for (let layerIndex = 0; layerIndex < this.numLayers; layerIndex++) {
        let layer = this.layers.get(layerIndex);
        if (layer.isTrainable)
          for (let biasIndex = 0; biasIndex < layer.numBiases; biasIndex++)
            hyperParams.push(layer.biases.get(biasIndex));
      }
      return hyperParams;
    }

    setHyperParams(hyperParams) {
      let hyperParamIndexOffset = 0;
      for (let layerIndex = 0; layerIndex < this.numLayers; layerIndex++) {
        let layer = this.layers.get(layerIndex);
        if (layer.isTrainable) {
          for (let weightIndex = 0; weightIndex < layer.numWeights; weightIndex++)
            layer.weights.set(weightIndex, hyperParams[weightIndex + hyperParamIndexOffset]);
          hyperParamIndexOffset += layer.numWeights;
        }
      }
      for (let layerIndex = 0; layerIndex < this.numLayers; layerIndex++) {
        let layer = this.layers.get(layerIndex);
        if (layer.isTrainable) {
          for (let biasIndex = 0; biasIndex < layer.numBiases; biasIndex++)
            layer.biases.set(biasIndex, hyperParams[biasIndex + hyperParamIndexOffset]);
          hyperParamIndexOffset += layer.numBiases;
        }
      }
    }
  }

  static TrainingSystem = class {
    static GeneticAlgorithm = class {
      static MutateSystem = class {
        static AddativeRandomNormalDistribution = class {
          constructor(mean, stddev) {
            this.mean = mean;
            this.stddev = stddev;
          }

          mutate(value) {
            return value + Project.Utils.randomNormalDistribution(this.mean, this.stddev);
          }
        }

        static MultiplicativeRandomNormalDistribution = class {
          constructor(mean, stddev) {
            this.mean = mean;
            this.stddev = stddev;
          }

          mutate(value) {
            return value * Project.Utils.randomNormalDistribution(this.mean, this.stddev);
          }
        }
      }

      constructor(baseModel, numAgents, numParents, numElitist, weightMutateSystem, weightReplaceRate, biasMutateSystem, biasReplaceRate) {
        this.baseModel = baseModel;
        this.numAgents = numAgents;
        this.numParents = numParents;
        this.numElitist = numElitist;
        this.weightMutateSystem = weightMutateSystem;
        this.weightReplaceRate = weightReplaceRate;
        this.biasMutateSystem = biasMutateSystem;
        this.biasReplaceRate = biasReplaceRate;
      }

      initialize() {
        this.agents = new Project.Array(this.numAgents);
        this.agentLosses = new Project.Array(this.numAgents, Project.DataType.Float64);
        this.fittestAgentIndexes = new Project.Array(this.numAgents, Project.DataType.Uint16);

        for (let agentIndex = 0; agentIndex < this.numAgents; agentIndex++) {
          this.agents.set(agentIndex, Project.Utils.deepClone(this.baseModel));

          this.agents.get(agentIndex).build();
          this.agents.get(agentIndex).initialize();

          this.fittestAgentIndexes.set(agentIndex, agentIndex);
        }

        this.baseModel.build();
        this.baseModel.initialize();
      }

      step() {
        this.fittestAgentIndexes.data.sort((a, b) => {
          return this.agentLosses.get(a) - this.agentLosses.get(b);
        });
        for (let layerIndex = 0; layerIndex < this.baseModel.numLayers; layerIndex++) {
          if (!this.baseModel.layers.get(layerIndex).isTrainable)
            continue;
          let parentWeights = new Project.Array(this.baseModel.layers.get(layerIndex).numWeights * this.numParents, Project.DataType.Float64);
          let parentWeightIndexOffset = 0;
          let parentBiases = new Project.Array(this.baseModel.layers.get(layerIndex).numBiases * this.numParents, Project.DataType.Float64);
          let parentBiasIndexOffset = 0;
          for (let parentIndex = 0; parentIndex < this.numParents; parentIndex++) {
            for (let weightIndex = 0; weightIndex < this.baseModel.layers.get(layerIndex).numWeights; weightIndex++)
              parentWeights.set(weightIndex + parentWeightIndexOffset, this.agents.get(
                this.fittestAgentIndexes.get(parentIndex)).layers.get(layerIndex).weights.get(weightIndex));
            parentWeightIndexOffset += this.baseModel.layers.get(layerIndex).numWeights;
            for (let biasIndex = 0; biasIndex < this.baseModel.layers.get(layerIndex).numBiases; biasIndex++)
              parentBiases.set(biasIndex + parentBiasIndexOffset, this.agents.get(
                this.fittestAgentIndexes.get(parentIndex)).layers.get(layerIndex).biases.get(biasIndex));
            parentBiasIndexOffset += this.baseModel.layers.get(layerIndex).numBiases;
          }
          for (let agentIndex = 0; agentIndex < this.numAgents - this.numElitist; agentIndex++) {
            for (let weightIndex = 0; weightIndex < this.baseModel.layers.get(layerIndex).numWeights; weightIndex++) {
              let agentLayerWeights = this.agents.get(this.fittestAgentIndexes.get(this.numAgents - agentIndex - 1)).layers.get(layerIndex).weights;
              agentLayerWeights.set(weightIndex, parentWeights.get(
                weightIndex + Math.floor(Math.random() * this.numParents) * this.baseModel.layers.get(layerIndex).numWeights));
              if (Math.random() < this.weightReplaceRate)
                agentLayerWeights.set(weightIndex, this.baseModel.layers.get(layerIndex).weightInitializer.calculateValue());
              else
                agentLayerWeights.set(weightIndex, this.weightMutateSystem.mutate(agentLayerWeights.get(weightIndex)));
            }
            for (let biasIndex = 0; biasIndex < this.baseModel.layers.get(layerIndex).numBiases; biasIndex++) {
              let agentLayerBiases = this.agents.get(this.fittestAgentIndexes.get(this.numAgents - agentIndex - 1)).layers.get(layerIndex).biases;
              agentLayerBiases.set(biasIndex, parentBiases.get(
                biasIndex + Math.floor(Math.random() * this.numParents) * this.baseModel.layers.get(layerIndex).numBiases));
              if (Math.random() < this.biasReplaceRate)
                agentLayerBiases.set(biasIndex, this.baseModel.layers.get(layerIndex).biasInitializer.calculateValue());
              else
                agentLayerBiases.set(biasIndex, this.biasMutateSystem.mutate(agentLayerBiases.get(biasIndex)));
            }
          }
        }
      }
    }
  }
}
