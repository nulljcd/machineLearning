class MachineLearning {
  static DataType = class {
    static Float32 = 'Float32';
    static Float64 = 'Float64';
  }

  static Tensor1d = class {
    constructor(data, dataType = undefined) {
      if (typeof data != 'object') {
        this.length = data;
        this.dataType = dataType;
        switch (this.dataType) {
          case 'Float32':
            this.data = new Float32Array(this.length);
            break;
          case 'Float64':
            this.data = new Float64Array(this.length);
            break;
          default:
            this.data = new Array(this.length);
        }
      } else {
        this.data = data;
        this.length = this.data.length;
        this.dataType = undefined;
      }
    }

    get(index) {
      return this.data[index];
    }

    set(index, data) {
      this.data[index] = data;
    }

    fill(data) {
      for (let index = 0; index < this.length; index++)
        this.set(index, data);
    }
  }

  static Initializer = class {
    static Zero = class {
      initialize(values) {
        for (let valueIndex = 0; valueIndex < values.length; valueIndex++)
          values.set(valueIndex, 0);
      }
    }

    static One = class {
      initialize(values) {
        for (let valueIndex = 0; valueIndex < values.length; valueIndex++)
          values.set(valueIndex, 1);
      }
    }

    static RandomNormal = class {
      constructor(mean, stddev) {
        this.mean = mean;
        this.stddev = stddev;
      }

      initialize(values) {
        for (let valueIndex = 0; valueIndex < values.length; valueIndex++)
          values.set(valueIndex, Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random()) * this.stddev + this.mean);
      }
    }
  }

  static Activation = class {
    static Sigmoid = class {
      activate(inputs, index) {
        return 1 / (1 + Math.exp(-inputs.get(index)));
      }

      derivative(inputs, index) {
        let a = this.activate(inputs, index);
        return a * (1 - a);
      }
    }

    static TanH = class {
      activate(inputs, index) {
        let e2 = Math.exp(2 * inputs.get(index));
        return (e2 - 1) / (e2 + 1);
      }

      derivative(inputs, index) {
        let t = this.activate(inputs, index);
        return 1 - t * t;
      }
    }

    static ReLu = class {
      activate(inputs, index) {
        return Math.max(0, inputs.get(index));
      }

      derivative(inputs, index) {
        return inputs.get(index) > 0 ? 1 : 0;
      }
    }

    static SiLu = class {
      activate(inputs, index) {
        return inputs.get(index) / (1 + Math.exp(-inputs.get(index)));
      }

      derivative() {
        let sig = 1 / (1 + Math.exp(-inputs.get(index)));
        return inputs.get(index) * sig * (1 - sig) + sig;
      }
    }

    static SoftMax = class {
      activate(inputs, index) {
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

  static Cost = class {
    static MeanSquaredError = class {
      cost(predictedOutputs, expectedOutputs) {
        let cost = 0;
        for (let outputIndex = 0; outputIndex < predictedOutputs.Length; outputIndex++) {
          let error = predictedOutputs.get(outputIndex) - expectedOutputs.get(outputIndex);
          cost += error * error;
        }
        return 0.5 * cost;
      }

      derivative(predictedOutput, expectedOutput) {
        return predictedOutput - expectedOutput;
      }
    }

    static CrossEntropy = class {
      cost(predictedOutputs, expectedOutputs) {
        let cost = 0;
        for (let outputIndex = 0; outputIndex < predictedOutputs.Length; outputIndex++) {
          let x = predictedOutputs[outputIndex];
          let y = expectedOutputs[outputIndex];
          let v = y == 1 ? -Math.log(x) : -Math.log(1 - x);
          cost += !isNaN(v) ? v : 0;
        }
        return cost;
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
      constructor(units) {
        this.units = units;
      }

      build(unitsIn) {

      }

      initialize() {

      }

      forward(inputs) {
        return inputs;
      }
    }

    static Dense = class {
      constructor(units, kernelInitializer, biasInitializer, activation, dataType) {
        this.units = units;
        this.kernelInitializer = kernelInitializer;
        this.biasInitializer = biasInitializer;
        this.activation = activation;
        this.dataType = dataType;
      }

      build(unitsIn) {
        this.unitsIn = unitsIn;

        this.weights = new MachineLearning.Tensor1d(this.unitsIn * this.units, this.dataType);
        this.biases = new MachineLearning.Tensor1d(this.units, this.dataType);
      }

      initialize() {
        this.kernelInitializer.initialize(this.weights);
        this.biasInitializer.initialize(this.biases);
      }

      forward(inputs) {
        let weightedInputs = new MachineLearning.Tensor1d(this.units, this.dataType);
        for (let outIndex = 0; outIndex < this.units; outIndex++) {
          let weightedInput = this.biases.get(outIndex);
          for (let inIndex = 0; inIndex < this.unitsIn; inIndex++)
            weightedInput += inputs.get(inIndex) * this.weights.get(inIndex + outIndex * this.unitsIn);
          weightedInputs.set(outIndex, weightedInput);
        }
        let activations = new MachineLearning.Tensor1d(this.units, this.dataType);
        for (let outIndex = 0; outIndex < this.units; outIndex++)
          activations.set(outIndex, this.activation.activate(weightedInputs, outIndex));
        return activations;
      }
    }
  }

  static Model = class {
    constructor(layers) {
      this.layers = layers;

      this.build();
    }

    build() {
      for (let layerIndex = 0; layerIndex < this.layers.length - 1; layerIndex++)
        this.layers.get(layerIndex + 1).build(this.layers.get(layerIndex).units);
    }

    initialize() {
      for (let layerIndex = 0; layerIndex < this.layers.length; layerIndex++) {
        this.layers.get(layerIndex).initialize();
      }
    }

    forward(inputs) {
      let outputs = inputs;
      for (let layerIndex = 0; layerIndex < this.layers.length; layerIndex++)
        outputs = this.layers.get(layerIndex).forward(outputs);
      return outputs;
    }
  }
}
