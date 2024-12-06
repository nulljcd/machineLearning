class Neuroevolution {
  static Utils = class {
    static gaussianRandom(mean, stddev) {
      // https://stackoverflow.com/questions/25582882/javascript-math-random-normal-distribution-gaussian-bell-curve
      return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random()) + mean * stddev;
    }
  }

  static Activation = class {
    static ReLu = class {
      compute(z) {
        let a = new Float32Array(z.length);
        for (let j = 0; j < z.length; j++)
          a[j] = z[j] > 0 ? z[j] : 0;
        return a;
      }
    }

    static LeakyReLu = class {
      constructor(alpha) {
        this.alpha = alpha;
      }

      compute(z) {
        let a = new Float32Array(z.length);
        for (let j = 0; j < z.length; j++)
          a[j] = z[j] > 0 ? z[j] : z[j] * this.alpha;
        return a;
      }
    }

    static TanH = class {
      compute(z) {
        let a = new Float32Array(z.length);
        for (let j = 0; j < z.length; j++) {
          let e2 = Math.exp(2 * z[j]);
          a[j] = (e2 - 1) / (e2 + 1);
        }
        return a;
      }
    }

    static SoftMax = class {
      compute(z) {
        let a = new Float32Array(z.length);
        let expSum = 0;
        for (let j = 0; j < z.length; j++)
          expSum += Math.exp(z[j]);
        for (let j = 0; j < z.length; j++)
          a[j] = Math.exp(z[j]) / expSum;
        return a;
      }
    }
  }

  static Cost = class {
    static MeanSquaredError = class {
      compute(a, y) {
        let cost = 0;
        for (let j = 0; j < a.length; j++) {
          error = a[j] - y[j];
          cost += error * error;
        }
        return cost * 0.5;
      }
    }
  }

  static Model = class {
    constructor(layerSizes, activation, outputActivation) {
      this.layerSizes = layerSizes;
      this.activation = activation;
      this.outputActivation = outputActivation;

      this.numLayers = this.layerSizes.length;
      this.numWeights = 0;
      this.numBiases = 0;
      this.numParameters = 0;
      for (let l = 1; l < this.numLayers; l++)
        for (let j = 0; j < this.layerSizes[l]; j++) {
          this.numBiases++;
          for (let k = 0; k < this.layerSizes[l - 1]; k++)
            this.numWeights++;
        }
      this.numParameters = this.numWeights + this.numBiases;
      this.parameters = new Float32Array(this.numParameters);
    }

    initialize() {
      for (let i = 0; i < this.numParameters; i++)
        this.parameters[i] = Neuroevolution.Utils.gaussianRandom(0, 1);
    }

    feedForward(a) {
      let biasIndex = this.numWeights;
      let weightIndex = 0;
      for (let l = 1; l < this.numLayers; l++) {
        let z = new Float32Array(this.layerSizes[l]);
        for (let j = 0; j < this.layerSizes[l]; j++) {
          z[j] = this.parameters[biasIndex];
          biasIndex++;
          for (let k = 0; k < this.layerSizes[l - 1]; k++) {
            z[j] += a[k] * this.parameters[weightIndex];
            weightIndex++;
          }
        }
        a = l != this.numLayers - 1 ? this.activation.compute(z) : this.outputActivation.compute(z);
      }
      return a;
    }

    exportParameters() {
      let parameters = "";
      for (let i = 0; i < this.numParameters; i++)
        parameters += `${this.parameters[i]} `;
      return parameters.slice(0, parameters.length - 1);
    }

    importParameters(parameters) {
      parameters = parameters.split(" ").map(Number);
      for (let i = 0; i < this.numParameters; i++)
        this.parameters[i] = parameters[i];
    }
  }

  static GeneticTrainingSystem = class {
    constructor(model, numAgents, numParents, mutateRate, mutatePower) {
      this.model = model;
      this.numAgents = numAgents;
      this.numParents = numParents;
      this.mutateRate = mutateRate;
      this.mutatePower = mutatePower;

      this.agents = new Array(this.numAgents);
      this.agentFitnesses = new Float32Array(this.numAgents);
      this.fittestAgentsIndexes = new Uint16Array(this.numAgents);
      for (let i = 0; i < this.numAgents; i++) {
        this.fittestAgentsIndexes[i] = i;
        this.agents[i] = new Neuroevolution.Model(this.model.layerSizes, this.model.activation);
        this.agents[i].initialize();
      }
    }

    step() {
      this.fittestAgentsIndexes.sort((a, b) => { return this.agentFitnesses[b] - this.agentFitnesses[a]; });
      this.model.parameters = this.agents[this.fittestAgentsIndexes[0]].parameters;
      let fittestAgentsParameters = new Float32Array(this.model.numParameters * 2);
      for (let i = 0; i < this.numParents; i++)
        for (let j = 0; j < this.model.numParameters; j++)
          fittestAgentsParameters[j + i * this.model.numParameters] = this.agents[this.fittestAgentsIndexes[i]].parameters[j];
      for (let i = 0; i < this.numAgents; i++)
        for (let j = 0; j < this.model.numParameters; j++) {
          this.agents[i].parameters[j] = fittestAgentsParameters[j + Math.floor(Math.random() * this.numParents) * this.model.numParameters];
          if (Math.random() < this.mutateRate)
            this.agents[i].parameters[j] += Neuroevolution.Utils.gaussianRandom(0, this.mutatePower);
        }
    }
  }
}
