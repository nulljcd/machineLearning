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
      this.numHyperParameters = 0;
      for (let l = 1; l < this.numLayers; l++)
        for (let j = 0; j < this.layerSizes[l]; j++) {
          this.numBiases++;
          for (let k = 0; k < this.layerSizes[l - 1]; k++)
            this.numWeights++;
        }
      this.numHyperParameters = this.numWeights + this.numBiases;
      this.hyperParameters = new Float32Array(this.numHyperParameters);
    }

    initialize() {
      for (let i = 0; i < this.numHyperParameters; i++)
        this.hyperParameters[i] = Neuroevolution.Utils.gaussianRandom(0, 1);
    }

    feedForward(a) {
      let biasIndex = this.numWeights;
      let weightIndex = 0;
      for (let l = 1; l < this.numLayers; l++) {
        let z = new Float32Array(this.layerSizes[l]);
        for (let j = 0; j < this.layerSizes[l]; j++) {
          z[j] = this.hyperParameters[biasIndex];
          biasIndex++;
          for (let k = 0; k < this.layerSizes[l - 1]; k++) {
            z[j] += a[k] * this.hyperParameters[weightIndex];
            weightIndex++;
          }
        }
        a = l != this.numLayers - 1 ? this.activation.compute(z) : this.outputActivation.compute(z);
      }
      return a;
    }

    exportHyperParameters() {
      let hyperParameters = "";
      for (let i = 0; i < this.numHyperParameters; i++)
        hyperParameters += `${this.hyperParameters[i]} `;
      return hyperParameters.slice(0, hyperParameters.length - 1);
    }

    importHyperParameters(hyperParameters) {
      hyperParameters = hyperParameters.split(" ").map(Number);
      for (let i = 0; i < this.numHyperParameters; i++)
        this.hyperParameters[i] = hyperParameters[i];
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
      this.model.hyperParameters = this.agents[this.fittestAgentsIndexes[0]].hyperParameters;
      let fittestAgentsHyperParameters = new Float32Array(this.model.numHyperParameters * 2);
      for (let i = 0; i < this.numParents; i++)
        for (let j = 0; j < this.model.numHyperParameters; j++)
          fittestAgentsHyperParameters[j + i * this.model.numHyperParameters] = this.agents[this.fittestAgentsIndexes[i]].hyperParameters[j];
      for (let i = 0; i < this.numAgents; i++)
        for (let j = 0; j < this.model.numHyperParameters; j++) {
          this.agents[i].hyperParameters[j] = fittestAgentsHyperParameters[j + Math.floor(Math.random() * this.numParents) * this.model.numHyperParameters];
          if (Math.random() < this.mutateRate)
            this.agents[i].hyperParameters[j] += Neuroevolution.Utils.gaussianRandom(0, this.mutatePower);
        }
    }
  }
}
