class MachineLearning {
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

      derivative(z) {
        let d = new Float32Array(z.length);
        for (let j = 0; j < z.length; j++)
          d[j] = z[j] > 0 ? 1 : 0;
        return d;
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

      derivative(z) {
        let d = new Float32Array(z.length);
        for (let j = 0; j < z.length; j++)
          d[j] = z[j] > 0 ? 1 : this.alpha;
        return d;
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

      derivative(z) {
        let t = this.compute(z);
        let d = new Float32Array(z.length);
        for (let j = 0; j < z.length; j++)
          d[j] = 1 - t[j] * t[j];
        return d;
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

      derivative(z) {
        let d = new Float32Array(z.length);
        let expSum = 0;
        for (let j = 0; j < z.length; j++)
          expSum += Math.exp(z[j]);
        for (let j = 0; j < z.length; j++) {
          let exp = Math.exp(z[j]);
          d[j] = (exp * expSum - exp * exp) / (expSum * expSum);
        }
        return d;
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

      derivative(a, y) {
        let d = new Float32Array(a.length);
        for (let j = 0; j < a.length; j++)
          d[j] = a[j] - y[j];
        return d;
      }
    }
  }

  static WeightInitializer = class {
    static Glorot = class {
      initialize(weights, layerSizes) {
        for (let l = 1; l < layerSizes.length; l++)
          for (let j = 0; j < layerSizes[l]; j++)
            for (let k = 0; k < layerSizes[l - 1]; k++)
              weights[l - 1][k][j] = MachineLearning.Utils.gaussianRandom(0, 1) / Math.sqrt(layerSizes[l - 1]);
      }
    }

    static He = class {
      initialize(weights, layerSizes) {
        for (let l = 1; l < layerSizes.length; l++)
          for (let j = 0; j < layerSizes[l]; j++)
            for (let k = 0; k < layerSizes[l - 1]; k++)
              weights[l - 1][k][j] = MachineLearning.Utils.gaussianRandom(0, 1) / Math.sqrt(layerSizes[l]);
      }
    }
  }

  static BiasInitializer = class {
    static Zero = class {
      initialize(biases, layerSizes) {
        for (let l = 0; l < layerSizes.length; l++)
          for (let j = 0; j < layerSizes[l]; j++)
            biases[l][j] = 0;
      }
    }

    static Constant = class {
      constructor(value) {
        this.value = value;
      }

      initialize(biases, layerSizes) {
        for (let l = 0; l < layerSizes.length; l++)
          for (let j = 0; j < layerSizes[l]; j++)
            biases[l][j] = this.value;
      }
    }
  }

  static Model = class {
    constructor(layerSizes, activation, outputActivation, weightInitializer, biasInitializer) {
      this.layerSizes = layerSizes;
      this.activation = activation;
      this.outputActivation = outputActivation;
      this.weightInitializer = weightInitializer;
      this.biasInitializer = biasInitializer;

      this.numLayers = this.layerSizes.length;
      this.weights = new Array();
      for (let l = 1; l < this.numLayers; l++) {
        let weightLayer = new Array();
        for (let k = 0; k < this.layerSizes[l - 1]; k++)
          weightLayer.push(new Float32Array(this.layerSizes[l]));
        this.weights.push(weightLayer);
      }
      this.biases = new Array();
      for (let l = 0; l < this.numLayers; l++)
        this.biases.push(new Float32Array(this.layerSizes[l]));
      this.gradientW = new Array();
      for (let l = 1; l < this.numLayers; l++) {
        let gradientWLayer = new Array();
        for (let k = 0; k < this.layerSizes[l - 1]; k++)
          gradientWLayer.push(new Float32Array(this.layerSizes[l]));
        this.gradientW.push(gradientWLayer);
      }
      this.gradientB = new Array();
      for (let l = 0; l < this.numLayers; l++)
        this.gradientB.push(new Float32Array(this.layerSizes[l]));
    }

    initialize() {
      this.weightInitializer.initialize(this.weights, this.layerSizes);
      this.biasInitializer.initialize(this.biases, this.layerSizes);
    }

    feedForward(a) {
      for (let l = 1; l < this.numLayers; l++) {
        let z = new Float32Array(this.layerSizes[l]);
        for (let j = 0; j < this.layerSizes[l]; j++) {
          z[j] = this.biases[l][j];
          for (let k = 0; k < this.layerSizes[l - 1]; k++)
            z[j] += a[k] * this.weights[l - 1][k][j];
        }
        a = l != this.numLayers - 1 ? this.activation.compute(z) : this.outputActivation.compute(z);
      }
      return a;
    }

    backPropagate(a, y, cost) {
      let zs = [null];
      let as = [a];
      for (let l = 1; l < this.numLayers; l++) {
        let z = new Float32Array(this.layerSizes[l]);
        for (let j = 0; j < this.layerSizes[l]; j++) {
          z[j] = this.biases[l][j];
          for (let k = 0; k < this.layerSizes[l - 1]; k++)
            z[j] += a[k] * this.weights[l - 1][k][j];
        }
        a = l != this.numLayers - 1 ? this.activation.compute(z) : this.outputActivation.compute(z);
        zs.push(z);
        as.push(a);
      }
      let error = new Array();
      for (let l = 0; l < this.numLayers; l++) {
        let array0 = new Float32Array(this.layerSizes[l]);
        for (let j = 0; j < this.layerSizes[l]; j++)
          array0[j] = 0;
        error.push(array0);
      }
      let aPrime = this.outputActivation.derivative(zs[this.numLayers - 1]);
      let cPrime = cost.derivative(as[this.numLayers - 1], y);
      for (let j = 0; j < this.layerSizes[this.numLayers - 1]; j++)
        error[this.numLayers - 1][j] = aPrime[j] * cPrime[j];
      for (let l = this.numLayers - 1; l > 1; l--) {
        let sum = new Float32Array(this.layerSizes[l - 1]);
        for (let k = 0; k < this.layerSizes[l - 1]; k++)
          for (let j = 0; j < this.layerSizes[l]; j++)
            sum[k] += this.weights[l - 1][k][j] * error[l][j];
        let aPrime = this.activation.derivative(zs[l - 1]);
        for (let k = 0; k < this.layerSizes[l - 1]; k++)
          error[l - 1][k] = sum[k] * aPrime[k];
      }
      for (let l = 1; l < this.numLayers; l++)
        for (let k = 0; k < this.layerSizes[l - 1]; k++)
          for (let j = 0; j < this.layerSizes[l]; j++)
            this.gradientW[l - 1][k][j] = as[l - 1][k] * error[l][j];
      for (let l = 0; l < this.numLayers; l++)
        for (let j = 0; j < this.layerSizes[l]; j++)
          this.gradientB[l][j] = error[l][j];
    }

    applyGradients(eta) {
      for (let l = 1; l < this.numLayers; l++)
        for (let k = 0; k < this.layerSizes[l - 1]; k++)
          for (let j = 0; j < this.layerSizes[l]; j++)
            this.weights[l - 1][k][j] -= this.gradientW[l - 1][k][j] * eta;
      for (let l = 0; l < this.numLayers; l++)
        for (let j = 0; j < this.layerSizes[l]; j++)
          this.biases[l][j] -= this.gradientB[l][j] * eta;
    }
  }

  static Optimizer = class {
    static RMSProp = class {
      constructor(model, eta, beta, epsilon, momentum) {
        this.model = model;
        this.eta = eta;
        this.beta = beta;
        this.epsilon = epsilon;
        this.momentum = momentum;
    
        this.vW = new Array();
        this.vB = new Array();
        this.vWMomentum = new Array();
        this.vBMomentum = new Array();
        for (let l = 1; l < this.model.numLayers; l++) {
          let vWLayer = new Array();
          let vWMomentumLayer = new Array();
          for (let k = 0; k < this.model.layerSizes[l - 1]; k++) {
            vWLayer.push(new Float32Array(this.model.layerSizes[l]));
            vWMomentumLayer.push(new Float32Array(this.model.layerSizes[l]));
          }
          this.vW.push(vWLayer);
          this.vWMomentum.push(vWMomentumLayer);
        }
        for (let l = 0; l < this.model.numLayers; l++) {
          this.vB.push(new Float32Array(this.model.layerSizes[l]));
          this.vBMomentum.push(new Float32Array(this.model.layerSizes[l]));
        }
      }
    
      applyGradients() {
        for (let l = 1; l < this.model.numLayers; l++) {
          for (let k = 0; k < this.model.layerSizes[l - 1]; k++) {
            for (let j = 0; j < this.model.layerSizes[l]; j++) {
              const gradientW = this.model.gradientW[l - 1][k][j];
              this.vW[l - 1][k][j] = this.beta * this.vW[l - 1][k][j] + (1 - this.beta) * gradientW ** 2;
              this.vWMomentum[l - 1][k][j] = this.momentum * this.vWMomentum[l - 1][k][j] + (1 - this.momentum) * gradientW;
              const adjustedLearningRateW = this.eta / (Math.sqrt(this.vW[l - 1][k][j] + this.epsilon));
              this.model.weights[l - 1][k][j] -= adjustedLearningRateW * (this.vWMomentum[l - 1][k][j]);
            }
          }
          for (let j = 0; j < this.model.layerSizes[l]; j++) {
            const gradientB = this.model.gradientB[l - 1][j];
            this.vB[l - 1][j] = this.beta * this.vB[l - 1][j] + (1 - this.beta) * gradientB ** 2;
            this.vBMomentum[l - 1][j] = this.momentum * this.vBMomentum[l - 1][j] + (1 - this.momentum) * gradientB;
            const adjustedLearningRateB = this.eta / (Math.sqrt(this.vB[l - 1][j] + this.epsilon));
            this.model.biases[l - 1][j] -= adjustedLearningRateB * (this.vBMomentum[l - 1][j]);
          }
        }
      }
    }
  }
}
