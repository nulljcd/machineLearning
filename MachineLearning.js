class MachineLearning {

  static Utils = class {
    static gaussianRandom(mean, stddev) {
      // https://stackoverflow.com/questions/25582882/javascript-math-random-normal-distribution-gaussian-bell-curve
      return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random()) * stddev + mean;
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

    static SiLu = class {
      compute(z) {
        let a = new Float32Array(z.length);
        for (let j = 0; j < z.length; j++)
          a[j] = z[j] / (1 + Math.exp(-z[j]));
        return a;
      }

      derivative(z) {
        let a = this.compute(z);
        let d = new Float32Array(z.length);
        for (let j = 0; j < z.length; j++) {
          let sig = 1 / (1 + Math.exp(-z[j]));
          d[j] = z[j] * sig * (1 - sig) + sig;
        }
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
        let a = this.compute(z);
        let d = new Float32Array(z.length);
        for (let j = 0; j < z.length; j++)
          d[j] = 1 - a[j] * a[j];
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
          let error = a[j] - y[j];
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
        for (let l = 1; l < layerSizes.length; l++) {
          for (let j = 0; j < layerSizes[l]; j++) {
            for (let k = 0; k < layerSizes[l - 1]; k++)
              weights[l - 1][k][j] = MachineLearning.Utils.gaussianRandom(0, 1) / Math.sqrt(layerSizes[l - 1]);
          }
        }
      }
    }

    static He = class {
      initialize(weights, layerSizes) {
        for (let l = 1; l < layerSizes.length; l++) {
          for (let j = 0; j < layerSizes[l]; j++) {
            for (let k = 0; k < layerSizes[l - 1]; k++)
              weights[l - 1][k][j] = MachineLearning.Utils.gaussianRandom(0, 1) / Math.sqrt(layerSizes[l]);
          }
        }
      }
    }
  }

  static BiasInitializer = class {

    static Zero = class {
      initialize(biases, layerSizes) {
        for (let l = 0; l < layerSizes.length; l++) {
          for (let j = 0; j < layerSizes[l]; j++)
            biases[l][j] = 0;
        }
      }
    }

    static Constant = class {
      constructor(value) {
        this.value = value;
      }

      initialize(biases, layerSizes) {
        for (let l = 0; l < layerSizes.length; l++) {
          for (let j = 0; j < layerSizes[l]; j++)
            biases[l][j] = this.value;
        }
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
      this.weights = Array.from({ length: this.numLayers - 1 }, (_, l) => Array.from({ length: this.layerSizes[l] }, () => new Float32Array(this.layerSizes[l + 1])));
      this.biases = Array.from({ length: this.numLayers }, (_, l) => new Float32Array(this.layerSizes[l]));
      this.gradientW = Array.from({ length: this.numLayers - 1 }, (_, l) => Array.from({ length: this.layerSizes[l] }, () => new Float32Array(this.layerSizes[l + 1])));
      this.gradientB = Array.from({ length: this.numLayers }, (_, l) => new Float32Array(this.layerSizes[l]));
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
        a = l !== this.numLayers - 1 ? this.activation.compute(z) : this.outputActivation.compute(z);
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
        a = l !== this.numLayers - 1 ? this.activation.compute(z) : this.outputActivation.compute(z);
        zs.push(z);
        as.push(a);
      }

      let error = Array.from({ length: this.numLayers }, (_, l) => new Float32Array(this.layerSizes[l]));

      let aPrime = this.outputActivation.derivative(zs[this.numLayers - 1]);
      let cPrime = cost.derivative(as[this.numLayers - 1], y);
      for (let j = 0; j < this.layerSizes[this.numLayers - 1]; j++)
        error[this.numLayers - 1][j] = aPrime[j] * cPrime[j];

      for (let l = this.numLayers - 1; l > 1; l--) {
        let sum = new Float32Array(this.layerSizes[l - 1]);
        for (let k = 0; k < this.layerSizes[l - 1]; k++) {
          for (let j = 0; j < this.layerSizes[l]; j++)
            sum[k] += this.weights[l - 1][k][j] * error[l][j];
        }
        let aPrime = this.activation.derivative(zs[l - 1]);
        for (let k = 0; k < this.layerSizes[l - 1]; k++)
          error[l - 1][k] = sum[k] * aPrime[k];
      }

      for (let l = 1; l < this.numLayers; l++) {
        for (let k = 0; k < this.layerSizes[l - 1]; k++) {
          for (let j = 0; j < this.layerSizes[l]; j++)
            this.gradientW[l - 1][k][j] += as[l - 1][k] * error[l][j];
        }
      }

      for (let l = 0; l < this.numLayers; l++) {
        for (let j = 0; j < this.layerSizes[l]; j++)
          this.gradientB[l][j] += error[l][j];
      }
    }

    zeroGradients() {
      for (let l = 1; l < this.numLayers; l++) {
        for (let k = 0; k < this.layerSizes[l - 1]; k++) {
          for (let j = 0; j < this.layerSizes[l]; j++)
            this.gradientW[l - 1][k][j] = 0;
        }
      }

      for (let l = 0; l < this.numLayers; l++) {
        for (let j = 0; j < this.layerSizes[l]; j++)
          this.gradientB[l][j] = 0;
      }
    }
  }

  static Optimizer = class {

    static GradientDescent = class {
      constructor(model, eta) {
        this.model = model;
        this.eta = eta;
      }

      step() {
        for (let l = 1; l < this.model.numLayers; l++) {
          for (let k = 0; k < this.model.layerSizes[l - 1]; k++) {
            for (let j = 0; j < this.model.layerSizes[l]; j++)
              this.model.weights[l - 1][k][j] -= this.model.gradientW[l - 1][k][j] * this.eta;
          }
        }

        for (let l = 0; l < this.model.numLayers; l++) {
          for (let j = 0; j < this.model.layerSizes[l]; j++)
            this.model.biases[l][j] -= this.model.gradientB[l][j] * this.eta;
        }
      }
    }

    static Adam = class {
      constructor(model, eta, beta1, beta2, epsilon, lambda) {
        this.model = model;
        this.eta = eta;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.lambda = lambda;

        this.mW = Array.from({ length: this.model.numLayers - 1 }, (_, l) => Array.from({ length: this.model.layerSizes[l] }, () => new Float32Array(this.model.layerSizes[l + 1])));
        this.vW = Array.from({ length: this.model.numLayers - 1 }, (_, l) => Array.from({ length: this.model.layerSizes[l] }, () => new Float32Array(this.model.layerSizes[l + 1])));
        this.mB = Array.from({ length: this.model.numLayers }, (_, l) => new Float32Array(this.model.layerSizes[l]));
        this.vB = Array.from({ length: this.model.numLayers }, (_, l) => new Float32Array(this.model.layerSizes[l]));

        this.t = 0;
      }

      step() {
        this.t++;

        for (let l = 1; l < this.model.numLayers; l++) {
          for (let k = 0; k < this.model.layerSizes[l - 1]; k++) {
            for (let j = 0; j < this.model.layerSizes[l]; j++) {
              let gradientW = this.model.gradientW[l - 1][k][j];
              gradientW += this.lambda * this.model.weights[l - 1][k][j];

              this.mW[l - 1][k][j] = this.beta1 * this.mW[l - 1][k][j] + (1 - this.beta1) * gradientW;
              this.vW[l - 1][k][j] = this.beta2 * this.vW[l - 1][k][j] + (1 - this.beta2) * gradientW ** 2;

              let mHatW = this.mW[l - 1][k][j] / (1 - Math.pow(this.beta1, this.t));
              let vHatW = this.vW[l - 1][k][j] / (1 - Math.pow(this.beta2, this.t));

              let adjustedLearningRateW = this.eta / (Math.sqrt(vHatW) + this.epsilon);
              this.model.weights[l - 1][k][j] -= adjustedLearningRateW * mHatW;
            }
          }
        }

        for (let l = 0; l < this.model.numLayers; l++) {
          for (let j = 0; j < this.model.layerSizes[l]; j++) {
            let gradientB = this.model.gradientB[l][j];

            this.mB[l][j] = this.beta1 * this.mB[l][j] + (1 - this.beta1) * gradientB;
            this.vB[l][j] = this.beta2 * this.vB[l][j] + (1 - this.beta2) * gradientB ** 2;

            let mHatB = this.mB[l][j] / (1 - Math.pow(this.beta1, this.t));
            let vHatB = this.vB[l][j] / (1 - Math.pow(this.beta2, this.t));

            let adjustedLearningRateB = this.eta / (Math.sqrt(vHatB) + this.epsilon);
            this.model.biases[l][j] -= adjustedLearningRateB * mHatB;
          }
        }
      }
    }
  }
}
