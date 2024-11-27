class Utils {
  static gaussianRandom(mean, stddev) {
    // https://stackoverflow.com/questions/25582882/javascript-math-random-normal-distribution-gaussian-bell-curve
    return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random()) * stddev + mean;
  }
}

class Activation {
  static TanH = class {
    compute(z) {
      let a = new Float64Array(z.length);
      for (let j = 0; j < z.length; j++) {
        let e2 = Math.exp(2 * z[j]);
        a[j] = (e2 - 1) / (e2 + 1);
      }
      return a;
    }

    derivative(z) {
      let t = this.compute(z);
      let d = new Float64Array(z.length);
      for (let j = 0; j < z.length; j++)
        d[j] = 1 - t[j] * t[j];
      return d;
    }
  }

  static LeakyReLu = class {
    constructor(alpha) {
      this.alpha = alpha;
    }

    compute(z) {
      let a = new Float64Array(z.length);
      for (let j = 0; j < z.length; j++)
        a[j] = z[j] > 0 ? z[j] : z[j] * this.alpha;
      return a;
    }

    derivative(z) {
      let d = new Float64Array(z.length);
      for (let j = 0; j < z.length; j++)
        d[j] = z[j] > 0 ? 1 : this.alpha;
      return d;
    }
  }
}

class Cost {
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
      let d = new Float64Array(a.length);
      for (let j = 0; j < a.length; j++)
        d[j] = a[j] - y[j];
      return d;
    }
  }
}

class Network {
  constructor(layerSizes, activation, cost) {
    this.layerSizes = layerSizes;
    this.activation = activation;
    this.cost = cost;

    this.numLayers = this.layerSizes.length;
    this.weights = new Array();
    for (let l = 1; l < this.numLayers; l++) {
      let array0 = new Array();
      for (let k = 0; k < this.layerSizes[l - 1]; k++) {
        let array1 = new Float64Array(this.layerSizes[l]);
        for (let j = 0; j < this.layerSizes[l]; j++)
          array1[j] = Utils.gaussianRandom(0, 1) / Math.sqrt(this.layerSizes[l - 1]);
        array0.push(array1);
      }
      this.weights.push(array0);
    }
    this.biases = new Array();
    for (let l = 0; l < this.numLayers; l++) {
      let array0 = new Float64Array(this.layerSizes[l]);
      this.biases.push(array0);
    }

    this.gradientW = new Array();
    for (let l = 1; l < this.numLayers; l++) {
      let array0 = new Array();
      for (let k = 0; k < this.layerSizes[l - 1]; k++) {
        let array1 = new Float64Array(this.layerSizes[l]);
        array0.push(array1);
      }
      this.gradientW.push(array0);
    }
    this.gradientB = new Array();
    for (let l = 0; l < this.numLayers; l++) {
      let array0 = new Float64Array(this.layerSizes[l]);
      this.gradientB.push(array0);
    }
  }

  feedForward(a) {
    for (let l = 1; l < this.numLayers; l++) {
      let z = new Float64Array(this.layerSizes[l]);
      for (let j = 0; j < this.layerSizes[l]; j++) {
        z[j] = this.biases[l][j];
        for (let k = 0; k < this.layerSizes[l - 1]; k++)
          z[j] += a[k] * this.weights[l - 1][k][j];
      }
      a = this.activation.compute(z);
    }
    return a;
  }

  backPropagate(a, y) {
    let zs = [null];
    let as = [a];
    for (let l = 1; l < this.numLayers; l++) {
      let z = new Float64Array(this.layerSizes[l]);
      for (let j = 0; j < this.layerSizes[l]; j++) {
        z[j] = this.biases[l][j];
        for (let k = 0; k < this.layerSizes[l - 1]; k++)
          z[j] += a[k] * this.weights[l - 1][k][j];
      }
      a = this.activation.compute(z);
      zs.push(z);
      as.push(a);
    }
    let error = new Array();
    for (let l = 0; l < this.numLayers; l++) {
      let array0 = new Float64Array(this.layerSizes[l]);
      for (let j = 0; j < this.layerSizes[l]; j++)
        array0[j] = 0;
      error.push(array0);
    }
    let aPrime = this.activation.derivative(zs[this.numLayers - 1]);
    let cPrime = this.cost.derivative(as[this.numLayers - 1], y);
    for (let j = 0; j < this.layerSizes[this.numLayers - 1]; j++)
      error[this.numLayers - 1][j] = aPrime[j] * cPrime[j];
    for (let l = this.numLayers - 1; l > 1; l--) {
      let sum = new Float64Array(this.layerSizes[l - 1]);
      for (let k = 0; k < this.layerSizes[l - 1]; k++)
        for (let j = 0; j < this.layerSizes[l]; j++)
          sum[k] += this.weights[l - 1][k][j] * error[l][j];
      let aPrime = this.activation.derivative(zs[l - 1]);
      for (let k = 0; k < this.layerSizes[this.numLayers - 1]; k++)
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
        this.biases[l][j] -= this.gradientB[l][j] * eta
  }
}



let network = new Network(
  [1, 1],
  new Activation.TanH(),
  new Cost.MeanSquaredError());