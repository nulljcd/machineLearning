class MachineLearning {

  // Utility functions for random number generation
  static Utils = class {
    // Generates a Gaussian random number with a given mean and standard deviation
    static gaussianRandom(mean, stddev) {
      // https://stackoverflow.com/questions/25582882/javascript-math-random-normal-distribution-gaussian-bell-curve
      return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random()) + mean * stddev;
    }
  }

  // Activation functions (used in layers of the neural network)
  static Activation = class {

    // ReLU activation function
    static ReLu = class {
      compute(z) {
        let a = new Float32Array(z.length);
        for (let j = 0; j < z.length; j++) {
          a[j] = z[j] > 0 ? z[j] : 0; // Apply ReLU: f(z) = max(0, z)
        }
        return a;
      }

      derivative(z) {
        let d = new Float32Array(z.length);
        for (let j = 0; j < z.length; j++) {
          d[j] = z[j] > 0 ? 1 : 0; // Derivative of ReLU: 1 if z > 0, else 0
        }
        return d;
      }
    }

    // Leaky ReLU activation function
    static LeakyReLu = class {
      constructor(alpha) {
        this.alpha = alpha; // Alpha is the slope for negative values
      }

      compute(z) {
        let a = new Float32Array(z.length);
        for (let j = 0; j < z.length; j++) {
          a[j] = z[j] > 0 ? z[j] : z[j] * this.alpha; // Apply Leaky ReLU
        }
        return a;
      }

      derivative(z) {
        let d = new Float32Array(z.length);
        for (let j = 0; j < z.length; j++) {
          d[j] = z[j] > 0 ? 1 : this.alpha; // Derivative of Leaky ReLU
        }
        return d;
      }
    }

    // Hyperbolic Tangent (TanH) activation function
    static TanH = class {
      compute(z) {
        let a = new Float32Array(z.length);
        for (let j = 0; j < z.length; j++) {
          let e2 = Math.exp(2 * z[j]);
          a[j] = (e2 - 1) / (e2 + 1); // Apply TanH: f(z) = tanh(z) = (e^(2z) - 1) / (e^(2z) + 1)
        }
        return a;
      }

      derivative(z) {
        let t = this.compute(z);
        let d = new Float32Array(z.length);
        for (let j = 0; j < z.length; j++) {
          d[j] = 1 - t[j] * t[j]; // Derivative of TanH: f'(z) = 1 - tanh^2(z)
        }
        return d;
      }
    }

    // SoftMax activation function (used for multi-class classification)
    static SoftMax = class {
      compute(z) {
        let a = new Float32Array(z.length);
        let expSum = 0;
        for (let j = 0; j < z.length; j++) {
          expSum += Math.exp(z[j]); // Sum of exponentials of inputs
        }
        for (let j = 0; j < z.length; j++) {
          a[j] = Math.exp(z[j]) / expSum; // Apply SoftMax: f(z) = exp(z) / sum(exp(z))
        }
        return a;
      }

      derivative(z) {
        let d = new Float32Array(z.length);
        let expSum = 0;
        for (let j = 0; j < z.length; j++) {
          expSum += Math.exp(z[j]); // Sum of exponentials
        }
        for (let j = 0; j < z.length; j++) {
          let exp = Math.exp(z[j]);
          d[j] = (exp * expSum - exp * exp) / (expSum * expSum); // Derivative of SoftMax
        }
        return d;
      }
    }
  }

  // Cost functions (used to compute loss during training)
  static Cost = class {

    // Mean Squared Error cost function
    static MeanSquaredError = class {
      compute(a, y) {
        let cost = 0;
        for (let j = 0; j < a.length; j++) {
          let error = a[j] - y[j]; // Calculate the error
          cost += error * error; // Squared error
        }
        return cost * 0.5; // Half of squared error (for gradient calculation convenience)
      }

      derivative(a, y) {
        let d = new Float32Array(a.length);
        for (let j = 0; j < a.length; j++) {
          d[j] = a[j] - y[j]; // Derivative of MSE: f'(a) = a - y
        }
        return d;
      }
    }
  }

  // Weight Initializers (for setting initial weights before training)
  static WeightInitializer = class {

    // Glorot (Xavier) initialization (for symmetric activation functions)
    static Glorot = class {
      initialize(weights, layerSizes) {
        for (let l = 1; l < layerSizes.length; l++) {
          for (let j = 0; j < layerSizes[l]; j++) {
            for (let k = 0; k < layerSizes[l - 1]; k++) {
              weights[l - 1][k][j] = MachineLearning.Utils.gaussianRandom(0, 1) / Math.sqrt(layerSizes[l - 1]);
            }
          }
        }
      }
    }

    // He initialization (for ReLU-based activation functions)
    static He = class {
      initialize(weights, layerSizes) {
        for (let l = 1; l < layerSizes.length; l++) {
          for (let j = 0; j < layerSizes[l]; j++) {
            for (let k = 0; k < layerSizes[l - 1]; k++) {
              weights[l - 1][k][j] = MachineLearning.Utils.gaussianRandom(0, 1) / Math.sqrt(layerSizes[l]);
            }
          }
        }
      }
    }
  }

  // Bias Initializers (for setting initial biases before training)
  static BiasInitializer = class {

    // Initialize all biases to zero
    static Zero = class {
      initialize(biases, layerSizes) {
        for (let l = 0; l < layerSizes.length; l++) {
          for (let j = 0; j < layerSizes[l]; j++) {
            biases[l][j] = 0; // Set all biases to 0
          }
        }
      }
    }

    // Initialize all biases to a constant value
    static Constant = class {
      constructor(value) {
        this.value = value; // Value to set for all biases
      }

      initialize(biases, layerSizes) {
        for (let l = 0; l < layerSizes.length; l++) {
          for (let j = 0; j < layerSizes[l]; j++) {
            biases[l][j] = this.value; // Set all biases to a constant value
          }
        }
      }
    }
  }

  // Model class (defines the architecture and operations of the neural network)
  static Model = class {
    constructor(layerSizes, activation, outputActivation, weightInitializer, biasInitializer) {
      this.layerSizes = layerSizes; // Size of each layer
      this.activation = activation; // Activation function for hidden layers
      this.outputActivation = outputActivation; // Activation function for output layer
      this.weightInitializer = weightInitializer; // Weight initialization strategy
      this.biasInitializer = biasInitializer; // Bias initialization strategy

      this.numLayers = this.layerSizes.length; // Total number of layers
      this.weights = new Array(); // Weights of the network
      for (let l = 1; l < this.numLayers; l++) {
        let weightLayer = new Array();
        for (let k = 0; k < this.layerSizes[l - 1]; k++) {
          weightLayer.push(new Float32Array(this.layerSizes[l])); // Initialize weight matrix for each layer
        }
        this.weights.push(weightLayer);
      }
      this.biases = new Array(); // Biases of the network
      for (let l = 0; l < this.numLayers; l++) {
        this.biases.push(new Float32Array(this.layerSizes[l])); // Initialize bias vector for each layer
      }
      this.gradientW = new Array(); // Gradients for weights during backpropagation
      for (let l = 1; l < this.numLayers; l++) {
        let gradientWLayer = new Array();
        for (let k = 0; k < this.layerSizes[l - 1]; k++) {
          gradientWLayer.push(new Float32Array(this.layerSizes[l])); // Initialize gradient matrix for weights
        }
        this.gradientW.push(gradientWLayer);
      }
      this.gradientB = new Array(); // Gradients for biases during backpropagation
      for (let l = 0; l < this.numLayers; l++) {
        this.gradientB.push(new Float32Array(this.layerSizes[l])); // Initialize gradient vector for biases
      }
    }

    // Initialize weights and biases
    initialize() {
      this.weightInitializer.initialize(this.weights, this.layerSizes);
      this.biasInitializer.initialize(this.biases, this.layerSizes);
    }

    // Perform feedforward pass
    feedForward(a) {
      for (let l = 1; l < this.numLayers; l++) {
        let z = new Float32Array(this.layerSizes[l]); // Calculate weighted sum (z)
        for (let j = 0; j < this.layerSizes[l]; j++) {
          z[j] = this.biases[l][j]; // Add bias term
          for (let k = 0; k < this.layerSizes[l - 1]; k++) {
            z[j] += a[k] * this.weights[l - 1][k][j]; // Add weighted inputs
          }
        }
        // Apply activation function for hidden layers, output activation for final layer
        a = l !== this.numLayers - 1 ? this.activation.compute(z) : this.outputActivation.compute(z);
      }
      return a; // Return final output after feedforward pass
    }

    // Backpropagation algorithm to compute gradients
    backPropagate(a, y, cost) {
      let zs = [null]; // Array to store z values (weighted sums)
      let as = [a]; // Array to store activation values (outputs of neurons)

      // Forward pass to compute z and a for each layer
      for (let l = 1; l < this.numLayers; l++) {
        let z = new Float32Array(this.layerSizes[l]);
        for (let j = 0; j < this.layerSizes[l]; j++) {
          z[j] = this.biases[l][j]; // Compute weighted sum (z) for each neuron
          for (let k = 0; k < this.layerSizes[l - 1]; k++) {
            z[j] += a[k] * this.weights[l - 1][k][j]; // Add weighted inputs from previous layer
          }
        }
        a = l !== this.numLayers - 1 ? this.activation.compute(z) : this.outputActivation.compute(z); // Apply activation function
        zs.push(z);
        as.push(a);
      }

      let error = new Array(); // Array to store error terms for each layer
      for (let l = 0; l < this.numLayers; l++) {
        let errorLayer = new Float32Array(this.layerSizes[l]);
        for (let j = 0; j < this.layerSizes[l]; j++) {
          errorLayer[j] = 0; // Initialize error to zero for each neuron
        }
        error.push(errorLayer);
      }

      let aPrime = this.outputActivation.derivative(zs[this.numLayers - 1]);
      let cPrime = cost.derivative(as[this.numLayers - 1], y);
      // Calculate error at output layer
      for (let j = 0; j < this.layerSizes[this.numLayers - 1]; j++) {
        error[this.numLayers - 1][j] = aPrime[j] * cPrime[j];
      }

      // Propagate error backwards through the network
      for (let l = this.numLayers - 1; l > 1; l--) {
        let sum = new Float32Array(this.layerSizes[l - 1]); // Calculate the error for the previous layer
        for (let k = 0; k < this.layerSizes[l - 1]; k++) {
          for (let j = 0; j < this.layerSizes[l]; j++) {
            sum[k] += this.weights[l - 1][k][j] * error[l][j]; // Weighted sum of errors
          }
        }
        let aPrime = this.activation.derivative(zs[l - 1]);
        // Compute the error for the previous layer
        for (let k = 0; k < this.layerSizes[l - 1]; k++) {
          error[l - 1][k] = sum[k] * aPrime[k]; // Element-wise multiplication
        }
      }

      // Compute weight gradients
      for (let l = 1; l < this.numLayers; l++) {
        for (let k = 0; k < this.layerSizes[l - 1]; k++) {
          for (let j = 0; j < this.layerSizes[l]; j++) {
            this.gradientW[l - 1][k][j] = as[l - 1][k] * error[l][j]; // Gradient for weight
          }
        }
      }

      // Compute bias gradients
      for (let l = 0; l < this.numLayers; l++) {
        for (let j = 0; j < this.layerSizes[l]; j++) {
          this.gradientB[l][j] = error[l][j]; // Gradient for bias
        }
      }
    }
  }

  // Optimizers (used to update weights and biases during training)
  static Optimizer = class {

    // Gradient Descent optimizer
    static GradientDescent = class {
      constructor(model, eta) {
        this.model = model; // Model to optimize
        this.eta = eta; // Learning rate
      }

      applyGradients() {
        // Update weights and biases using gradients
        for (let l = 1; l < this.model.numLayers; l++) {
          for (let k = 0; k < this.model.layerSizes[l - 1]; k++) {
            for (let j = 0; j < this.model.layerSizes[l]; j++) {
              this.model.weights[l - 1][k][j] -= this.model.gradientW[l - 1][k][j] * this.eta; // Gradient descent update
            }
          }
        }

        // Update biases
        for (let l = 0; l < this.model.numLayers; l++) {
          for (let j = 0; j < this.model.layerSizes[l]; j++) {
            this.model.biases[l][j] -= this.model.gradientB[l][j] * this.eta; // Gradient descent update for biases
          }
        }
      }
    }

    // Adam optimizer (adaptive learning rate)
    static Adam = class {
      constructor(model, eta, beta1, beta2, epsilon, lambda) {
        this.model = model; // Model to optimize
        this.eta = eta; // Learning rate
        this.beta1 = beta1; // First moment decay rate
        this.beta2 = beta2; // Second moment decay rate
        this.epsilon = epsilon; // Small value to prevent division by zero
        this.lambda = lambda; // Regularization term

        // Initialize moment estimates for weights and biases
        this.mW = new Array();
        this.vW = new Array();
        this.mB = new Array();
        this.vB = new Array();

        // Initialize first and second moments for each weight and bias
        for (let l = 1; l < this.model.numLayers; l++) {
          let mWLayer = new Array();
          let vWLayer = new Array();
          for (let k = 0; k < this.model.layerSizes[l - 1]; k++) {
            mWLayer.push(new Float32Array(this.model.layerSizes[l]));
            vWLayer.push(new Float32Array(this.model.layerSizes[l]));
          }
          this.mW.push(mWLayer);
          this.vW.push(vWLayer);
        }

        // Initialize moment estimates for biases
        for (let l = 0; l < this.model.numLayers; l++) {
          this.mB.push(new Float32Array(this.model.layerSizes[l]));
          this.vB.push(new Float32Array(this.model.layerSizes[l]));
        }

        this.t = 0; // Time step for moment updates
      }

      // Apply gradients using Adam optimization
      applyGradients() {
        this.t++; // Increment time step

        // Update weights using Adam
        for (let l = 1; l < this.model.numLayers; l++) {
          for (let k = 0; k < this.model.layerSizes[l - 1]; k++) {
            for (let j = 0; j < this.model.layerSizes[l]; j++) {
              let gradientW = this.model.gradientW[l - 1][k][j];
              gradientW += this.lambda * this.model.weights[l - 1][k][j]; // Add regularization

              // Update first and second moments
              this.mW[l - 1][k][j] = this.beta1 * this.mW[l - 1][k][j] + (1 - this.beta1) * gradientW;
              this.vW[l - 1][k][j] = this.beta2 * this.vW[l - 1][k][j] + (1 - this.beta2) * gradientW ** 2;

              // Compute bias-corrected first and second moments
              let mHatW = this.mW[l - 1][k][j] / (1 - Math.pow(this.beta1, this.t));
              let vHatW = this.vW[l - 1][k][j] / (1 - Math.pow(this.beta2, this.t));

              // Adjust learning rate and update weights
              let adjustedLearningRateW = this.eta / (Math.sqrt(vHatW) + this.epsilon);
              this.model.weights[l - 1][k][j] -= adjustedLearningRateW * mHatW;
            }
          }
        }

        // Update biases using Adam
        for (let l = 0; l < this.model.numLayers; l++) {
          for (let j = 0; j < this.model.layerSizes[l]; j++) {
            let gradientB = this.model.gradientB[l][j];

            // Update first and second moments for biases
            this.mB[l][j] = this.beta1 * this.mB[l][j] + (1 - this.beta1) * gradientB;
            this.vB[l][j] = this.beta2 * this.vB[l][j] + (1 - this.beta2) * gradientB ** 2;

            // Compute bias-corrected first and second moments
            let mHatB = this.mB[l][j] / (1 - Math.pow(this.beta1, this.t));
            let vHatB = this.vB[l][j] / (1 - Math.pow(this.beta2, this.t));

            // Adjust learning rate and update biases
            let adjustedLearningRateB = this.eta / (Math.sqrt(vHatB) + this.epsilon);
            this.model.biases[l][j] -= adjustedLearningRateB * mHatB;
          }
        }
      }
    }
  }
}
