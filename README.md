# machineLearning
Exporing machine learning.

## MachineLearning.js overview
This is a good starting point for building basic neural networks in JavaScript that is easy to expand and maintain. However, for practical, large-scale, or real-world applications, you would face performance issues, a lack of modern features, and limited flexibility.

### Example usage
Simple xor gate
```js
// Create data
let data = [
  [[0, 0], [0, 1]],
  [[1, 0], [1, 0]],
  [[0, 1], [1, 0]],
  [[1, 1], [0, 1]]];



// Create the AI
let model = new MachineLearning.Model(
  [2, 6, 2], // layer sizes
  new MachineLearning.Activation.LeakyReLu(0.1), // activation
  new MachineLearning.Activation.LeakyReLu(0.1), // output activation
  new MachineLearning.WeightInitializer.He(), // weight initializer
  new MachineLearning.BiasInitializer.Constant(0.1) // bias initializer
);

let cost = new MachineLearning.Cost.MeanSquaredError();

let optimizer = new MachineLearning.Optimizer.Adam(
  model, // model
  0.03, // eta
  0.9, // beta1
  0.999, // beta2
  1e-8, // epsilon
  0.001 // lambda
);



// Setup
model.initialize();

// Train
for (let i = 0; i < 100; i++)
  for (let j = 0; j < data.length; j++) {
    model.backPropagate(data[j][0], data[j][1], cost);
    optimizer.applyGradients();
  }

// Evaluate
for (let j = 0; j < data.length; j++)
  console.log(model.feedForward(data[j][0]));
```
