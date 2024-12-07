# machineLearning
Exporing machine learning.

## MachineLearning.js overview
- A good base for use ONLY with vanilla neural networks
- Well structured and modularized for maintenance and expansion
- No external libraries

### example usage
simple xor gate
```js
// create data
let data = [
  [[0, 0], [0, 1]],
  [[1, 0], [1, 0]],
  [[0, 1], [1, 0]],
  [[1, 1], [0, 1]]];



// create the AI
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



// setup
model.initialize();

// train
for (let i = 0; i < 100; i++)
  for (let j = 0; j < data.length; j++) {
    model.backPropagate(data[j][0], data[j][1], cost);
    optimizer.applyGradients();
  }

// evaluate
for (let j = 0; j < data.length; j++)
  console.log(model.feedForward(data[j][0]));
```
