# machineLearning
Exporing machine learning.

- vanilla fully connected feed forward neural network
- backpropagation
- optimizers
- neuroevolution

## example usage
simple xor gate
```js
// create data
let data = [
  [[0, 0], [0, 1]],
  [[1, 0], [1, 0]],
  [[0, 1], [1, 0]],
  [[1, 1], [0, 1]]];



// create AI
let model = new MachineLearning.Model(
  [2, 6, 2],
  new MachineLearning.Activation.LeakyReLu(0.1),
  new MachineLearning.Activation.LeakyReLu(0.1),
  new MachineLearning.WeightInitializer.He(),
  new MachineLearning.BiasInitializer.Constant(0.1));

let cost = new MachineLearning.Cost.MeanSquaredError();

let optimizer = new MachineLearning.Optimizer.AdamW(model, 0.03, 0.9, 0.999, 1e-8, 0.001);



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


## notes:
- easily portable to any lang
- no external libraries
- prioritize readability over speed
