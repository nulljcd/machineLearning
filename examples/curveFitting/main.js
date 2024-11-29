// utility functions
function shuffle(array) {
  for (var i = array.length - 1; i >= 0; i--) {
    var j = Math.floor(Math.random() * (i + 1));
    var temp = array[i];
    array[i] = array[j];
    array[j] = temp;
  }
}

function addFourierFeatures(x, numOrders) {
  let features = new Float64Array(1 + numOrders * 2);
  features[0] = x;
  for (let i = 0; i < numOrders; i++) {
    features[i * 2 + 1] = Math.sin(x * (i + 1));
    features[i * 2 + 2] = Math.cos(x * (i + 1));
  }
  return features;
}



// global variables
let network, renderer, functions, data, inputs;



// parameters
let numDataPoints = 100;
let hiddenLayerSizes = [24, 12];

let learningRate = 0.2;
let learningDecay = 0.01;
let fourierOrders = 4;
let randomness = 0.08;
let activation = new Network.Activation.TanH();



// run functions
function setup() {
  network = new Network(
    [1 + fourierOrders * 2, ...hiddenLayerSizes, 1],
    activation,
    new Network.Activation.TanH(),
    new Network.Cost.MeanSquaredError()
  );

  renderer = new Renderer(document.querySelector("#canvas"), 16 * 42, 10 * 42, 40);

  createData();
}

function createData() {
  renderer.clearPoints();
  data = new Array(numDataPoints);
  for (let i = 0; i < numDataPoints; i++) {
    let x = (i / numDataPoints) * 2 - 1;
    let y = Math.sin(x * 1 * Math.PI) * 0.4 + Math.cos(x * 2.8 * Math.PI) * 0.2
    y += (Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random())) * randomness;
    data[i] = new Float64Array(2);
    data[i][0] = x;
    data[i][1] = y;
    renderer.addPoint(data[i][0], data[i][1]);
  }
}

function handleInputs() {
  inputs = {
    learningRate: document.querySelector("#learningRate"),
    learningDecay: document.querySelector("#learningDecay"),
    fourierOrders: document.querySelector("#fourierOrders"),
    randomness: document.querySelector("#randomness"),
    activation: document.querySelector("#activation"),
    run: document.querySelector("#run")
  };

  inputs.run.onclick = () => {
    learningRate = inputs.learningRate.value;
    learningDecay = inputs.learningDecay.value;
    fourierOrders = inputs.fourierOrders.value;
    randomness = inputs.randomness.value;
    console.log(inputs.activation.value)
    activation = inputs.activation.value == "TanH" ? new Network.Activation.TanH() : new Network.Activation.LeakyReLu(0.1);
    createData();
    network = new Network(
      [1 + fourierOrders * 2, ...hiddenLayerSizes, 1],
      activation,
      new Network.Activation.TanH(),
      new Network.Cost.MeanSquaredError()
    );
  };
}

function run() {
  requestAnimationFrame(run);
  
  shuffle(data);
  for (let i = 0; i < numDataPoints; i++) {
    network.backPropagate(addFourierFeatures(data[i][0], fourierOrders), [data[i][1]]);
    network.applyGradients(learningRate);
  }
  learningRate *= 1 - learningDecay;

  renderer.clearLines();
  for (let i = 0; i < numDataPoints; i++) {
    let x0 = (i / numDataPoints) * 2 - 1;
    let x1 = ((i + 1) / numDataPoints) * 2 - 1;
    renderer.addLine(x0, network.feedForward(addFourierFeatures(x0, fourierOrders)), x1, network.feedForward(addFourierFeatures(x1, fourierOrders)));
  }
  renderer.render();
}



setup();
handleInputs();
run();