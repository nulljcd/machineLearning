// helper functions
function shuffle(array) {
  for (let i = array.length - 1; i >= 0; i--) {
    let j = Math.floor(Math.random() * (i + 1));
    let temp = array[i];
    array[i] = array[j];
    array[j] = temp;
  }
}

function addFourierFeatures(x, numOrders) {
  let features = new Float32Array(1 + numOrders * 2);
  features[0] = x;
  for (let i = 0; i < numOrders; i++) {
    features[i * 2 + 0] = Math.sin(x * (i + 1));
    features[i * 2 + 1] = Math.cos(x * (i + 1));
  }
  return features;
}

function createData(numPoints, f) {
  let data = new Array(numPoints);
  for (let i = 0; i < numPoints; i++) {
    let x = (i / numPoints) * 2 - 1;
    let y = f(x);

    data[i] = new Float32Array(2);
    data[i][0] = x;
    data[i][1] = y;
  }
  return data;
}



// create renderer
let width = 16;
let height = 10;
let scale = Math.min(window.innerWidth / width, window.innerHeight / height);
let graph = new Graph(document.querySelector("#canvas"), width * scale, height * scale, 10);
window.onresize = () => {
  scale = Math.min(window.innerWidth / width, window.innerHeight / height);
  graph.setResolution(width * scale, height * scale);
};



// create the data
let numPoints = 80;
let fs = [
  x => Math.sin(x * 2 * Math.PI) * 0.5,
  x => Math.sin(x * Math.PI) * 0.5 + Math.cos(x * 3 * Math.PI) * 0.2,
  x => Math.abs(Math.sin(x * 2 * Math.PI) * 0.5) - 0.25,
  x => Math.abs(x) - 0.5,
];
let data = createData(numPoints, fs[0]);
let fChangeCounter = 0;
let fNum = 0;
graph.clearPoints();
for (let i = 0; i < numPoints; i++)
  graph.addPoint(data[i][0], data[i][1]);

let numFourierOrders = 12;
let linePoints = 500;



// Create the AI
let model = new MachineLearning.Model(
  [1 + numFourierOrders * 2, 12, 1], // layer sizes
  new MachineLearning.Activation.TanH(), // activation
  new MachineLearning.Activation.TanH(), // output activation
  new MachineLearning.WeightInitializer.Glorot(), // weight initializer
  new MachineLearning.BiasInitializer.Zero() // bias initializer
);

let cost = new MachineLearning.Cost.MeanSquaredError();

let optimizer = new MachineLearning.Optimizer.Adam(
  model, // model
  0.00005, // eta
  0.99, // beta1
  0.9, // beta2
  1e-8, // epsilon
  0 // lambda
);



// Setup
model.initialize();



// run
function run() {
  requestAnimationFrame(run);
  shuffle(data);

  for (let i = 0; i < numPoints; i++) {
    model.backPropagate(addFourierFeatures(data[i][0], numFourierOrders), [data[i][1]], cost);
    optimizer.applyGradients();
  }

  graph.clearLines();
  let lastX = null;
  let lastY = null;
  for (let i = 0; i < linePoints; i++) {
    let x = (i / linePoints) * 2 - 1;
    let y = model.feedForward(addFourierFeatures(x, numFourierOrders))[0];
    if (lastX == null) {
      lastX = x;
      lastY = y;
    } else {
      graph.addLine(lastX, lastY, x, y);
      lastX = x;
      lastY = y;
    }
  }

  graph.render();

  fChangeCounter++;
  if (fChangeCounter > 200) {
    fChangeCounter = 0;
    fNum++;
    if (fNum == fs.length)
      fNum = 0;
    data = createData(numPoints, fs[fNum]);
    graph.clearPoints();
    for (let i = 0; i < numPoints; i++)
      graph.addPoint(data[i][0], data[i][1]);
    optimizer.t = 0;
  }
}

run();