import("./resources.mjs").then(main);
async function main(res) {
  const { network, mnist } = res;
  const outputDiv = document.querySelector("#output");
  const predictednumberDiv = document.querySelector("#predictednumber");
  const correctnumberDiv = document.querySelector("#correctnumber");
  const dataindexDiv = document.querySelector("#dataindex");

  let data_index = 0;
  const clearButton = document.querySelector("#clear");
  const prevButton = document.querySelector("#prev");
  const nextButton = document.querySelector("#next");
  const nextwrongButton = document.querySelector("#nextwrong");

  clearButton.addEventListener("click", () => {
    ctx.beginPath();
    nn_input.fill(0);
    runData(nn_input);
    drawNumber(nn_input);
    ctx.closePath();
  });
  prevButton.addEventListener("click", () => {
    data_index--;
    if (data_index < 0) data_index = mnist.label.length - 1;
    runData();
  });
  nextButton.addEventListener("click", () => {
    data_index = (data_index + 1) % mnist.label.length;
    runData();
  });
  nextwrongButton.addEventListener("click", () => {
    do {
      data_index = (data_index + 1) % mnist.label.length;
      runData();
    } while (mnist.label[data_index] === predicted);
  });

  let predicted;
  runData();

  function runData(data) {
    let got_data = !!data;
    if (!got_data) {
      data = mnist.data.slice(28 * 28 * data_index, 28 * 28 * (data_index + 1));
      drawNumber(data);
    }
    const output = runNN(data, network);
    predicted = output.reduce((p, c, i, a) => (c > a[p] ? i : p), 0);
    outputDiv.innerHTML =
      "predicted numbers: " + output.map((v, i) => i + ": " + v).join("\n");
    predictednumberDiv.innerHTML = "predicted number: " + predicted;

    if (!got_data) {
      correctnumberDiv.innerHTML = "correct number: " + mnist.label[data_index];
      dataindexDiv.innerHTML = "data index: " + data_index;
      document.body.style.backgroundColor =
        predicted === mnist.label[data_index] ? "unset" : "red";
    }
  }
  window.runData = runData;
}

function runNN(input, { biases, weights, structure }) {
  let bias_index = 0;
  let weight_index = 0;
  let layers_output = structure.map(() => []);
  layers_output[0] = input;
  for (let layer_index = 1; layer_index < structure.length; layer_index++) {
    for (let out_index = 0; out_index < structure[layer_index]; out_index++) {
      let output = biases[bias_index];
      bias_index++;
      for (
        let in_index = 0;
        in_index < structure[layer_index - 1];
        in_index++
      ) {
        output +=
          weights[weight_index] * layers_output[layer_index - 1][in_index];
        weight_index++;
      }
      layers_output[layer_index].push(activation(output));
    }
  }
  return layers_output.at(-1);
  function activation(x) {
    return x > 0 ? x : 0.01 * x;
  }
}
