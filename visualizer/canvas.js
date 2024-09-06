const canvas = document.getElementsByTagName("canvas")[0];
const ctx = canvas.getContext("2d");
ctx.fillStyle = "rgba(255,255,255,0.7)";

let is_drawing = false;
let mouse_position;

canvas.addEventListener("pointerdown", () => ((is_drawing = true), draw()));
canvas.addEventListener("pointerup", () => (is_drawing = false));
canvas.addEventListener(
  "pointermove",
  (ev) => (mouse_position = [ev.offsetX / 10, ev.offsetY / 10])
);

function draw() {
  if (!is_drawing) return;
  ctx.beginPath();
  ctx.arc(mouse_position[0], mouse_position[1], 1.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.closePath();
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  for (let i = 0; i < nn_input.length; i++) {
    nn_input[i] = data[i * 4];
  }

  runData(nn_input);
  requestAnimationFrame(draw);
}

function drawNumber(data) {
  const imageData = ctx.createImageData(canvas.width, canvas.height);
  for (let i = 0; i < data.length; i++) {
    for (let rgb_i = 0; rgb_i < 3; rgb_i++) {
      imageData.data[i * 4 + rgb_i] = data[i] * 255;
    }
    imageData.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}
