class Renderer {
  constructor(canvas, width, height, margin) {
    this.canvas = canvas;
    this.width = width;
    this.height = height;
    this.margin = margin;

    this.canvas.width = width;
    this.canvas.height = height;
    this.ctx = this.canvas.getContext("2d");

    this.numPoints = 0;
    this.numLines = 0;
    this.points = [];
    this.lines = [];
  }

  clearPoints() {
    this.numPoints = 0;
    this.points = [];
  }

  clearLines() {
    this.numLines = 0;
    this.lines = [];
  }

  addPoint(x, y) {
    this.points.push(x);
    this.points.push(y);
    this.numPoints++;
  }

  addLine(x0, y0, x1, y1) {
    this.lines.push(x0);
    this.lines.push(y0);
    this.lines.push(x1);
    this.lines.push(y1);
    this.numLines++;
  }

  render() {
    this.ctx.fillStyle = "#2c3344";
    this.ctx.fillRect(0, 0, this.width, this.height);

    this.ctx.save();
    this.ctx.translate(this.width / 2, this.height / 2);

    this.ctx.strokeStyle = "#8345c1";
    this.ctx.lineWidth = 4;
    for (let i = 0; i < this.numLines; i++) {
      this.ctx.beginPath();
      this.ctx.moveTo(this.lines[i * 4 + 0] * (this.width - this.margin * 2) / 2, this.lines[i * 4 + 1] * -1 * (this.height - this.margin * 2) / 2);
      this.ctx.lineTo(this.lines[i * 4 + 2] * (this.width - this.margin * 2) / 2, this.lines[i * 4 + 3] * -1 * (this.height - this.margin * 2) / 2);
      this.ctx.stroke();
    }

    this.ctx.fillStyle = "#929293";
    for (let i = 0; i < this.numPoints; i++) {
      this.ctx.beginPath();
      this.ctx.arc(this.points[i * 2 + 0] * (this.width - this.margin * 2) / 2, this.points[i * 2 + 1] * -1 * (this.height - this.margin * 2) / 2, 2, 0, Math.PI * 2);
      this.ctx.fill();
    }
    this.ctx.restore();
  }
}