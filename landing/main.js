const canvas = document.getElementById("block-scene");
const ctx = canvas.getContext("2d");

const palette = {
  top: "#e8e7dd",
  topEdge: "#f7f3ea",
  left: "#79c7cd",
  leftDeep: "#319baa",
  right: "#0b3e75",
  rightDeep: "#061f4f",
  line: "rgba(16,24,32,0.08)",
};

const blocks = [
  { x: 0.62, y: 0.22, w: 150, h: 82, d: 58, delay: 0.0 },
  { x: 0.73, y: 0.34, w: 178, h: 88, d: 64, delay: 0.7 },
  { x: 0.58, y: 0.48, w: 176, h: 80, d: 56, delay: 1.3 },
  { x: 0.75, y: 0.58, w: 130, h: 92, d: 70, delay: 2.0 },
  { x: 0.64, y: 0.68, w: 106, h: 106, d: 78, delay: 2.6 },
  { x: 0.84, y: 0.47, w: 74, h: 74, d: 52, delay: 3.2 },
];

let pointerX = 0;
let pointerY = 0;

window.addEventListener("pointermove", (event) => {
  pointerX = (event.clientX / Math.max(window.innerWidth, 1) - 0.5) * 2;
  pointerY = (event.clientY / Math.max(window.innerHeight, 1) - 0.5) * 2;
});

function resize() {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(window.innerWidth * ratio);
  canvas.height = Math.floor(window.innerHeight * ratio);
  canvas.style.width = `${window.innerWidth}px`;
  canvas.style.height = `${window.innerHeight}px`;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

window.addEventListener("resize", resize);
resize();

function isoPoint(x, y, z) {
  return {
    x: (x - y) * 0.86,
    y: (x + y) * 0.46 - z,
  };
}

function shade(colorTop, colorBottom, x1, y1, x2, y2) {
  const gradient = ctx.createLinearGradient(x1, y1, x2, y2);
  gradient.addColorStop(0, colorTop);
  gradient.addColorStop(1, colorBottom);
  return gradient;
}

function polygon(points, fill) {
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (const point of points.slice(1)) ctx.lineTo(point.x, point.y);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
}

function drawBlock(cx, cy, w, h, d, t) {
  const z = Math.sin(t) * 14;
  const p = {
    a: isoPoint(-w / 2, -h / 2, d + z),
    b: isoPoint(w / 2, -h / 2, d + z),
    c: isoPoint(w / 2, h / 2, d + z),
    d: isoPoint(-w / 2, h / 2, d + z),
    e: isoPoint(-w / 2, -h / 2, z),
    f: isoPoint(w / 2, -h / 2, z),
    g: isoPoint(w / 2, h / 2, z),
    h: isoPoint(-w / 2, h / 2, z),
  };

  for (const key of Object.keys(p)) {
    p[key].x += cx;
    p[key].y += cy;
  }

  ctx.save();
  ctx.shadowColor = "rgba(16,24,32,0.14)";
  ctx.shadowBlur = 30;
  ctx.shadowOffsetY = 24;

  polygon(
    [p.d, p.c, p.g, p.h],
    shade(palette.left, palette.leftDeep, p.d.x, p.d.y, p.g.x, p.g.y),
  );
  polygon(
    [p.b, p.c, p.g, p.f],
    shade(palette.right, palette.rightDeep, p.b.x, p.b.y, p.g.x, p.g.y),
  );
  polygon(
    [p.a, p.b, p.c, p.d],
    shade(palette.topEdge, palette.top, p.a.x, p.a.y, p.c.x, p.c.y),
  );

  ctx.shadowColor = "transparent";
  ctx.strokeStyle = "rgba(255,255,255,0.36)";
  ctx.lineWidth = 1;
  for (const face of [
    [p.a, p.b, p.c, p.d],
    [p.d, p.c, p.g, p.h],
    [p.b, p.c, p.g, p.f],
  ]) {
    ctx.beginPath();
    ctx.moveTo(face[0].x, face[0].y);
    for (const point of face.slice(1)) ctx.lineTo(point.x, point.y);
    ctx.closePath();
    ctx.stroke();
  }
  ctx.restore();
}

function drawGrid(width, height, time) {
  ctx.save();
  ctx.translate(width * 0.66 + pointerX * 12, height * 0.53 + pointerY * 10);
  ctx.rotate(-0.58);
  ctx.scale(1, 0.52);
  ctx.strokeStyle = palette.line;
  ctx.lineWidth = 1;

  const size = 46;
  const span = 680;
  const offset = (time * 10) % size;
  for (let x = -span; x <= span; x += size) {
    ctx.beginPath();
    ctx.moveTo(x + offset, -span);
    ctx.lineTo(x + offset, span);
    ctx.stroke();
  }
  for (let y = -span; y <= span; y += size) {
    ctx.beginPath();
    ctx.moveTo(-span, y + offset);
    ctx.lineTo(span, y + offset);
    ctx.stroke();
  }
  ctx.restore();
}

function draw(timeMs) {
  const time = timeMs / 1000;
  const width = window.innerWidth;
  const height = window.innerHeight;
  ctx.clearRect(0, 0, width, height);
  drawGrid(width, height, time);

  for (const block of blocks) {
    const x = width * block.x + pointerX * 18;
    const y = height * block.y + pointerY * 14;
    drawBlock(x, y, block.w, block.h, block.d, time + block.delay);
  }

  requestAnimationFrame(draw);
}

requestAnimationFrame(draw);
