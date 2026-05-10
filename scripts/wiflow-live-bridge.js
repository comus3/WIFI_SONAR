#!/usr/bin/env node
'use strict';
const fs = require('fs'), path = require('path'), http = require('http');
const WebSocket = require('ws');

function createRng(seed) { let s = seed || 42; return () => { s = (s * 1664525 + 1013904223) | 0; return (s >>> 0) / 4294967296; }; }
function glorotUniform(shape, rng) { const limit = Math.sqrt(6 / (shape[0] + shape[1])); const arr = new Float32Array(shape[0] * shape[1]); for (let i = 0; i < arr.length; i++) arr[i] = (rng() * 2 - 1) * limit; return arr; }
function zeros(shape) { return new Float32Array(shape[0] * (shape[1] || 1)); }

class Conv1d {
  constructor(inCh, outCh, kernel, dilation, rng) { this.inCh = inCh; this.outCh = outCh; this.kernel = kernel; this.dilation = dilation || 1; this.weight = glorotUniform([outCh, inCh * kernel], rng); this.bias = zeros([outCh]); }
  forward(input, T) { const {inCh, outCh, kernel, dilation} = this; const ek = kernel + (kernel - 1) * (dilation - 1); const pad = ek - 1, Tp = T + pad; const padded = new Float32Array(inCh * Tp); for (let c = 0; c < inCh; c++) for (let t = 0; t < T; t++) padded[c * Tp + t + pad] = input[c * T + t]; const out = new Float32Array(outCh * T); for (let oc = 0; oc < outCh; oc++) for (let t = 0; t < T; t++) { let s = this.bias[oc]; for (let ic = 0; ic < inCh; ic++) for (let k = 0; k < kernel; k++) { const ti = t + pad - k * dilation; if (ti >= 0 && ti < Tp) s += this.weight[oc * inCh * kernel + ic * kernel + k] * padded[ic * Tp + ti]; } out[oc * T + t] = s; } return out; }
}

class BatchNorm1d {
  constructor(ch) { this.ch = ch; this.gamma = new Float32Array(ch).fill(1); this.beta = zeros([ch]); this.eps = 1e-5; }
  forward(input, T) { const out = new Float32Array(input.length); for (let c = 0; c < this.ch; c++) { let m = 0, v = 0; for (let t = 0; t < T; t++) m += input[c * T + t]; m /= T; for (let t = 0; t < T; t++) v += (input[c * T + t] - m) ** 2; v /= T; const inv = 1 / Math.sqrt(v + this.eps); for (let t = 0; t < T; t++) out[c * T + t] = this.gamma[c] * (input[c * T + t] - m) * inv + this.beta[c]; } return out; }
}

class TCNBlock {
  constructor(inCh, outCh, kernel, dilation, rng) { this.inCh = inCh; this.outCh = outCh; this.conv1 = new Conv1d(inCh, outCh, kernel, dilation, rng); this.bn1 = new BatchNorm1d(outCh); this.conv2 = new Conv1d(outCh, outCh, kernel, dilation, rng); this.bn2 = new BatchNorm1d(outCh); this.hasResProj = inCh !== outCh; if (this.hasResProj) this.resConv = new Conv1d(inCh, outCh, 1, 1, rng); }
  forward(x, T) { let h = this.conv1.forward(x, T); h = this.bn1.forward(h, T); for (let i = 0; i < h.length; i++) if (h[i] < 0) h[i] = 0; let h2 = this.conv2.forward(h, T); h2 = this.bn2.forward(h2, T); for (let i = 0; i < h2.length; i++) if (h2[i] < 0) h2[i] = 0; if (this.hasResProj) { const r = this.resConv.forward(x, T); for (let i = 0; i < h2.length; i++) h2[i] += r[i]; } else { for (let i = 0; i < h2.length; i++) h2[i] += x[i]; } return h2; }
  collectParams() { const p = [{weight:this.conv1.weight},{weight:this.conv1.bias},{weight:this.bn1.gamma},{weight:this.bn1.beta},{weight:this.conv2.weight},{weight:this.conv2.bias},{weight:this.bn2.gamma},{weight:this.bn2.beta}]; if (this.hasResProj) p.push({weight:this.resConv.weight},{weight:this.resConv.bias}); return p; }
}

class Linear {
  constructor(inF, outF, rng) { this.weight = glorotUniform([outF, inF], rng); this.bias = zeros([outF]); }
  forward(x) { const out = new Float32Array(this.bias.length); for (let o = 0; o < this.bias.length; o++) { let s = this.bias[o]; for (let i = 0; i < x.length; i++) s += x[i] * this.weight[o * x.length + i]; out[o] = s; } return out; }
  collectParams() { return [{weight:this.weight},{weight:this.bias}]; }
}

class WiFlowModel {
  constructor(inputDim, timeSteps, numKP, sc, rng) { this.timeSteps = timeSteps; this.numKP = numKP || 17; const ch = sc.tcnChannels, k = sc.kernel || 3, dil = [1, 2, 4, 8], nb = sc.tcnBlocks || 2; this.tcnBlocks = []; let p = inputDim; for (let i = 0; i < nb; i++) { this.tcnBlocks.push(new TCNBlock(p, ch[i], k, dil[i], rng)); p = ch[i]; } const fd = p * timeSteps, hd = sc.hiddenDim || 256; this.fc1 = new Linear(fd, hd, rng); this.fc2 = new Linear(hd, (numKP || 17) * 2, rng); }
  collectParams() { let pp = []; for (const b of this.tcnBlocks) pp = pp.concat(b.collectParams()); return pp.concat(this.fc1.collectParams(), this.fc2.collectParams()); }
  loadWeights(fw) { const params = this.collectParams(); let off = 0; for (const pw of params) { for (let i = 0; i < pw.weight.length && off + i < fw.length; i++) pw.weight[i] = fw[off + i]; off += pw.weight.length; } return off; }
  forward(csi) { const T = this.timeSteps; let x = csi; for (const b of this.tcnBlocks) x = b.forward(x, T); let h = this.fc1.forward(x); for (let i = 0; i < h.length; i++) if (h[i] < 0) h[i] = 0; let out = this.fc2.forward(h); for (let i = 0; i < out.length; i++) out[i] = 1 / (1 + Math.exp(-out[i])); return out; }
}

// ── Main ─────────────────────────────────────────────────
const args = process.argv.slice(2);
const modelPath = args.find(a => a.endsWith('.json')) || 'models/wiflow-supervised/wiflow-v1.json';
const wsUrl = args.find(a => a.startsWith('ws')) || 'ws://localhost:3001/ws/sensing';
const PORT = parseInt(args.find(a => /^\d+$/.test(a))) || 3002;
const UI_DIR = path.join(__dirname, '..', 'ui');

console.log('=== WiFlow Bridge v6 (UI + Pose) ===');

// ── Load model ────────────────────────────────────────────
const modelJson = JSON.parse(fs.readFileSync(modelPath, 'utf-8'));
const arch = modelJson.architecture;
const weights = new Float32Array(Buffer.from(modelJson.weightsBase64, 'base64').buffer);

const sc = { tcnChannels: arch.tcnChannels || [32,32,32,32], kernel: arch.tcnKernel || 3, tcnBlocks: (arch.tcnChannels || [32,32,32,32]).length, hiddenDim: arch.hiddenDim || 256 };
const rng = createRng(42);
const model = new WiFlowModel(arch.inputDim, arch.timeSteps, arch.numKeypoints, sc, rng);
model.loadWeights(weights);
console.log(`Model loaded, test nose=(${model.forward(new Float32Array(arch.inputDim*arch.timeSteps).map(()=>Math.random()))[0].toFixed(2)},...)`);

// ── CSI buffer + calibration ──────────────────────────────
const buf = [], CALIB = 100;
let latestPose = null;
// Temporal smoothing: exponential moving average per keypoint
const EMA_ALPHA = 0.35; // higher = more responsive, lower = smoother
let smoothedPose = null;

const names = ['nose','l_eye','r_eye','l_ear','r_ear','l_shoulder','r_shoulder','l_elbow','r_elbow','l_wrist','r_wrist','l_hip','r_hip','l_knee','r_knee','l_ankle','r_ankle'];

function emaPose(newPose) {
  if (!smoothedPose || smoothedPose.length !== newPose.length) {
    smoothedPose = new Float32Array(newPose);
    return newPose;
  }
  for (let i = 0; i < newPose.length; i++) {
    smoothedPose[i] = smoothedPose[i] * (1 - EMA_ALPHA) + newPose[i] * EMA_ALPHA;
  }
  return smoothedPose;
}

function fuseNodeAmplitudes(nodes) {
  // Average amplitude across all active nodes
  const active = nodes.filter(n => n.amplitude && n.amplitude.length > 0);
  if (active.length === 0) return null;
  const dim = arch.inputDim;
  const fused = new Float32Array(dim);
  for (let i = 0; i < dim; i++) {
    let sum = 0, count = 0;
    for (const n of active) {
      if (i < n.amplitude.length) { sum += n.amplitude[i]; count++; }
    }
    fused[i] = count > 0 ? sum / count : 0;
  }
  return fused;
}

let fc = 0;
const calibSum = new Float64Array(arch.inputDim), calibSumSq = new Float64Array(arch.inputDim);
let calibN = 0, ready = false;
const liveM = new Float32Array(arch.inputDim), liveS = new Float32Array(arch.inputDim).fill(1);

function processCSI(amp) {
  const dim = arch.inputDim;
  const feat = new Float32Array(dim);
  for (let i = 0; i < Math.min(dim, amp.length); i++) feat[i] = amp[i] || 0;
  if (!ready) {
    for (let i = 0; i < dim; i++) { calibSum[i] += feat[i]; calibSumSq[i] += feat[i] * feat[i]; }
    if (++calibN >= CALIB) {
      for (let i = 0; i < dim; i++) { const m = calibSum[i] / calibN; liveM[i] = m; liveS[i] = Math.sqrt(Math.max(calibSumSq[i] / calibN - m * m, 1e-8)); }
      ready = true; console.log(`Calibrated: mean[${Math.min(...liveM).toFixed(1)},${Math.max(...liveM).toFixed(1)}]`);
    }
    return null;
  }
  for (let i = 0; i < dim; i++) { const v = (feat[i] - liveM[i]) / liveS[i]; feat[i] = isNaN(v) || !isFinite(v) ? 0 : v; }
  buf.push(feat);
  if (buf.length > arch.timeSteps) buf.shift();
  if (buf.length < arch.timeSteps) return null;
  const csi = new Float32Array(dim * arch.timeSteps);
  for (let t = 0; t < arch.timeSteps; t++) for (let s = 0; s < dim; s++) csi[t * dim + s] = buf[t][s];
  return model.forward(csi);
}

// ── Connect to sensing server ─────────────────────────────
const backendWs = new WebSocket(wsUrl, { rejectUnauthorized: false });
backendWs.on('open', () => console.log('Connected to backend'));
backendWs.on('error', e => console.error('Backend err:', e.message));

backendWs.on('message', data => {
  try {
    const msg = JSON.parse(data.toString());
    if (msg.type === 'sensing_update' && msg.nodes && msg.nodes.length > 0) {
      // Fuse amplitudes from all active nodes
      const fused = fuseNodeAmplitudes(msg.nodes);
      if (fused) {
        const pose = processCSI(fused);
        if (pose !== null) {
          const smooth = emaPose(pose);
          latestPose = smooth;
          fc++;
          if (fc % 200 === 1) console.log(`#${fc} nose=(${smooth[0].toFixed(2)},${smooth[1].toFixed(2)}) nodes=${msg.nodes.length}`);
        }
      }
    }
    // Forward enriched message to all UI clients
    const enriched = latestPose ? enrichMessage(msg) : msg;
    const str = JSON.stringify(enriched);
    uiClients.forEach(c => { try { c.send(str); } catch(e) {} });
  } catch(e) {}
});

function enrichMessage(msg) {
  if (!latestPose) return msg;
  const kps = [];
  for (let i = 0; i < arch.numKeypoints; i++)
    kps.push([latestPose[i*2], latestPose[i*2+1], 0, 0.7]);
  return { ...msg, pose_keypoints: kps, pose_source: 'wiflow_model' };
}

// ── HTTP server + UI WebSocket ────────────────────────────
const uiClients = new Set();
const MIME = { '.html': 'text/html', '.js': 'application/javascript', '.css': 'text/css', '.png': 'image/png', '.svg': 'image/svg+xml', '.json': 'application/json', '.wasm': 'application/wasm' };

const server = http.createServer((req, res) => {
  // Proxy /health and /api requests to the real sensing server
  if (req.url.startsWith('/health') || req.url.startsWith('/api')) {
    const opts = { hostname: 'localhost', port: 3000, path: req.url, method: req.method, headers: req.headers };
    const proxy = http.request(opts, (proxyRes) => {
      res.writeHead(proxyRes.statusCode, proxyRes.headers);
      proxyRes.pipe(res);
    });
    proxy.on('error', () => { res.writeHead(502); res.end('Backend unreachable'); });
    req.pipe(proxy);
    return;
  }
  let filePath = path.join(UI_DIR, req.url === '/' ? '/index.html' : req.url.split('?')[0]);
  const ext = path.extname(filePath);
  try {
    const content = fs.readFileSync(filePath);
    res.writeHead(200, { 'Content-Type': MIME[ext] || 'application/octet-stream' });
    res.end(content);
  } catch(e) {
    res.writeHead(404); res.end('Not found');
  }
});

const wss = new WebSocket.Server({ server });
wss.on('connection', ws => {
  uiClients.add(ws);
  ws.on('close', () => uiClients.delete(ws));
});

server.listen(PORT, () => console.log(`UI + Pose: http://localhost:${PORT}`));

