/*
 * Dependency-free Transformer inference for the exported teaching model.
 *
 * This favors readability over speed: each generation step recomputes the
 * visible context. The model is intentionally small enough for that trade-off.
 */

type ModelShape = {
  vocab_size: number;
  block_size: number;
  n_embd: number;
  n_head: number;
  n_layer: number;
  dropout: number;
};

type WeightEntry = {
  name: string;
  shape: number[];
  offset: number;
  length: number;
};

type ExportConfig = {
  format: string;
  model: ModelShape;
  tokenizer: { tokens: string[] };
  checkpoint: { step?: number; val_loss?: number };
  weights: WeightEntry[];
};

class WeightStore {
  private entries = new Map<string, WeightEntry>();

  constructor(private buffer: ArrayBuffer, entries: WeightEntry[]) {
    for (const entry of entries) this.entries.set(entry.name, entry);
  }

  get(name: string): Float32Array {
    const entry = this.entries.get(name);
    if (!entry) throw new Error(`模型缺少权重：${name}`);
    return new Float32Array(this.buffer, entry.offset, entry.length);
  }
}

function linearRows(
  input: Float32Array,
  rows: number,
  inputSize: number,
  weight: Float32Array,
  outputSize: number,
  bias?: Float32Array,
): Float32Array {
  const output = new Float32Array(rows * outputSize);
  for (let row = 0; row < rows; row += 1) {
    const inputBase = row * inputSize;
    const outputBase = row * outputSize;
    for (let out = 0; out < outputSize; out += 1) {
      let sum = bias ? bias[out] : 0;
      const weightBase = out * inputSize;
      for (let i = 0; i < inputSize; i += 1) {
        sum += input[inputBase + i] * weight[weightBase + i];
      }
      output[outputBase + out] = sum;
    }
  }
  return output;
}

function layerNormRows(
  input: Float32Array,
  rows: number,
  width: number,
  weight: Float32Array,
  bias: Float32Array,
): Float32Array {
  const output = new Float32Array(input.length);
  for (let row = 0; row < rows; row += 1) {
    const base = row * width;
    let mean = 0;
    for (let i = 0; i < width; i += 1) mean += input[base + i];
    mean /= width;
    let variance = 0;
    for (let i = 0; i < width; i += 1) {
      const centered = input[base + i] - mean;
      variance += centered * centered;
    }
    variance /= width;
    const inverseStd = 1 / Math.sqrt(variance + 1e-5);
    for (let i = 0; i < width; i += 1) {
      output[base + i] = (input[base + i] - mean) * inverseStd * weight[i] + bias[i];
    }
  }
  return output;
}

function add(left: Float32Array, right: Float32Array): Float32Array {
  const output = new Float32Array(left.length);
  for (let i = 0; i < left.length; i += 1) output[i] = left[i] + right[i];
  return output;
}

function gelu(value: number): number {
  const scaled = Math.sqrt(2 / Math.PI) * (value + 0.044715 * value * value * value);
  return 0.5 * value * (1 + Math.tanh(scaled));
}

export class BrowserStoryModel {
  readonly trainingStep?: number;
  readonly validationLoss?: number;
  readonly parameterCount: number;
  private config: ModelShape;
  private tokens: string[];
  private tokenToId = new Map<string, number>();
  private weights: WeightStore;

  private constructor(config: ExportConfig, buffer: ArrayBuffer) {
    if (config.format !== 'monogatari-f32-v1') throw new Error('无法识别模型格式');
    this.config = config.model;
    this.tokens = config.tokenizer.tokens;
    this.tokens.forEach((token, index) => this.tokenToId.set(token, index));
    this.weights = new WeightStore(buffer, config.weights);
    this.trainingStep = config.checkpoint.step;
    this.validationLoss = config.checkpoint.val_loss;
    this.parameterCount = config.weights.reduce((sum, entry) => sum + entry.length, 0);
  }

  static async load(): Promise<BrowserStoryModel> {
    const configResponse = await fetch('/model-config.json', { cache: 'no-store' });
    if (!configResponse.ok) throw new Error('MODEL_NOT_EXPORTED');
    const config = (await configResponse.json()) as ExportConfig;
    const weightResponse = await fetch('/model.bin', { cache: 'no-store' });
    if (!weightResponse.ok) throw new Error('MODEL_NOT_EXPORTED');
    return new BrowserStoryModel(config, await weightResponse.arrayBuffer());
  }

  private encode(text: string): number[] {
    const unknown = this.tokenToId.get('<unk>') ?? 3;
    return Array.from(text, (char) => this.tokenToId.get(char) ?? unknown);
  }

  private nextLogits(ids: number[]): Float32Array {
    const { block_size: blockSize, n_embd: width, n_head: heads, n_layer: layers, vocab_size: vocabSize } = this.config;
    const context = ids.slice(-blockSize);
    const time = context.length;
    const tokenEmbedding = this.weights.get('token_embedding.weight');
    const positionEmbedding = this.weights.get('position_embedding.weight');
    let x = new Float32Array(time * width);
    for (let t = 0; t < time; t += 1) {
      for (let i = 0; i < width; i += 1) {
        x[t * width + i] = tokenEmbedding[context[t] * width + i] + positionEmbedding[t * width + i];
      }
    }

    const headSize = width / heads;
    for (let layer = 0; layer < layers; layer += 1) {
      const prefix = `blocks.${layer}`;
      const normalized = layerNormRows(
        x,
        time,
        width,
        this.weights.get(`${prefix}.ln1.weight`),
        this.weights.get(`${prefix}.ln1.bias`),
      );
      const qkv = linearRows(
        normalized,
        time,
        width,
        this.weights.get(`${prefix}.attn.qkv.weight`),
        3 * width,
        this.weights.get(`${prefix}.attn.qkv.bias`),
      );
      const attended = new Float32Array(time * width);

      for (let head = 0; head < heads; head += 1) {
        for (let target = 0; target < time; target += 1) {
          const scores = new Float32Array(target + 1);
          let maxScore = -Infinity;
          for (let source = 0; source <= target; source += 1) {
            let score = 0;
            for (let i = 0; i < headSize; i += 1) {
              const qIndex = target * 3 * width + head * headSize + i;
              const kIndex = source * 3 * width + width + head * headSize + i;
              score += qkv[qIndex] * qkv[kIndex];
            }
            score /= Math.sqrt(headSize);
            scores[source] = score;
            if (score > maxScore) maxScore = score;
          }
          let total = 0;
          for (let source = 0; source <= target; source += 1) {
            scores[source] = Math.exp(scores[source] - maxScore);
            total += scores[source];
          }
          for (let source = 0; source <= target; source += 1) {
            const probability = scores[source] / total;
            for (let i = 0; i < headSize; i += 1) {
              const valueIndex = source * 3 * width + 2 * width + head * headSize + i;
              attended[target * width + head * headSize + i] += probability * qkv[valueIndex];
            }
          }
        }
      }

      const projected = linearRows(
        attended,
        time,
        width,
        this.weights.get(`${prefix}.attn.proj.weight`),
        width,
        this.weights.get(`${prefix}.attn.proj.bias`),
      );
      x = add(x, projected);
      const normalizedFf = layerNormRows(
        x,
        time,
        width,
        this.weights.get(`${prefix}.ln2.weight`),
        this.weights.get(`${prefix}.ln2.bias`),
      );
      const expanded = linearRows(
        normalizedFf,
        time,
        width,
        this.weights.get(`${prefix}.ff.net.0.weight`),
        4 * width,
        this.weights.get(`${prefix}.ff.net.0.bias`),
      );
      for (let i = 0; i < expanded.length; i += 1) expanded[i] = gelu(expanded[i]);
      const contracted = linearRows(
        expanded,
        time,
        4 * width,
        this.weights.get(`${prefix}.ff.net.2.weight`),
        width,
        this.weights.get(`${prefix}.ff.net.2.bias`),
      );
      x = add(x, contracted);
    }

    const final = layerNormRows(
      x,
      time,
      width,
      this.weights.get('ln_f.weight'),
      this.weights.get('ln_f.bias'),
    );
    const last = final.subarray((time - 1) * width, time * width);
    return linearRows(last, 1, width, this.weights.get('lm_head.weight'), vocabSize);
  }

  async generate(
    name: string,
    place: string,
    style: string,
    options: { maxNewTokens?: number; temperature?: number; topK?: number; onProgress?: (count: number) => void } = {},
  ): Promise<string> {
    const safeName = name.trim().replace(/[\r\n]/g, '').slice(0, 16);
    const safePlace = place.trim().replace(/[\r\n]/g, '').slice(0, 16);
    const safeStyle = style.trim().replace(/[\r\n]/g, '').slice(0, 8);
    if (!['恋愛', '文芸', 'ユーモア', '恐怖'].includes(safeStyle)) {
      throw new Error('名前、場所、作風を入力してください。');
    }
    const prefix = `名前:${safeName}\n場所:${safePlace}\n作風:${safeStyle}\n物語:${safePlace}で、${safeName}は`;
    const bos = this.tokenToId.get('<bos>') ?? 1;
    const eos = this.tokenToId.get('<eos>') ?? 2;
    const ids = [bos, ...this.encode(prefix)];
    const start = ids.length;
    const maxNewTokens = options.maxNewTokens ?? 90;
    const temperature = Math.max(options.temperature ?? 0.85, 1e-5);
    const topK = Math.max(options.topK ?? 24, 1);

    for (let step = 0; step < maxNewTokens; step += 1) {
      const logits = this.nextLogits(ids);
      const ranked = Array.from(logits, (value, index) => ({ index, value: value / temperature }))
        .sort((a, b) => b.value - a.value)
        .slice(0, Math.min(topK, logits.length));
      const max = ranked[0].value;
      const probabilities = ranked.map((item) => Math.exp(item.value - max));
      const total = probabilities.reduce((sum, value) => sum + value, 0);
      let draw = Math.random() * total;
      let next = ranked[ranked.length - 1].index;
      for (let i = 0; i < ranked.length; i += 1) {
        draw -= probabilities[i];
        if (draw <= 0) {
          next = ranked[i].index;
          break;
        }
      }
      if (next === eos) break;
      ids.push(next);
      options.onProgress?.(step + 1);
      if (step % 2 === 0) await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    }

    const specials = new Set(['<pad>', '<bos>', '<eos>', '<unk>']);
    const continuation = ids.slice(start).map((id) => this.tokens[id]).filter((token) => !specials.has(token)).join('');
    return `${safePlace}で、${safeName}は${continuation}`;
  }
}
