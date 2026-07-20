# TinyModelZ — Comprehensive Codebase Audit

> **Purpose**: Actionable instructions for an agent to fix bugs, eliminate inefficiencies, and improve performance — without changing the project's architecture or goals (a from-scratch GPT in Java with CPU/OpenCL/CUDA backends and KV-cache inference).

---

## Legend

| Severity | Meaning |
|----------|---------|
| 🔴 BUG | Produces incorrect results or crashes |
| 🟠 PERF | Measurable performance bottleneck |
| 🟡 WASTE | Unnecessary allocation / redundant work |
| 🔵 QUALITY | Code quality, maintainability, robustness |

---

## 1. Tensor.java — Core Math Engine

### 🔴 1.1 — `Tensor.cat()` calls `toContiguous()` inside a hot loop

**File**: [Tensor.java](file:///home/umar/Documents/tinyModelZ/src/main/java/com/tinymodelz/math/Tensor.java#L570-L578)

**Problem**: Inside the nested `for (int o = 0; o < outerSize; o++) / for (Tensor t : tensors)` loop, `t.toContiguous()` is called on every iteration of the outer loop. For a KV cache concat of 100 cached tokens, this materializes the *same* contiguous copy `outerSize` times per tensor.

**Fix**: Hoist `toContiguous()` calls outside both loops. Create a `List<Tensor>` of pre-contiguous tensors before the loop:
```java
List<Tensor> contTensors = new ArrayList<>(tensors.size());
for (Tensor t : tensors) contTensors.add(t.toContiguous());
// then use contTensors inside the loop
```

---

### 🟠 1.2 — `isContiguous()` recomputes strides on every call

**File**: [Tensor.java:L209-L214](file:///home/umar/Documents/tinyModelZ/src/main/java/com/tinymodelz/math/Tensor.java#L209-L214)

**Problem**: `isContiguous()` calls `computeStrides(shape)` and compares element-by-element. This is called from `toContiguous()`, `add()`, `subtract()`, `multiply()`, `accumulateGrad()`, `gelu()`, and AdamW — dozens of times per forward+backward step. Each call allocates a new `int[]`.

**Fix**: Compute and cache a `boolean isContiguous` field at construction time. Invalidate only in constructors (all tensors are immutable after construction, so this is safe):
```java
private final boolean contiguous; // set in every constructor
public boolean isContiguous() { return contiguous; }
```

---

### 🟠 1.3 — `getShape()` clones the array on every call

**File**: [Tensor.java:L125-L127](file:///home/umar/Documents/tinyModelZ/src/main/java/com/tinymodelz/math/Tensor.java#L125-L127)

**Problem**: `getShape()` returns `shape.clone()` every time. This is called extremely frequently (every `forward()`, every `reshape()`, every matmul dimension check). Most callers only read the values.

**Fix**: Add a package-private `int[] shapeRef()` that returns the raw array for internal use. Keep `getShape()` with clone for public API safety:
```java
int[] shapeRef() { return shape; } // package-private, no clone
public int[] getShape() { return shape.clone(); } // public API
```
Then update all internal callers (MultiHeadAttention, TransformerBlock, FeedForward, TinyGPT, CrossEntropyLoss) to use `shapeRef()`.

---

### 🟠 1.4 — `parallel()` streams on small tensors cause overhead

**File**: [Tensor.java:L418](file:///home/umar/Documents/tinyModelZ/src/main/java/com/tinymodelz/math/Tensor.java#L418), [L454](file:///home/umar/Documents/tinyModelZ/src/main/java/com/tinymodelz/math/Tensor.java#L454), [L494](file:///home/umar/Documents/tinyModelZ/src/main/java/com/tinymodelz/math/Tensor.java#L494), [L660](file:///home/umar/Documents/tinyModelZ/src/main/java/com/tinymodelz/math/Tensor.java#L660), [L884](file:///home/umar/Documents/tinyModelZ/src/main/java/com/tinymodelz/math/Tensor.java#L884)

**Problem**: `IntStream.range(0, size).parallel()` is used for `add`, `subtract`, `multiply`, `matmul`, `gelu`, and `accumulateGrad`. For small tensors (e.g. bias vectors of size 64), ForkJoinPool overhead *exceeds* the computation time.

**Fix**: Add a threshold guard (e.g. `size > 8192`) before using `.parallel()`:
```java
private static final int PARALLEL_THRESHOLD = 8192;
// In add(), subtract(), multiply(), gelu():
if (outSize > PARALLEL_THRESHOLD) {
    IntStream.range(0, outSize).parallel().forEach(i -> { ... });
} else {
    for (int i = 0; i < outSize; i++) { ... }
}
```

---

### 🟡 1.5 — `multiply(float val)` creates a full scalar Tensor

**File**: [Tensor.java:L796-L798](file:///home/umar/Documents/tinyModelZ/src/main/java/com/tinymodelz/math/Tensor.java#L796-L798)

**Problem**: `multiply(float val)` delegates to `multiply(Tensor.scalar(val))` which triggers full broadcasting machinery (shape comparison, `broadcastShapes`, `getBroadcastedFlatIndex` per element). This is called in every attention score scaling step.

**Fix**: Add a fast-path scalar multiply that directly scales the data array:
```java
public Tensor multiply(float val) {
    float[] outData = new float[size];
    Tensor cont = this.toContiguous();
    for (int i = 0; i < size; i++) outData[i] = cont.data[cont.offset + i] * val;
    Tensor result = new Tensor(outData, shape);
    if (this.requiresGrad) {
        result.requiresGrad = true;
        result.creators = List.of(this);
        result.opName = "scale";
        result.backwardFn = (gradOutput) -> {
            float[] g = new float[gradOutput.length];
            for (int i = 0; i < g.length; i++) g[i] = gradOutput[i] * val;
            this.accumulateGrad(g);
        };
    }
    return result;
}
```
Apply the same optimization to `add(float val)`.

---

### 🔴 1.6 — `backward()` initializes grad to `1.0f` for entire `data.length`, not `size`

**File**: [Tensor.java:L991-L995](file:///home/umar/Documents/tinyModelZ/src/main/java/com/tinymodelz/math/Tensor.java#L991-L995)

**Problem**: `Arrays.fill(grad, 1.0f)` fills the entire `grad` array (which is `data.length`), not just the logical `size` elements. For non-zero-offset tensor views, this writes garbage gradient values into regions belonging to other logical tensors sharing the same backing array.

**Fix**:
```java
Arrays.fill(grad, offset, offset + size, 1.0f);
```

---

### 🔵 1.7 — `Tensor.cat()` lacks autograd support

**File**: [Tensor.java:L539-L581](file:///home/umar/Documents/tinyModelZ/src/main/java/com/tinymodelz/math/Tensor.java#L539-L581)

**Problem**: `Tensor.cat()` does not set `requiresGrad`, `creators`, or `backwardFn`. This means if KV cache is ever used during training (fine-tuning with cache), gradients will silently stop flowing through the concatenation boundary. Currently safe for inference-only KV cache, but fragile.

**Fix**: Not urgent for inference. If training with KV cache is planned, add a backward function that slices `gradOutput` back into per-tensor gradient chunks.

---

## 2. MultiHeadAttention.java — KV Cache Correctness

### 🔴 2.1 — Causal mask is wrong shape when KV cache is used with multi-token prompt

**File**: [MultiHeadAttention.java:L122-L124](file:///home/umar/Documents/tinyModelZ/src/main/java/com/tinymodelz/nn/MultiHeadAttention.java)

**Problem**: When processing a prompt of length $T > 1$ with KV cache, `scores` has shape `[B, H, T, T]` (since cache is empty on first call, so `kSplit` also has dim-2 = T). The causal mask created in `TinyGPT.forwardWithCache()` is `createCausalMask(T)` which has shape `[T, T]`. This works correctly for the first prompt pass.

However, if `forwardWithCache` is later called with `T > 1` **and** the cache already has entries (e.g., continuing a multi-chunk prompt), `scores` will have shape `[B, H, T, cached_T + T]` but `mask` will still be `[T, T]` — a shape mismatch in `maskedFill`. The current `T > 1` guard in `forwardWithCache` would apply the wrong-shaped mask.

**Fix**: When `layerCache != null` and cache already has entries, either:
- Build a properly shaped causal mask of size `[T, cached_T + T]` where position `i` can attend to all positions `<= startPos + i`, OR
- Skip masking entirely when cache is populated (since all cached positions are causally valid by construction — they are strictly in the past).

Recommended approach (simpler and correct):
```java
// In forwardWithCache: only apply mask when cache is empty (first call)
if (mask != null && T > 1 && layerCache != null 
    && layerCache.keyCache == null) {
    scores = scores.maskedFill(mask, -1e9f);
} else if (mask != null && layerCache == null) {
    scores = scores.maskedFill(mask, -1e9f);
}
```

---

## 3. Generator.java — KV Cache Integration

### 🔴 3.1 — No context window overflow protection with KV cache

**File**: [Generator.java:L78-L130](file:///home/umar/Documents/tinyModelZ/src/main/java/com/tinymodelz/inference/Generator.java#L78-L130)

**Problem**: The KV-cached path has no `seqLen` check. If `maxNewTokens` is large enough that `tokenIds.size() > maxSeqLen`, `forwardWithCache` will throw at line `startPos + T > maxSeqLen`. The non-cached path correctly truncates with `Math.max(0, tokenIds.size() - seqLen)`.

**Fix**: Add an overflow guard before the cached forward call:
```java
int currentPos = tokenIds.size() - 1;
if (currentPos >= seqLen) {
    // Context window exceeded; cannot extend KV cache further
    break;
}
logits = gpt.forwardWithCache(nextTokenTensor, kvCache, currentPos);
```

---

### 🟡 3.2 — Repetition penalty code is duplicated 3 times

**File**: [Generator.java](file:///home/umar/Documents/tinyModelZ/src/main/java/com/tinymodelz/inference/Generator.java)

**Problem**: The repetition penalty loop appears identically in the cached path and the fallback path.

**Fix**: Extract into a private method:
```java
private void applyRepetitionPenalty(float[] logits, List<Integer> tokenIds, float penalty) { ... }
```

---

## 4. DataLoader.java — Memory Inefficiency

### 🟠 4.1 — `reset()` rebuilds the entire offset list every epoch

**File**: [DataLoader.java:L55-L68](file:///home/umar/Documents/tinyModelZ/src/main/java/com/tinymodelz/train/DataLoader.java#L55-L68)

**Problem**: `reset()` creates a new `ArrayList` with potentially millions of integers (one per valid starting position). For a 10M-token dataset with seqLen=64, this allocates ~10M Integer objects every epoch.

**Fix**: Build the list once in the constructor. On `reset()`, only reshuffle and reset cursor:
```java
public void reset() {
    if (shuffle) Collections.shuffle(startOffsets);
    cursor = 0;
}
```
Move the list creation to the constructor body.

---

### 🟠 4.2 — `getTokenIds()` exposes internal mutable array

**File**: [TextDataset.java:L56-L58](file:///home/umar/Documents/tinyModelZ/src/main/java/com/tinymodelz/train/TextDataset.java#L56-L58)

**Problem**: `DataLoader.nextBatch()` and `DataLoader.reset()` both call `dataset.getTokenIds()` which returns the raw internal array. `nextBatch()` is called per batch, so this is called thousands of times per epoch. Currently harmless since nobody mutates it, but fragile.

**Fix**: Cache the reference in `DataLoader`'s constructor:
```java
private final int[] tokenIds; // cached in constructor
```

---

## 5. Embedding.java — Missing Optimization

### 🟠 5.1 — Embedding lookup uses `getValByFlatIndex()` per element

**File**: [Embedding.java:L58-L67](file:///home/umar/Documents/tinyModelZ/src/main/java/com/tinymodelz/nn/Embedding.java#L58-L67)

**Problem**: `weight.getValByFlatIndex(tokenIdx * embeddingDim + d)` calls `getContiguousToPhysicalOffset()` per element, which decomposes the flat index into coordinates. The embedding weight is always contiguous (created via `Tensor.fromMatrix`), so this decomposition is wasted work.

**Fix**: Use direct array access:
```java
float[] wData = weight.getData();
int wOffset = weight.offset();
// in loop:
outData[outOff + d] = wData[wOffset + tokenIdx * embeddingDim + d];
```
Or better, use `System.arraycopy`:
```java
System.arraycopy(wData, wOffset + tokenIdx * embeddingDim, outData, outOff, embeddingDim);
```

---

## 6. LayerNorm.java — Synchronization Overhead

### 🟠 6.1 — `synchronized` on gradient arrays kills parallelism

**File**: [LayerNorm.java:L126-L143](file:///home/umar/Documents/tinyModelZ/src/main/java/com/tinymodelz/nn/LayerNorm.java#L126-L143)

**Problem**: The backward pass for weight/bias gradients uses `synchronized(wGrad)` and `synchronized(bGrad)` inside a sequential loop. This means every iteration of the inner `j` loop contends on the same lock. The gradient accumulation for weight and bias should either:
1. Use thread-local accumulators and reduce, or
2. Run the N-loop sequentially (it already is sequential, so remove the `synchronized` which is pointless overhead).

**Fix**: The N-loop is already sequential (no `.parallel()`), so the `synchronized` blocks add pure overhead. Remove them:
```java
// Remove synchronized blocks — this loop is already sequential
if (weight.requiresGrad()) {
    wGrad[wOffset + j] += dy * xh;
}
```

---

## 7. RotaryEmbedding.java — Not Integrated

### 🔵 7.1 — RoPE exists but is never used by MultiHeadAttention

**File**: [RotaryEmbedding.java](file:///home/umar/Documents/tinyModelZ/src/main/java/com/tinymodelz/nn/RotaryEmbedding.java)

**Problem**: `RotaryEmbedding` is implemented but `MultiHeadAttention` uses absolute learned positional embeddings instead of RoPE. RoPE would provide better length generalization and is the standard in modern architectures (LLaMA, Mistral).

**Fix** (optional enhancement): Integrate RoPE by applying it to Q and K tensors inside `forwardWithCache`:
```java
RotaryEmbedding rope = new RotaryEmbedding(headDim);
qSplit = rope.apply(qSplit);
kSplit = rope.apply(kSplit);
```
This would also need a `startPos` parameter for correct position indexing with KV cache.

---

## 8. Dropout.java — Unnecessary Allocation

### 🟡 8.1 — `mask` array is allocated but only used in backward

**File**: [Dropout.java:L53](file:///home/umar/Documents/tinyModelZ/src/main/java/com/tinymodelz/nn/Dropout.java#L53)

**Problem**: `float[] mask = new float[size]` is always allocated even when `requiresGrad` is false (inference mode already returns early, but during training without grad tracking on input, the mask is wasted).

**Fix**: Only allocate `mask` when the input requires grad:
```java
float[] mask = x.requiresGrad() ? new float[size] : null;
// In the loop, only write to mask if non-null
```

---

## 9. AdamW.java — Weight Decay Ordering

### 🔴 9.1 — Weight decay is applied before the Adam update (incorrect AdamW)

**File**: [AdamW.java:L86-L88](file:///home/umar/Documents/tinyModelZ/src/main/java/com/tinymodelz/train/AdamW.java#L86-L88)

**Problem**: The current code applies weight decay (`w -= lr * weightDecay * w`) *before* computing the Adam momentum update, then uses the decayed `w` for the final parameter update (`data[i] = w - ...`). In the canonical AdamW paper (Loshchilov & Hutter, 2017), weight decay is applied *after* the Adam update:
```
θ_t = θ_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps)) - lr * λ * θ_{t-1}
```
The current ordering causes the weight decay to compound with the Adam step differently than intended.

**Fix**: Reorder so weight decay is applied last:
```java
float adamUpdate = (lr / ((float) Math.sqrt(vHat) + eps)) * mHat;
data[i] = w - adamUpdate - lr * weightDecay * w;
```

---

## 10. CrossEntropyLoss.java — Double Exponentiation

### 🟡 10.1 — `Math.exp()` is computed twice per element

**File**: [CrossEntropyLoss.java:L70-L78](file:///home/umar/Documents/tinyModelZ/src/main/java/com/tinymodelz/nn/CrossEntropyLoss.java#L70-L78)

**Problem**: First loop computes `sumExp += Math.exp(...)`. Second loop computes `probs[j] = Math.exp(...) / sumExp`. Each element's `exp()` is computed twice.

**Fix**: Store the exp values in the first pass:
```java
for (int j = 0; j < V; j++) {
    float expVal = (float) Math.exp(logitsData[i * V + j] - maxLogit);
    probs[i * V + j] = expVal;
    sumExp += expVal;
}
for (int j = 0; j < V; j++) {
    probs[i * V + j] /= sumExp;
}
```

---

## 11. KVCache — Growing Memory Without Bound

### 🟠 11.1 — `Tensor.cat()` copies the entire cache on every decode step

**File**: [MultiHeadAttention.java](file:///home/umar/Documents/tinyModelZ/src/main/java/com/tinymodelz/nn/MultiHeadAttention.java) (cache accumulation block)

**Problem**: Every decode step calls `Tensor.cat(List.of(cachedK, newK), 2)` which allocates a new tensor of size `cached_T + 1` and copies all previous data. Over N decode steps, this results in O(N²) total copies and O(N²) total allocations.

**Fix**: Pre-allocate a fixed-size cache buffer of shape `[B, H, maxSeqLen, D]` in `KVCache.LayerCache` and use a write cursor to append new K/V entries without copying:
```java
public class LayerCache {
    public float[] keyData;   // pre-allocated [B * H * maxSeqLen * D]
    public float[] valueData;
    public int cachedLen = 0; // current number of cached positions
    
    public void append(Tensor newK, Tensor newV, int newT) {
        // System.arraycopy into keyData at offset cachedLen * innerSize
        cachedLen += newT;
    }
    
    public Tensor getKeys(int B, int H, int D) {
        return new Tensor(keyData, new int[]{B, H, cachedLen, D});
    }
}
```
This changes O(N²) to O(N) and eliminates GC pressure during generation.

---

## 12. BenchmarkRunner.java — Missing Import

### 🔵 12.1 — Uses `HashMap` without explicit import

**File**: [BenchmarkRunner.java](file:///home/umar/Documents/tinyModelZ/src/main/java/com/tinymodelz/benchmark/BenchmarkRunner.java) (printPhase8)

**Problem**: `Map<String, BenchmarkRunResult> cpuBaselines = new HashMap<>()` uses `HashMap` and `Map` which may rely on wildcard imports. Verify `java.util.HashMap` and `java.util.Map` are imported.

**Fix**: Add explicit imports if missing:
```java
import java.util.HashMap;
import java.util.Map;
```

---

## 13. FeedForward.java — Hardcoded Output Dimension

### 🔵 13.1 — Output reshape uses input `C` instead of actual output dimension

**File**: [FeedForward.java:L66](file:///home/umar/Documents/tinyModelZ/src/main/java/com/tinymodelz/nn/FeedForward.java#L66)

**Problem**: `result2d.reshape(B, T, C)` — `C` comes from `xShape[2]` which is the input embedding dim. The `cProj` linear layer projects from `hiddenDim` back to `embedDim`. Since `embedDim == C` by construction, this is currently correct, but it's fragile — if someone constructs a `FeedForward` where input dim ≠ output dim, the reshape would silently produce wrong shapes.

**Fix**: Store `embedDim` as a field and use it in the reshape, or extract the output dim from `cProj`.

---

## Priority-Ordered Fix Plan

| Priority | Item | Impact |
|----------|------|--------|
| 1 | 🔴 9.1 — AdamW weight decay ordering | Training convergence correctness |
| 2 | 🔴 1.6 — `backward()` grad init overflow | Gradient correctness for views |
| 3 | 🔴 2.1 — Causal mask shape with KV cache | Multi-token cached inference correctness |
| 4 | 🔴 3.1 — KV cache context overflow | Crash prevention during long generation |
| 5 | 🟠 11.1 — O(N²) KV cache copying | Inference speed (biggest single bottleneck) |
| 6 | 🟠 1.2 — Cache `isContiguous` | ~5-10% overall speedup |
| 7 | 🟠 1.4 — Parallel threshold | Small tensor overhead elimination |
| 8 | 🟠 5.1 — Embedding direct array access | Training throughput |
| 9 | 🟠 6.1 — Remove spurious synchronized | Backward pass speedup |
| 10 | 🟠 4.1 — DataLoader rebuild avoidance | Epoch reset speedup |
| 11 | 🟡 1.1 — `cat()` toContiguous hoisting | KV cache inference speedup |
| 12 | 🟡 1.5 — Scalar multiply fast path | Attention scaling speedup |
| 13 | 🟡 10.1 — Double exp in CrossEntropy | Minor training speedup |
| 14 | 🟡 3.2 — Repetition penalty dedup | Code quality |
| 15 | 🔵 1.3 — `getShape()` clone avoidance | Micro-optimization |
| 16 | 🔵 7.1 — RoPE integration | Architecture enhancement |
| 17 | 🔵 1.7 — `cat()` autograd | Future training extensibility |
