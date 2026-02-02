# FAISS Index Compression Benchmarks

**Test Date**: February 2026

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | 148,362 real iMessage texts |
| Embedding model | BAAI/bge-small-en-v1.5 (384 dim) |
| Hardware | Apple Silicon (M-series) |
| Queries | 100 random messages |
| Metric | Recall@10 |

## Results

| Index Type | Size | Compression | Recall@10 | Search Time |
|------------|------|-------------|-----------|-------------|
| IndexFlatIP (brute force) | 217.3 MB | 1x | 100.0% | 0.341 ms |
| IndexIVFFlat (nprobe=128) | 220.0 MB | 1x | 92.8% | 0.548 ms |
| **IVFPQ 384x8 (4x)** | **57.3 MB** | **3.8x** | **91.9%** | 1.528 ms |
| **IVFPQ 192x8 (8x)** | **30.2 MB** | **7.2x** | **88.3%** | 0.698 ms |
| IVFPQ 128x8 (12x) | 21.1 MB | 10.3x | 84.3% | 0.351 ms |
| IVFPQ 96x8 (16x) | 16.6 MB | 13.1x | 78.7% | 0.294 ms |
| IVFPQ 64x8 (24x) | 12.1 MB | 18.0x | 71.9% | 0.160 ms |

## Compression Analysis

| Index Type | Memory Saved | Recall Loss |
|------------|--------------|-------------|
| IVFPQ 384x8 (4x) | 160.0 MB (73%) | 8.1% |
| IVFPQ 192x8 (8x) | 187.1 MB (86%) | 11.7% |
| IVFPQ 128x8 (12x) | 196.2 MB (90%) | 15.7% |

## Scaled Projections (400K Messages)

| Index Type | Est. Size | Memory Saved | Expected Recall |
|------------|-----------|--------------|-----------------|
| Brute force | ~586 MB | — | 100% |
| IVFPQ 384x8 (4x) | ~155 MB | **431 MB** | ~92% |
| IVFPQ 192x8 (8x) | ~81 MB | **505 MB** | ~88% |
| IVFPQ 128x8 (12x) | ~57 MB | **529 MB** | ~84% |

## Key Findings

1. **IndexIVFFlat provides no memory savings** - Same size as brute force, only useful at millions of vectors

2. **IVFPQ 384x8 is the quality sweet spot** - 3.8x compression, 91.9% recall, saves ~430MB on 400K messages

3. **IVFPQ 192x8 is the memory sweet spot** - 7.2x compression, 88.3% recall, for 4GB systems

4. **Search time is NOT the bottleneck** - All indexes <2ms, embedding query is ~100-150ms

5. **Training is one-time** - ~10-20s initially, index persists to disk

## Recommendations

| Situation | Index | Memory (400K) | Recall |
|-----------|-------|---------------|--------|
| **Default** | **IVFPQ 384x8 (4x)** | **~155 MB** | **~92%** |
| Tight memory (4GB) | IVFPQ 192x8 (8x) | ~81 MB | ~88% |
| Quality critical | IndexFlatIP | ~586 MB | 100% |

**Why 4x is default:** 4x vs 8x is only 74MB difference but gives 4% better recall.

## Benchmark Scripts

```bash
# Test on 50K messages
uv run python scripts/benchmark_faiss_real.py --limit 50000

# Full benchmark
uv run python scripts/benchmark_faiss_real.py --limit 500000
```
