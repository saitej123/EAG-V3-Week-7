# Attention Is All You Need (Vaswani et al., 2017)

**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin  
**Venue:** NeurIPS 2017  
**Task focus:** Machine translation (WMT English–German and English–French), with architecture ideas that later dominated NLP and beyond.

## Abstract and motivation

Sequence modeling historically relied on recurrent neural networks (RNNs) and gated variants such as LSTMs and GRUs. Recurrence forces sequential computation: hidden state at time *t* depends on step *t−1*, limiting parallelization on modern hardware and lengthening the path between distant tokens. Convolutional sequence models offered some parallelism but still required stacking many layers or dilated convolutions to connect far-apart positions.

The Transformer proposes **dispensing with recurrence and convolutions entirely** for the core sequence transduction stack, replacing them with **multi-head self-attention** and position-wise feed-forward networks. The model achieves strong translation quality while training faster because every layer can attend across the full sequence in parallel.

## Overall architecture

The model follows an **encoder–decoder** layout familiar from prior seq2seq work:

- **Encoder:** stacks of identical layers; each layer has multi-head self-attention followed by a position-wise feed-forward network (FFN), with residual connections and layer normalization.
- **Decoder:** stacks of identical layers; each layer has masked multi-head self-attention (causal mask so position *i* cannot attend to *j > i*), encoder–decoder attention (queries from decoder, keys/values from encoder output), then the FFN block.

Input tokens are embedded into dimension *d_model* (512 in the base model). Dropout and label smoothing are used during training. For translation, byte-pair encoding builds subword vocabularies.

## Contribution 1: Scaled dot-product self-attention

The central mechanism maps a set of queries, keys, and values (each row is a token position). For packed matrices **Q**, **K**, **V**:

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
\]

**Intuition:** each query vector scores compatibility with every key; softmax yields attention weights; values are weighted sums. The **scaling factor √d_k** prevents dot products from growing large in high dimensions, which would push softmax into regions with tiny gradients.

**Self-attention** means Q, K, and V all come from the same sequence (possibly after linear projections). Every position can directly attend to every other position in one layer—constant path length between any two tokens, unlike RNNs where distance grows with sequence length.

Compared to additive attention, dot-product attention is faster on modern accelerators when dimension is moderate, thanks to optimized matrix multiply kernels.

## Multi-head attention

Instead of one attention function, the model runs **h = 8 parallel heads** (in the base config). Each head projects Q, K, V to smaller dimensions (*d_k = d_v = 64* when *d_model = 512*), applies attention independently, concatenates outputs, and applies a final linear map.

**Why multiple heads:** different heads can learn different relational patterns—syntax, long-range coreference, local phrase structure—without forcing a single attention distribution to capture everything. This is still part of the self-attention contribution but is the standard way the paper operationalizes it.

Encoder–decoder attention uses queries from the decoder and keys/values from encoder memory, letting each generated token focus on relevant source positions.

## Contribution 2: Parallel computation across positions

Because there is **no recurrence**, all positions in a layer can be processed **simultaneously** (subject only to masking in the decoder). Attention and FFN blocks apply to the **entire sequence at once** as batched matrix operations.

**Training implications:** wall-clock per step scales better with sequence length on GPUs/TPUs than step-by-step RNN unrolling. The paper reports significantly reduced training time versus strong RNN and ConvS2S baselines at comparable quality.

**Inference note:** autoregressive decoding still generates one token at a time in the decoder, but encoder self-attention over the source runs in parallel once; decoder self-attention runs over the partial target prefix with caching optimizations in production systems.

**Path length:** any two positions interact in O(1) sequential operations depth-wise (through stacked layers), whereas RNN depth grows linearly with distance.

## Position-wise feed-forward networks

Each layer includes a two-layer MLP applied identically at every position:

\[
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
\]

Inner dimension is 2048 in the base model. This adds non-linear capacity per position after mixing via attention.

## Contribution 3: Positional encoding

Self-attention is **permutation-invariant** if you feed raw embeddings alone—the model cannot distinguish token order. The Transformer injects **positional information** without reintroducing recurrence.

The original paper uses **fixed sinusoidal encodings** added to input embeddings:

\[
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}}), \quad
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})
\]

**Properties:** each dimension oscillates at a different wavelength; the model can learn to attend by relative offsets because linear combinations of sinusoids at *pos+k* relate to those at *pos*. Later work also uses learned positional embeddings; the paper notes similar results.

Positional encodings use the same dimension as token embeddings and are **summed** before the first layer. This preserves **parallel forward passes** over the whole sequence while making order explicit.

## Regularization and training details

- Residual dropout and attention dropout (rate 0.1 base).
- Label smoothing (ε = 0.1) during training.
- Adam optimizer with warmup learning-rate schedule (4000 warmup steps).
- Batch size ~25k target tokens; training on 8 P100 GPUs.

## Results (selected)

- **WMT 2014 En–De:** 28.4 BLEU (base), 41.8 BLEU (big), improving over prior state of the art.
- **WMT 2014 En–Fr:** 41.0 BLEU (big model).
- Training cost fraction reported lower than RNN-based GNMT-style systems at similar quality.

## Three key contributions (summary for retrieval)

1. **Scaled dot-product self-attention** as the primary sequence mixing operator, replacing recurrence; includes multi-head variants and encoder–decoder attention for transduction.
2. **Parallel computation** over all token positions within each layer, enabling efficient hardware utilization and short gradient paths between distant tokens.
3. **Positional encoding** (sinusoidal in the paper) injected into embeddings so the model knows token order while retaining fully parallelizable attention blocks.

## Limitations noted in follow-on work

- Quadratic memory/time in sequence length for full attention (motivates sparse, linear, and local attention variants).
- Positional extrapolation beyond training lengths can degrade without modifications (relative positions, RoPE, ALiBi, etc.).
- Pure encoder stacks (BERT) and decoder-only stacks (GPT) adapt the same ideas to masked language modeling and autoregressive LM.

## Historical impact

The Transformer architecture became the backbone of BERT, GPT, T5, Vision Transformer (ViT), and most large language models. The three contributions above—self-attention, parallelism, and explicit position information—are the minimal conceptual package evaluators cite when asking what the original paper introduced.

## Appendix-style notes for indexing

**Layer normalization placement:** the paper uses post-norm residuals (normalize after sublayer add); later Pre-LN variants train deeper stacks more reliably.

**Attention complexity:** self-attention over length *n* costs O(n²) in memory for full pairwise weights; production systems use flash attention kernels, sliding windows, or sparse patterns for long documents.

**Decoder masking:** the upper triangular mask ensures autoregressive causality; bi-directional encoder representations power retrieval encoders (BERT) by removing masking in encoder-only stacks derived from the same blueprint.

**Translation-specific heads:** English–German and English–French experiments use shared subword vocabularies (~37k tokens) and beam search decoding; these operational details do not change the three architectural contributions but explain training logs in reproduction efforts.

When students summarize the paper for exams, the acceptable short list remains: (1) scaled dot-product self-attention / multi-head attention as the mixing primitive, (2) fully parallelizable sequence layers without recurrence, and (3) sinusoidal (or learned) positional encodings so order is explicit while keeping parallel attention.
