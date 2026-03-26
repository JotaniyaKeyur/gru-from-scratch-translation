# Seq2Seq GRU from Scratch (NumPy)

This project implements a full **Sequence-to-Sequence (Seq2Seq) model using GRU** from scratch using only **NumPy**, without any deep learning frameworks.

The goal of this project is **deep understanding of the math and training dynamics** behind seq2seq models, not achieving state-of-the-art performance.

---

## 🚀 Overview

- Dataset: WMT14 Hindi-English (`hi-en`)
- Model: Encoder–Decoder GRU (1-layer)
- Implementation: Pure NumPy
- Training: Manual forward + backward (BPTT)
- Decoding: Beam Search (beam width = 2)
- Evaluation: SacreBLEU

---

## 📂 Pipeline

### 1. Dataset Loading
dataset = load_dataset("wmt/wmt14", "hi-en")

Splits:
- Train
- Validation
- Test

---

### 2. Tokenization

- Tokenizer: Keyurjotaniya007/wmt-hi-en-tokenizer
- Shared vocabulary (Hindi + English)
- Max lengths:
  - Source: 32
  - Target: 32

Special step:
- Source sequence is reversed

---

### 3. Embeddings

Trainable embedding matrix:
E ∈ (vocab_size, embedding_dim)

Shared between encoder and decoder

---

### 4. Model Architecture

Encoder:
- GRU (manual implementation)
- Processes reversed English sentence

Decoder:
- GRU (manual implementation)
- Initialized with encoder final hidden state

Output:
logits = Wo @ h + bo

---

### 5. Training

Loss:
- Cross-entropy (manual)

Optimization:
- SGD + Momentum
- L2 Regularization
- StepLR Scheduler

Techniques:
- Teacher forcing
- Gradient clipping
- Full BPTT

---

### 6. Inference

- Autoregressive decoding
- Beam Search (beam width = 2)

---

### 7. Evaluation

- Metric: SacreBLEU

---

## 📊 Training Results

Epoch 1: 25.10  
Epoch 2: 21.83  
Epoch 3: 19.42  
Epoch 4: 17.87  
Epoch 5: 15.86  
Epoch 6: 14.22  
Epoch 7: 12.17  

---

## 📈 Evaluation Result

SacreBLEU: ~2.7e-08

---

## ⚠️ Notes

- Small model capacity (hidden_dim = 100)
- No attention mechanism
- Short training duration
- Focus is on learning, not performance

---

## 🧠 Key Learnings

- Seq2Seq architecture from scratch
- GRU math (forward & backward)
- Backpropagation Through Time (BPTT)
- Exposure bias (train vs inference gap)
- Beam search decoding
- Full training pipeline without frameworks

---

## 💾 Model Files

- model.safetensors
- tokenizer.json
- tokenizer_config.json

---

## 🔮 Future Improvements

- Add Attention mechanism
- Increase model size
- Use batching
- Improve decoding strategy
- Move to PyTorch

---

## 📌 Conclusion

This project demonstrates a complete seq2seq system built from scratch using NumPy, focusing on understanding the underlying mathematics and training behavior.
