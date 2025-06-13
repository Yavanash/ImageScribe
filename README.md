# 🖼️ Image Captioner

**Image Captioner** is a deep learning project that generates natural language captions for images using a combination of a ResNet-50 encoder and an LSTM decoder. Built with PyTorch and Gradio, this project demonstrates an end-to-end pipeline for vision-to-text generation.

---

## 🧠 Architecture

- **Encoder**: Pre-trained ResNet-50 CNN extracts visual features from input images.
- **Decoder**: LSTM-based sequence generator that produces text descriptions conditioned on visual features.
- **Vocabulary**: Custom vocabulary built from training captions with support for unknown words.

---

## 🚀 Features

- Convert any image into a meaningful caption.
- Gradio interface for easy web-based interaction.
- Pretrained model checkpoint loading and inference.

---

## 📁 Directory Structure

```
.
├── checkpoints/              # Model checkpoints (ignored in git)
│   └── final_model.pth
├── vocab/
│   └── vocab.pkl             # Serialized vocabulary
├── main.py                   # Gradio app entrypoint
├── model.py                  # Encoder and decoder models
├── utils.py                  # Preprocessing and helper functions
├── README.md
└── .gitignore
```

---

## 🛠️ Installation

```bash
git clone https://github.com/Yavanash/ImageCaptioner.git
cd ImageCaptioner
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Make sure you have the required files:

- `checkpoints/final_model.pth` — Trained model weights
- `vocab/vocab.pkl` — Vocabulary file

> Note: `.pth` files are large and ignored in version control. You may need to download them separately.

---

## 📸 Running the App

```bash
python main.py
```

This launches a Gradio interface at `http://localhost:7860` or a shared URL if enabled.

---

## 🧪 Example

Upload an image through the Gradio UI and the model will return a caption like:

```
"A group of people riding horses through a field"
```

---

## ⚙️ Model Details

- Encoder: `ResNet-50` (pretrained on ImageNet)
- Decoder: `1-layer LSTM` with hidden size 512 and embedding size 256
- Sequence length capped at 25 tokens
- Vocabulary built from training corpus with `min_freq=5`

---

## 🧾 License

This project is open-sourced under the [MIT License](LICENSE).

---

## ✨ Acknowledgements

- [PyTorch](https://pytorch.org/)
- [Gradio](https://www.gradio.app/)
- [Image Captioning Research](https://cs.stanford.edu/people/karpathy/deepimagesent/)

---

## 📬 Contact

Made with ❤️ by [@Yavanash](https://github.com/Yavanash)
