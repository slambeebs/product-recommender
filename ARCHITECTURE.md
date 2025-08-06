# 🧠 System Architecture Overview

This document provides a technical overview of the **Product Search System** — a lightweight multimodal search application combining text, image, and OCR-based product matching.

---

## 🔧 Components

### 1. `app.py` (Flask Backend)
Handles 3 main routes:
- `/product-recommendation`: Accepts user text query and returns top similar products.
- `/ocr-query`: Accepts uploaded image, extracts text using `ocr_query.py`, and returns top product matches.
- `/image-product-search`: Accepts product image, classifies it using a FAISS-based image similarity search, and returns top similar products.

### 2. `ocr_query.py` (OCR Pipeline)
- Uses **OCR.space API** to extract text from images.
- Text is cleaned and passed to the embedding pipeline for product matching.

### 3. Vector Embedding Pipeline
- **Text Embeddings**: Product descriptions are embedded using `sentence-transformers` (`all-MiniLM-L6-v2`).
- **Image Embeddings**: Product images are embedded using a pre-trained CNN model (`ResNet50`) for feature extraction.
- Embeddings are indexed using **FAISS** for fast nearest neighbor search.
- Output index is stored in `faiss_index.pkl`.

---

## 🔄 Data Flow

### A. Text Query Flow
```plaintext
User Text Query
    ↓
SentenceTransformer Embedding
    ↓
FAISS Search (text index)
    ↓
Top Matches → JSON Response
```

### B. OCR Image Flow
```plaintext
User Uploads Image
    ↓
OCR.space API (via ocr_query.py)
    ↓
Extracted Text
    ↓
SentenceTransformer → FAISS → Top Matches
```

### C. Image Search Flow
```plaintext
User Uploads Product Image
    ↓
Preprocessing → ResNet50 Feature Extraction
    ↓
FAISS Search (image index)
    ↓
Top Matches → JSON Response
```

---

## 🧪 Model Details

### Image Embedding Model
- Architecture: Pre-trained **ResNet50** (output: 2048-dim feature vector)
- Input: RGB image (resized to 224x224)
- Output: Image embeddings passed to FAISS

### Text Embedding Model
- Model: `all-MiniLM-L6-v2` from `sentence-transformers`
- Output: 384-dim vector per product description
- Similarity: Cosine distance via FAISS

---

## 🧼 Error Handling

- Empty queries and invalid file types are checked.
- OCR errors are caught and returned as JSON with appropriate status.
- Embedding and FAISS search exceptions are handled gracefully.

---

## 📈 Potential Improvements

- Add support for multilingual OCR
- Incorporate product metadata in ranking
- Introduce caching for faster repeat queries
- Switch to GPU-accelerated FAISS for large datasets
- Add Swagger/OpenAPI docs for all routes

---

---

## 🗂️ Folder Breakdown

```
📦 project-root
├── run.py
├── requirements.txt
├── ARCHITECTURE.md
├── README.md
├── structure.txt
├── dataset/
│   ├── clean_dataset.csv
│   └── CNN_Model_Train_Data.csv
├── cnn_data/
│   └── <multiple folders with product images>
├── templates/
│   ├── image_search.html
│   ├── ocr_query.html
│   ├── results.html
│   ├── sample_response.html
│   └── text_query.html
└── app/
    ├── __init__.py
    ├── config/
    │   └── config.py
    ├── models/
    │   ├── cnn_model.h5
    │   └── vectorized_dataset.pkl
    ├── routes/
    │   ├── image_search.py
    │   ├── ocr_query.py
    │   ├── pages.py
    │   └── text_search.py
    ├── scripts/
    │   └── train_cnn.py
    ├── services/
    │   ├── __init__py
    │   ├── old_cnn_service.py
    │   ├── vector_service.py
    │   └── cnn/
    │       ├── data_loader.py
    │       ├── model_builder.py
    │       └── train.py
    └── utils/
        ├── clean_images.py
        ├── image_scraper.py
        ├── model_tester.py
        └── ocr_tester.py
```

---

## 👥 Designed For

- Developers building **multimodal product search systems**
- Educational use in **computer vision + NLP projects**
- Lightweight experimentation for **small e-commerce apps**