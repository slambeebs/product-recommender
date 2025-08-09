# 🧠 System Architecture Overview

This document provides a technical overview of the **Product Recommendation System – Multimodal Search** — a production-ready platform combining text, image, and OCR-based product matching.

---

## 📐 System Architecture

```
User Request → Flask API → Service Layer → Model Inference → Product Recommendation → Response
```

**Workflow Diagram:**  
1. **Text Search** → SentenceTransformer → Vector Matching  
2. **OCR Search** → OCR.space API → SentenceTransformer → Vector Matching  
3. **Image Classification** → TensorFlow CNN → Class Prediction → Filter Products  

```
                         ┌────────────────────┐
                         │   User Request     │
                         └─────────┬──────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │                         │                         │
         ▼                         ▼                         ▼
 ┌─────────────────┐     ┌─────────────────┐       ┌──────────────────────┐
 │ Text Search     │     │ OCR Query       │       │ Image Classification │
 │ (Sentence       │     │ (OCR.space API) │       │ (TensorFlow CNN)     │
 │ Transformer)    │     └───────┬─────────┘       └─────────┬────────────┘
 └───────┬─────────┘             │                           │
         │                       │                           │
         │                       ▼                           ▼
         │             ┌──────────────────────┐    ┌──────────────────────┐
         │             │ Extracted Text       │    │ CNN Predicted Class  │
         │             └─────────┬────────────┘    └──────────┬───────────┘
         │                       │                            │
         ▼                       ▼                            ▼
 ┌─────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
 │Vector Search (SBERT)│  │ Vector Search (SBERT)│  │ Vector Search (SBERT)│
 │ + Embeddings DB     │  │ + Embeddings DB      │  │ + Embeddings DB      │
 └────────┬────────────┘  └─────────┬────────────┘  └─────────┬────────────┘
          │                         │                         │
          └─────────────────────────┴─────────────────────────┘
                                    │
                                    ▼
                        ┌──────────────────────┐
                        │   Return Products    │
                        │  to User (Web/API)   │
                        └──────────────────────┘
```

---

## 🔧 Components

**Flask app & UI**
- `run.py` — Application entry point (creates app/app factory).
- `app/__init__.py` — App factory, model loading, blueprint registration.
- `app/routes/`:
  - `text_search.py` — `/product-recommendation` endpoint + text UI (`/text-query`).
  - `ocr_query.py` — `/ocr-query` endpoint (uses OCR.space) + OCR form (`/ocr-query-form`).
  - `image_search.py` — `/image-product-search` endpoint + image upload UI (`/image-search-form`).
  - `pages.py` — UI routing rendering templates and bridging form → internal API calls.
- `app/templates/` — Jinja HTML templates used by routes:
  - `text_query.html`, `ocr_query.html`, `image_search.html`, `results.html`, `sample_response.html`.

**Models (pretrained / saved outputs)**
- `models/cnn_model.h5` — Trained CNN classifier (TensorFlow / Keras).
- `models/vectorized_dataset.pkl` — Pandas dataframe containing product metadata + embeddings.

**Services (business logic / ML pipelines)**
- `services/cnn/`:
  - `data_loader.py` — Dataset validation & loader helpers (`validate_images_with_tf`, `load_datasets`).
  - `model_builder.py` — CNN architecture builder (`build_model`).
  - `train.py` — Training pipeline entrypoint (`train_model()`).
- `services/vector_service.py` — Vector search helpers (load embeddings, similarity search using NumPy/cosine).
- `utils/` — helper utilities:
  - `clean_images.py`, `image_scraper.py`, `model_tester.py`, `ocr_tester.py`

**Data & training assets**
- `cnn_data/` — Labeled product images used for training; subfolders named by StockCode (e.g. `22384`, `22112`, ...).
- `dataset/` — CSVs: `clean_dataset.csv`, `CNN_Model_Train_Data.csv`.

**Config**
- `config/config.py` — central config (paths, image size, batch size, OCR API key placeholder, model path, epochs).

---

## 🔄 Data Flow (detailed)

### 1) Text Query
1. Client POSTs `query` → `app/routes/text_search.py`
2. `sentence_transformers` (`all-MiniLM-L6-v2`) encodes query → 384-dim embedding.
3. `services/vector_service` computes cosine similarity against vectors in `models/vectorized_dataset.pkl`.
4. Return top-N products (JSON or `results.html`).

### 2) OCR Query
1. Client uploads image → `app/routes/ocr_query.py`.
2. Server forwards file to **OCR.space** API (use API key in `config/config.py`).
3. Extracted text → cleaned → follow Text Query flow.

> **Note:** OCR.space is used (not Tesseract) — set `OCR_API_KEY` in `config/config.py` before production.

### 3) Image Classification Query
1. Client uploads product image → `app/routes/image_search.py`.
2. Image preprocessed (RGB, resized to `IMG_SIZE = (128,128)`), normalized.
3. `models/cnn_model.h5` (loaded in `app/__init__.py`) performs softmax prediction → predicted class index.
4. Lookup StockCode (class → StockCode mapping from training `class_names`) → fetch exact product rows from `models/vectorized_dataset.pkl` or `dataset/clean_dataset.csv`.
5. Return matched product(s) + `predicted_class`.

---

## 🧪 Model Details

### Text Embeddings
- **Model:** `all-MiniLM-L6-v2` (sentence-transformers)
- **Dim:** 384  
- **Search:** Cosine similarity computed with NumPy (no external vector DB used).

### CNN (Image Classification)
- **Framework:** TensorFlow/Keras (you used `tf_keras` alias in code; either `tf.keras` or `tf_keras` wrappers)
- **Input:** RGB images resized to **128×128** (consistent between training & inference)
- **Architecture:** small custom CNN (see `services/cnn/model_builder.py`)
- **Output:** Softmax over N classes (class order = `train_ds.class_names` at training time)
- **Checkpoint:** `models/cnn_model.h5`

---

## ⚙️ Key Implementation Notes / Requirements

- **Class label mapping must be stable**: when training, save `class_names` (e.g., JSON) and load the same mapping at inference to avoid label mismatch.
  - Save `train_ds.class_names` after training (e.g. `models/class_names.json`) and load in `app/__init__.py`.
- **OCR API key**: put real OCR.space key in `config/config.py` or environment variable `OCR_API_KEY`.
- **Model paths** in `app/__init__.py` load from `models/` folder (relative to repo root).
- **Templates path**: Flask will discover `app/templates` only if `create_app()` sets the correct `template_folder` or you run from repo root and `app` is a package. Your current `app` package layout is correct — ensure you call `create_app()` in `run.py` and run `python run.py` from repo root.

---

## 🔁 API Reference (examples)

- `POST /product-recommendation`  
  - Body form: `query=white ceramic mug`  
  - Response: `{ "products":[{StockCode, Description, UnitPrice, Country}], "response": "…" }`

- `POST /ocr-query`  
  - Multipart form: `image_data` (file)  
  - Response: `{ "extracted_text": "...", "products":[...], "response": "..." }`

- `POST /image-product-search`  
  - Multipart form: `product_image` (file)  
  - Response: `{ "predicted_class": "22384", "products":[...] }`

> Example cURL (text):
> ```bash
> curl -X POST -F "query=red lunch bag" http://localhost:5000/product-recommendation
> ```

---

## 🧼 Error Handling

- Invalid files and empty queries are rejected with error messages.
- OCR API failures are caught and returned as structured JSON.
- Model inference errors return descriptive responses.

---

## 📈 Potential Improvements

- Support for multilingual OCR
- Caching for repeat queries
- GPU-accelerated similarity search
- Better ranking using product metadata
- OpenAPI/Swagger documentation

---

## 🗂️ Folder Breakdown

```
📦 project-root
├── run.py
├── requirements.txt
├── ARCHITECTURE.md
├── README.md
├── dataset/
│   ├── clean_dataset.csv
│   └── CNN_Model_Train_Data.csv
├── models/
│   ├── cnn_model.h5
│   └── vectorized_dataset.pkl
├── services/
│   ├── cnn_service.py
│   ├── ocr_service.py
│   ├── text_service.py
│   └── vector_service.py
├── cnn-data/
│   └── <product images>
├── scripts/
│   └── train_cnn.py
├── utils/
│   ├── clean_images.py
│   ├── image_scraper.py
│   ├── model_tester.py
│   └── ocr_tester.py
└── app/
    ├── __init__.py
    ├── routes/
    │   ├── image_search.py
    │   ├── ocr_query.py
    │   ├── pages.py
    │   └── text_search.py
    ├── templates/
    │   ├── image_search.html
    │   ├── ocr_query.html
    │   ├── results.html
    │   ├── sample_response.html
    │   └── text_query.html
```

---

## 👥 Designed For

- Developers building **multimodal product search systems**
- Educational use in **computer vision + NLP projects**
- Lightweight experimentation for **small e-commerce apps**