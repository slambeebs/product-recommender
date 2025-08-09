# Product Recommendation System – Multimodal Search

A production-ready **multimodal product search platform** that supports:  
1. **Text Search** → Retrieve products based on semantic similarity.  
2. **Image Search (OCR)** → Extract text from an image and match products.  
3. **Image Classification (CNN)** → Predict product class from an image and find similar products.

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

## 🛠 Services

| Service          | Technology Used                           | Purpose |
|------------------|-------------------------------------------|---------|
| **Text Search**  | SentenceTransformer (MiniLM)              | Finds semantically similar products |
| **OCR Service**  | OCR.space API                             | Extracts text from images |
| **CNN Service**  | TensorFlow/Keras                          | Classifies products by image |
| **Vector Search**| NumPy cosine similarity                   | Finds nearest products in embedding space |

---

## 🌐 API Reference

| Endpoint                   | Method | Parameters / Body                              | Response |
|----------------------------|--------|-----------------------------------------------|----------|
| `/product-recommendation`  | POST   | `query` (str)                                 | JSON with `response`, `products` |
| `/ocr-query`               | POST   | `image_data` (file)                           | JSON with `response`, `products` |
| `/image-product-search`    | POST   | `product_image` (file)                        | JSON with `response`, `products`, `predicted_class` |
| `/text-query`              | GET/POST | HTML form submission                        | HTML page |
| `/ocr-query-form`          | GET/POST | HTML form submission                        | HTML page |
| `/image-search-form`       | GET/POST | HTML form submission                        | HTML page |

---

## ⚙ Workflow

1. **Text Query**  
   - User inputs product description → Encoded via SentenceTransformer → Compared with stored embeddings → Top matches returned.

2. **OCR Query**  
   - User uploads image → OCR.space extracts text → Same as Text Query workflow.

3. **Image Classification Query**  
   - User uploads product image → CNN model predicts class → Products of same class retrieved.

---

## 📂 Project Structure

```
project/
├── app/                    # Main Flask application package
│   ├── routes/             # API and UI endpoints (Flask Blueprints)
│   ├── templates/          # HTML templates for web interface
│   ├── __init__.py         # App factory, model loading, blueprint registration
├── cnn-data/               # Raw and processed images for CNN model training
├── config/                 # Configuration files (API keys, environment settings)
├── dataset/                # CSVs and datasets for embeddings & CNN
├── models/                 # Trained models (.h5 for CNN, .pkl for embeddings)
├── notebooks/              # Experiment notebooks:
│   ├── 01_data_exploration.ipynb – Dataset inspection & sample visualizations
│   ├── 02_cnn_training_experiments.ipynb – CNN model training & accuracy plots
│   └── 03_model_evaluation.ipynb – Model testing & confusion matrix
├── scripts/                # Scripts for training and preprocessing
├── services/               # Core service logic (OCR, CNN inference, vector search)
├── utils/                  # Helper scripts (data cleaning, testing, scraping)
├── run.py                  # Application entry point
├── requirements.txt        # Python dependencies
├── ARCHITECTURE.md         # System design and architecture documentation
├── README.md               # Project documentation
└── structure.txt           # Project file tree
```

---

## 💻 Setup & Run

```bash
git clone <repo-url>
cd product-recommendation
pip install -r requirements.txt
python run.py
```
---

## 📦 Dependencies

- Flask  
- TensorFlow / tf.keras  
- SentenceTransformers  
- Pandas, NumPy, Pillow  
- Requests (OCR.space API)  

---

## 🧠 Model Training

Run:
```bash
python training/train_cnn.py
```

---

## 👤 Author
Abdul Salam