# Product Recommendation System

A multimodal product search system that supports three types of queries:

1. **Text-based Query:** Enter a product description to get relevant suggestions.
2. **Image-based Query (OCR):** Upload an image with product text; the system extracts text and provides suggestions.
3. **Image Classification (CNN):** Upload a product image; the system predicts the product and finds similar items.

---

## 🚀 Features

- Semantic search with `SentenceTransformer` (MiniLM)
- OCR via [OCR.space API](https://ocr.space/)
- CNN model built with TensorFlow for image classification
- Flask-powered API + Web interface
- Pandas-powered product DB with vector embeddings

---

## 🧱 Project Structure

.
├── app/
│   ├── __init__.py               # Initializes the Flask app
│   ├── config/                   # App configuration
│   │   └── config.py
│   ├── models/                   # Pretrained models & embeddings
│   │   ├── cnn_model.h5
│   │   └── vectorized_dataset.pkl
│   ├── routes/                   # API endpoints
│   │   ├── image_search.py
│   │   ├── ocr_query.py
│   │   ├── pages.py
│   │   └── text_search.py
│   ├── scripts/                  # Training scripts
│   │   └── train_cnn.py
│   ├── services/                 # Core logic and ML services
│   │   ├── __init__.py
│   │   ├── cnn/
│   │   │   ├── data_loader.py
│   │   │   ├── model_builder.py
│   │   │   └── train.py
│   │   ├── old_cnn_service.py
│   │   └── vector_service.py
│   └── utils/                    # Utility tools and testing scripts
│       ├── clean_images.py
│       ├── image_scraper.py
│       ├── model_tester.py
│       └── ocr_tester.py
├── cnn_data/                     # Product images for training/testing
├── dataset/
│   ├── clean_dataset.csv
│   └── CNN_Model_Train_Data.csv
├── templates/                    # Web UI templates
│   ├── image_search.html
│   ├── ocr_query.html
│   ├── results.html
│   ├── sample_response.html
│   └── text_query.html
├── ARCHITECTURE.md               # System design and architecture
├── run.py                        # App entry point
├── requirements.txt              # Python dependencies
├── structure.txt                 # Project file structure
└── README.md                     # You are here

---

## 🛠️ Setup

1. Clone the repo
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download / prepare data:
   - `cnn_data/`: Directory of labeled product images (for CNN)
   - `vectorized_dataset.pkl`: Embedding-based product dataset

4. Run the app:

```bash
python app.py
```

---

## 🔍 Example Usage

**Text Query:**  
`"white porcelain mug"` → returns 5 most similar products.

**OCR Image Query:**  
Upload a label/product image → Text is extracted → Matches are returned.

**Product Image Query:**  
Upload an image of the product → CNN predicts class → Relevant product(s) returned.

---

## 📦 Dependencies

- Flask
- TensorFlow / Keras
- SentenceTransformers
- scikit-learn
- numpy, pandas, Pillow
- OCR.space API

---

## 🧠 Model Training

Run `cnn_train.py` to retrain the image classification model.

---

## 👤 Author

Developed by Abdul Salam