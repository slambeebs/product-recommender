from flask import Flask
from .routes.text_search import text_search_bp
from .routes.ocr_query import ocr_query_bp
from .routes.image_search import image_search_bp
from .routes.pages import pages_bp
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tf_keras.models import load_model

df = pd.read_pickle("models/vectorized_dataset.pkl")
embeddings_matrix = np.vstack(df['embedding'].values)
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
cnn_model = load_model("models/cnn_model.h5")
class_labels = ['20726', '21034', '21931', '22077', '22112', '22139', '22384', '22423', '22727', '23298']

def create_app():
    app = Flask(__name__)
    
    # Add shared resources to app config
    app.config['df'] = df
    app.config['embeddings_matrix'] = embeddings_matrix
    app.config['sentence_model'] = sentence_model
    app.config['cnn_model'] = cnn_model
    app.config['class_labels'] = class_labels

    # Register blueprints
    app.register_blueprint(text_search_bp)
    app.register_blueprint(ocr_query_bp)
    app.register_blueprint(image_search_bp)
    app.register_blueprint(pages_bp)

    return app