from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pytesseract
from PIL import Image
import io
import requests
from tf_keras.models import load_model
from tf_keras.preprocessing import image
import os


app = Flask(__name__)

# Load the vectorized product data
df = pd.read_pickle("vectorized_dataset.pkl")

# Convert list of embeddings to a matrix
embeddings_matrix = np.vstack(df['embedding'].values)

# Load the same model used for encoding
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load your trained CNN model
cnn_model = load_model("cnn_model.h5")

# Mapping class indices to StockCodes (same order used during training)
class_labels = ['20726', '21034', '21931', '22077', '22112', '22139', '22384', '22423', '22727', '23298']

@app.route('/product-recommendation', methods=['POST'])
def product_recommendation():
    query = request.form.get('query', '')

    if not query.strip():
        return jsonify({"error": "Query is empty"}), 400

    # Generate embedding for the input query
    query_vec = model.encode([query])

    # Calculate cosine similarities
    similarities = cosine_similarity(query_vec, embeddings_matrix)[0]

    # Get top 5 most similar product indices
    top_indices = similarities.argsort()[-5:][::-1]

    # Fetch matching products
    matched_products = df.iloc[top_indices][["StockCode", "Description", "UnitPrice", "Country"]]

    # Convert to list of dicts
    product_list = matched_products.to_dict(orient="records")

    # Generate a natural language summary
    descriptions = matched_products["Description"].tolist()
    response = "Here are some product suggestions based on your query: " + "; ".join(descriptions)

    return jsonify({"products": product_list, "response": response})


@app.route('/ocr-query', methods=['POST'])
def ocr_query():
    image_file = request.files.get('image_data')

    if not image_file:
        return jsonify({"error": "No image uploaded"}), 400

    # Make sure the file stream is at the beginning
    image_file.stream.seek(0)

    # Send image to OCR.space API
    ocr_api_key = "helloworld"  # Replace with your real API key if needed
    ocr_url = "https://api.ocr.space/parse/image"

    response = requests.post(
        ocr_url,
        files={"file": (image_file.filename, image_file.stream, image_file.content_type)},
        data={
            "apikey": ocr_api_key,
            "language": "eng",
            "OCREngine": "2",
            "isOverlayRequired": "false"
        }
    )

    result = response.json()

    if result.get("IsErroredOnProcessing"):
        return jsonify({"error": "OCR failed", "details": result.get("ErrorMessage", "")}), 500

    parsed_results = result.get("ParsedResults", [])
    if not parsed_results:
        return jsonify({"error": "No readable text found"}), 400

    extracted_text = parsed_results[0].get("ParsedText", "").strip()
    if not extracted_text:
        return jsonify({"error": "No readable text found"}), 400

    # Reuse recommendation logic
    query_vec = model.encode([extracted_text])
    similarities = cosine_similarity(query_vec, embeddings_matrix)[0]
    top_indices = similarities.argsort()[-5:][::-1]
    matched_products = df.iloc[top_indices][["StockCode", "Description", "UnitPrice", "Country"]]
    product_list = matched_products.to_dict(orient="records")
    response_text = f"Extracted: '{extracted_text}'. Based on that, here are some product suggestions: " + "; ".join(matched_products["Description"])

    return jsonify({
        "products": product_list,
        "response": response_text,
        "extracted_text": extracted_text
    })



@app.route('/image-product-search', methods=['POST'])
def image_product_search():
    product_image = request.files.get('product_image')

    if not product_image:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        # Load image and preprocess it
        img = Image.open(product_image).convert('RGB')
        img = img.resize((128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict using CNN
        prediction = cnn_model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_stockcode = class_labels[predicted_index]

        # 🔍 Find exact match from your product DB
        matched_products = df[df["StockCode"] == predicted_stockcode]

        if matched_products.empty:
            return jsonify({
                "error": f"No matching product found for StockCode {predicted_stockcode}"
            }), 404

        product_list = matched_products[["StockCode", "Description", "UnitPrice", "Country"]].to_dict(orient="records")

        response_text = response_text = f"Detected product with StockCode '{predicted_stockcode}'."

        return jsonify({
            "products": product_list,
            "response": response_text,
            "predicted_class": predicted_stockcode
        })

    except Exception as e:
        return jsonify({"error": "Image processing failed", "details": str(e)}), 500

@app.route('/sample_response', methods=['GET'])
def sample_response():
    """
    Endpoint to return a sample JSON response for the API.
    Output: JSON with 'products' (array of objects) and 'response' (string).
    """
    return render_template('sample_response.html')

# Show text query page
@app.route('/text-query', methods=['GET', 'POST'])
def handle_text_query():
    if request.method == 'POST':
        query = request.form.get('query', '')
        # Call internal API
        response = app.test_client().post('/product-recommendation', data={'query': query})
        result = response.get_json()
        return render_template('results.html', 
                               response=result['response'],
                               products=result['products'],
                               predicted_class=None)
    return render_template('text_query.html')


# Show OCR image query page
@app.route('/ocr-query-form', methods=['GET', 'POST'])
def handle_ocr_query_form():
    if request.method == 'POST':
        file = request.files['image_data']
        response = app.test_client().post(
            '/ocr-query', 
            data={'image_data': file}, 
            content_type='multipart/form-data'
        )
        result = response.get_json()

        if 'error' in result:
            return render_template('results.html',
                                   response=result['error'],
                                   products=[],
                                   predicted_class=None)

        return render_template('results.html',
                               response=result['response'],
                               products=result['products'],
                               predicted_class=None)
    return render_template('ocr_query.html')



# Show CNN image query page
@app.route('/image-search-form', methods=['GET', 'POST'])
def handle_image_query_form():
    if request.method == 'POST':
        file = request.files['product_image']
        response = app.test_client().post('/image-product-search', data={'product_image': file}, content_type='multipart/form-data')
        result = response.get_json()
        return render_template('results.html',
                               response=result['response'],
                               products=result['products'],
                               predicted_class=result.get('predicted_class'))
    return render_template('image_search.html')

if __name__ == '__main__':
    app.run(debug=True)
