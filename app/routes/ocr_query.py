from flask import Blueprint, request, jsonify, current_app
import requests
import numpy as np

ocr_query_bp = Blueprint('ocr_query', __name__)

@ocr_query_bp.route('/ocr-query', methods=['POST'])
def ocr_query():
    image_file = request.files.get('image_data')
    if not image_file:
        return jsonify({"error": "No image uploaded"}), 400

    image_file.stream.seek(0)
    ocr_api_key = "helloworld"
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

    model = current_app.config['sentence_model']
    df = current_app.config['df']
    embeddings_matrix = current_app.config['embeddings_matrix']

    query_vec = model.encode([extracted_text])
    similarities = np.dot(query_vec, embeddings_matrix.T)[0]
    top_indices = similarities.argsort()[-5:][::-1]
    matched_products = df.iloc[top_indices][["StockCode", "Description", "UnitPrice", "Country"]]
    product_list = matched_products.to_dict(orient="records")
    response_text = f"Extracted: '{extracted_text}'. Based on that, here are some product suggestions: " + "; ".join(matched_products["Description"])

    return jsonify({
        "products": product_list,
        "response": response_text,
        "extracted_text": extracted_text
    })