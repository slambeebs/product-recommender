from flask import Blueprint, request, jsonify, current_app
import numpy as np

text_search_bp = Blueprint('text_search', __name__)

@text_search_bp.route('/product-recommendation', methods=['POST'])
def product_recommendation():
    query = request.form.get('query', '')
    if not query.strip():
        return jsonify({"error": "Query is empty"}), 400

    model = current_app.config['sentence_model']
    df = current_app.config['df']
    embeddings_matrix = current_app.config['embeddings_matrix']

    query_vec = model.encode([query])
    similarities = np.dot(query_vec, embeddings_matrix.T)[0]
    top_indices = similarities.argsort()[-5:][::-1]
    matched_products = df.iloc[top_indices][["StockCode", "Description", "UnitPrice", "Country"]]
    product_list = matched_products.to_dict(orient="records")
    descriptions = matched_products["Description"].tolist()
    response = "Here are some product suggestions based on your query: " + "; ".join(descriptions)

    return jsonify({"products": product_list, "response": response})