from flask import Blueprint, request, jsonify, current_app
from PIL import Image
import numpy as np
from tf_keras.preprocessing import image as keras_image

image_search_bp = Blueprint('image_search', __name__)

@image_search_bp.route('/image-product-search', methods=['POST'])
def image_product_search():
    product_image = request.files.get('product_image')
    if not product_image:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        img = Image.open(product_image).convert('RGB')
        img = img.resize((128, 128))
        img_array = keras_image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        cnn_model = current_app.config['cnn_model']
        df = current_app.config['df']
        class_labels = current_app.config['class_labels']

        prediction = cnn_model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_stockcode = class_labels[predicted_index]

        matched_products = df[df["StockCode"] == predicted_stockcode]
        if matched_products.empty:
            return jsonify({"error": f"No matching product found for StockCode {predicted_stockcode}"}), 404

        product_list = matched_products[["StockCode", "Description", "UnitPrice", "Country"]].to_dict(orient="records")
        response_text = f"Detected product with StockCode '{predicted_stockcode}'."

        return jsonify({
            "products": product_list,
            "response": response_text,
            "predicted_class": predicted_stockcode
        })

    except Exception as e:
        return jsonify({"error": "Image processing failed", "details": str(e)}), 500