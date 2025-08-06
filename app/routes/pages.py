from flask import Blueprint, render_template, request, current_app

pages_bp = Blueprint('pages', __name__)

@pages_bp.route('/sample_response', methods=['GET'])
def sample_response():
    return render_template('sample_response.html')

@pages_bp.route('/text-query', methods=['GET', 'POST'])
def handle_text_query():
    if request.method == 'POST':
        query = request.form.get('query', '')
        response = current_app.test_client().post('/product-recommendation', data={'query': query})
        result = response.get_json()
        return render_template('results.html', 
                               response=result['response'],
                               products=result['products'],
                               predicted_class=None)
    return render_template('text_query.html')

@pages_bp.route('/ocr-query-form', methods=['GET', 'POST'])
def handle_ocr_query_form():
    if request.method == 'POST':
        file = request.files['image_data']
        response = current_app.test_client().post('/ocr-query', data={'image_data': file}, content_type='multipart/form-data')
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

@pages_bp.route('/image-search-form', methods=['GET', 'POST'])
def handle_image_query_form():
    if request.method == 'POST':
        file = request.files['product_image']
        response = current_app.test_client().post('/image-product-search', data={'product_image': file}, content_type='multipart/form-data')
        result = response.get_json()
        return render_template('results.html',
                               response=result['response'],
                               products=result['products'],
                               predicted_class=result.get('predicted_class'))
    return render_template('image_search.html')