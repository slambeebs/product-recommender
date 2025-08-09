import requests

# Path to your image file
image_path = "WhatsApp Image 2025-02-20 at 01.47.24 (1).jpg" #TODO: Replace with your image path

# OCR.space API endpoint
url = "https://api.ocr.space/parse/image"

# Your API key (use the free demo key or get your own)
API_KEY = "helloworld"  # Replace with your own key for production use

# Read image file as binary
with open(image_path, 'rb') as image_file:
    response = requests.post(
        url,
        files={"file": image_file},
        data={
            "apikey": API_KEY,
            "language": "eng",             # English
            "OCREngine": "2",              # Engine 2 = better accuracy
            "isOverlayRequired": "false"
        }
    )

# Parse the response
result = response.json()

# Extract text if present
if result.get("IsErroredOnProcessing"):
    print("Error:", result.get("ErrorMessage", "Unknown error"))
else:
    parsed_results = result.get("ParsedResults", [])
    if parsed_results:
        text = parsed_results[0].get("ParsedText", "").strip()
        print("Extracted Text:")
        print(text)
    else:
        print("No text found.")