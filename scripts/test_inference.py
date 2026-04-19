import requests
import json
import argparse
import os

def test_inference(image_path, api_url="http://0.0.0.0:8000/api/v1/predict"):
    """
    Tests the prediction API with a local file.
    """
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    # test with file upload
    print(f"Testing API at {api_url} with file {image_path}...")
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        response = requests.post(api_url, files=files)

    if response.status_code == 200:
        result = response.json()
        print("Success!")
        print(f"Confidence: {result['confidence']}")
        print(f"Detected Polygons: {len(result['area_geojson']['features'])}")
        
        # Save output to a file
        output_file = "test_prediction_output.json"
        with open(output_file, "w") as out:
            json.dump(result, out, indent=2)
        print(f"Full response saved to {output_file}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image file to test")
    parser.add_argument("--url", default="http://0.0.0.0:8000/api/v1/predict")
    args = parser.parse_args()
    
    test_inference(args.image, args.url)
