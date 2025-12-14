"""
Test script for the deployed Cloud Run API
"""

import requests
import json


SERVICE_URL = "https://your-service-url.run.app"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{SERVICE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_single_prediction():
    """Test single prediction"""
    print("Testing /predict endpoint...")
    
    data = {
        "premise": "A person is walking a dog in the park",
        "hypothesis": "A person is outside"
    }
    
    response = requests.post(
        f"{SERVICE_URL}/predict",
        json=data
    )
    
    print(f"Status: {response.status_code}")
    print(f"Request: {json.dumps(data, indent=2)}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_batch_prediction():
    """Test batch prediction"""
    print("Testing /batch_predict endpoint...")
    
    data = {
        "pairs": [
            {
                "premise": "A person is walking a dog",
                "hypothesis": "A person is outside"
            },
            {
                "premise": "The cat is sleeping on the couch",
                "hypothesis": "The cat is awake"
            },
            {
                "premise": "It's raining heavily outside",
                "hypothesis": "The weather is sunny"
            }
        ]
    }
    
    response = requests.post(
        f"{SERVICE_URL}/batch_predict",
        json=data
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_edge_cases():
    """Test edge cases"""
    print("Testing edge cases...")
    
    # Very long text
    long_premise = "This is a very long premise. " * 100
    data = {
        "premise": long_premise,
        "hypothesis": "This is short"
    }
    
    response = requests.post(f"{SERVICE_URL}/predict", json=data)
    print(f"Long text - Status: {response.status_code}")
    
    # Empty text
    data = {
        "premise": "",
        "hypothesis": ""
    }
    
    response = requests.post(f"{SERVICE_URL}/predict", json=data)
    print(f"Empty text - Status: {response.status_code}\n")

if __name__ == "__main__":
    print("=" * 60)
    print("Cloud Run API Test Suite")
    print("=" * 60 + "\n")
    
    try:
        test_health()
        test_single_prediction()
        test_batch_prediction()
        test_edge_cases()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to service.")
        print("Make sure SERVICE_URL is set correctly and the service is running.")
    except Exception as e:
        print(f"ERROR: {e}")
