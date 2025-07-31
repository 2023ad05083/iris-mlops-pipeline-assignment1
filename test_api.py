import requests
import json
import time

# Test data
test_data = {
    "features": [
        {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
        {
            "sepal_length": 6.2,
            "sepal_width": 2.9,
            "petal_length": 4.3,
            "petal_width": 1.3,
        },
    ]
}


def test_api():
    base_url = "http://localhost:8000"

    print("🧪 Testing Containerized Iris API")
    print("=" * 50)

    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health Status: {health_data['status']}")
            print(f"   Model loaded: {health_data['model_loaded']}")
            print(f"   Version: {health_data['version']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

    # Test root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            root_data = response.json()
            print(f"✅ API Message: {root_data['message']}")
            print(f"   Available endpoints: {len(root_data['endpoints'])}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")

    # Test prediction endpoint
    print("\n3. Testing prediction endpoint...")
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        response_time = (time.time() - start_time) * 1000

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful!")
            print(f"   Response time: {response_time:.1f}ms")
            print(f"   Request ID: {result['request_id']}")

            for i, prediction in enumerate(result["predictions"]):
                print(
                    f"   Sample {i+1}: {prediction['prediction_name']} ({prediction['confidence']:.2%})"
                )

        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Prediction error: {e}")

    # Test metrics endpoint
    print("\n4. Testing metrics endpoint...")
    try:
        response = requests.get(f"{base_url}/metrics", timeout=10)
        if response.status_code == 200:
            metrics = response.json()
            print(f"✅ Metrics retrieved!")
            print(f"   Total predictions: {metrics['total_predictions']}")
            print(f"   Uptime: {metrics['uptime_seconds']:.1f} seconds")
            if metrics["model_accuracy"]:
                print(f"   Model accuracy: {metrics['model_accuracy']:.2%}")
        else:
            print(f"❌ Metrics failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Metrics error: {e}")

    # Test API documentation
    print("\n5. Testing API documentation...")
    try:
        response = requests.get(f"{base_url}/docs", timeout=10)
        if response.status_code == 200:
            print(f"✅ API documentation accessible!")
            print(f"   Visit: {base_url}/docs")
        else:
            print(f"❌ API docs failed: {response.status_code}")
    except Exception as e:
        print(f"❌ API docs error: {e}")

    print("\n" + "=" * 50)
    print("🎉 Container testing completed!")
    print(f"🌐 API running at: {base_url}")
    print(f"📚 Documentation: {base_url}/docs")
    print("=" * 50)

    return True


if __name__ == "__main__":
    # Give the container a moment to fully start
    print("⏳ Waiting for container to be ready...")
    time.sleep(3)

    test_api()
