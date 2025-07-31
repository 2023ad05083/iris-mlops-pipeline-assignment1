#!/bin/bash

set -e

API_URL=${1:-http://localhost:8000}
MAX_RETRIES=10
RETRY_INTERVAL=5

echo "Performing health check on ${API_URL}..."

for i in $(seq 1 ${MAX_RETRIES}); do
    echo "Attempt ${i}/${MAX_RETRIES}..."
    
    if curl -f -s "${API_URL}/health" > /dev/null; then
        echo "Health check passed!"
        
        # Test prediction endpoint
        echo "Testing prediction endpoint..."
        curl -X POST "${API_URL}/predict" \
            -H "Content-Type: application/json" \
            -d '{
                "features": [
                    {
                        "sepal_length": 5.1,
                        "sepal_width": 3.5,
                        "petal_length": 1.4,
                        "petal_width": 0.2
                    }
                ]
            }' > /dev/null
        
        echo "All health checks passed!"
        exit 0
    fi
    
    echo "Health check failed, retrying in ${RETRY_INTERVAL} seconds..."
    sleep ${RETRY_INTERVAL}
done

echo "Health check failed after ${MAX_RETRIES} attempts"
exit 1