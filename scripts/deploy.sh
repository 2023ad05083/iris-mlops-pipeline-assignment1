#!/bin/bash

set -e

ENVIRONMENT=${1:-staging}
IMAGE_NAME="iris-mlops-api"
CONTAINER_NAME="iris-api-${ENVIRONMENT}"

echo "Deploying to ${ENVIRONMENT} environment..."

# Stop and remove existing container
docker stop ${CONTAINER_NAME} 2>/dev/null || true
docker rm ${CONTAINER_NAME} 2>/dev/null || true

# Pull latest image
docker pull ${DOCKER_USERNAME}/${IMAGE_NAME}:latest

# Run new container
docker run -d \
    --name ${CONTAINER_NAME} \
    -p 8000:8000 \
    --restart unless-stopped \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    ${DOCKER_USERNAME}/${IMAGE_NAME}:latest

echo "Deployment completed successfully!"
echo "Container ${CONTAINER_NAME} is running on port 8000"