docker rm -f groundingdino_container 2>/dev/null || true
docker run --gpus '"device=1"' -it --rm \
  --name groundingdino_container \
  -v "$(pwd)/output:/app/GroundingDINO/output" \
  groundingdino_image
