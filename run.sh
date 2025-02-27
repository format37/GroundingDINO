docker rm -f groundingdino_container 2>/dev/null || true
docker run --gpus '"device=0"' -it --rm \
  --name groundingdino_container \
  --network host \
  -v "$(pwd)/outputs:/app/GroundingDINO/outputs" \
  -v "$(pwd)/cache/huggingface:/root/.cache/huggingface" \
  groundingdino_image
