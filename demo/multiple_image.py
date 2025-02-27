from PIL import Image
import numpy as np
import cv2
import logging
import os
import warnings
import torch
import time

# Set up logging    
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# prepare the environment
os.system("python setup.py build develop --user")
os.system("pip install packaging==21.3")
os.system("pip install gradio==3.50.2")

warnings.filterwarnings("ignore")

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, predict
import groundingdino.datasets.transforms as T
from huggingface_hub import hf_hub_download

logger.info("Starting Grounding DINO single image demo")

def load_model_hf(model_config_path, repo_id, filename, device='cpu'):
    args = SLConfig.fromfile(model_config_path) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    logger.info("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model

def image_transform_grounding(init_image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image, _ = transform(init_image, None) # 3, h, w
    return init_image, image

def image_transform_grounding_for_vis(init_image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
    ])
    image, _ = transform(init_image, None) # 3, h, w
    return image

def main():
    # Model configuration
    config_file = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filename = "groundingdino_swint_ogc.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load the model
    model = load_model_hf(config_file, ckpt_repo_id, ckpt_filename, device)

    # Load and preprocess the image
    IMAGE_PATH = "demo/animals.jpg"
    TEXT_PROMPT = "bernese_mountain_dog.bulldog.raccoon.dhole.african_wild_dog.butterfly.dingo.shark.prairie_dog.salamander.bush_dog.snapper_fish"
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25

    # Load and process image (do this once outside the loop)
    init_image = Image.open(IMAGE_PATH).convert("RGB")
    _, image_tensor = image_transform_grounding(init_image)
    
    # Run inference loop for FPS measurement
    logger.info("Starting inference loop for FPS measurement...")
    num_iterations = 100
    start_time = time.time()
    
    for _ in range(num_iterations):
        boxes, logits, phrases = predict(
            model=model,
            image=image_tensor,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=device
        )
    
    end_time = time.time()
    total_time = end_time - start_time
    fps = num_iterations / total_time
    logger.info(f"Average FPS over {num_iterations} iterations: {fps:.2f}")

    logger.info("Image loaded and processed. Running inference...")

    # Run inference
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=device
    )

    logger.info("Predictions done. Annotating image...")

    # Annotate the image with the detections
    annotated_frame = annotate(
        image_source=np.asarray(image_transform_grounding_for_vis(init_image)),
        boxes=boxes,
        logits=logits,
        phrases=phrases
    )

    logger.info("Saving annotated image...")
    # Save the annotated image
    cv2.imwrite("outputs/result.jpg", annotated_frame)
    logger.info("Annotated image saved to outputs/result.jpg. Sleeping..")
    # # Wait indefinitely to keep the container running
    # while True:
    #     time.sleep(1)

if __name__ == "__main__":
    main()
