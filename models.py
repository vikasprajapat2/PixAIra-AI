import torch
from diffusers import DiffusionPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import gc

# Global variables to hold models
t2i_pipe = None
i2t_model = None
i2t_processor = None

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_t2i_model():
    global t2i_pipe
    if t2i_pipe is None:
        print("Loading Text-to-Image model (SD 1.5)...")
        device = get_device()
        try:
            # Using Stable Diffusion 1.5 which is lighter and more reliable than SDXL
            t2i_pipe = DiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True
            )
            t2i_pipe.to(device)
            # Enable memory slicing to save VRAM
            if device == "cuda":
                t2i_pipe.enable_attention_slicing()
            print(f"T2I Model loaded on {device}")
        except Exception as e:
            print(f"Error loading T2I model: {e}")
            raise e
    return t2i_pipe

def load_i2t_model():
    global i2t_model, i2t_processor
    if i2t_model is None:
        print("Loading Image-to-Text model (BLIP Base)...")
        # Using BLIP Base instead of Large for speed and lower memory usage
        i2t_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        i2t_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        device = get_device()
        i2t_model.to(device)
        print(f"I2T Model loaded on {device}")
    return i2t_model, i2t_processor

def text_to_image(prompt):
    # 1. Offload Image-to-Text model to CPU to free up VRAM
    global i2t_model
    if i2t_model is not None and torch.cuda.is_available():
        print("Offloading I2T model to CPU...")
        i2t_model.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()
    
    # 2. Load/Get T2I pipe
    pipe = load_t2i_model()
    
    # 3. Ensure pipe is on the correct device
    device = get_device()
    if pipe.device.type != device:
        pipe.to(device)

    print(f"Generating image for: {prompt}")
    # 25 steps is standard for SD1.5. 
    image = pipe(prompt, num_inference_steps=25).images[0]
    return image

def image_to_text(image_path):
    # 1. Offload Text-to-Image model to CPU to free up VRAM
    global t2i_pipe
    if t2i_pipe is not None and torch.cuda.is_available():
        print("Offloading T2I model to CPU...")
        t2i_pipe.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

    # 2. Load/Get I2T model
    model, processor = load_i2t_model()
    
    # 3. Ensure model is on the correct device
    device = get_device()
    if model.device.type != device:
        model.to(device)

    print(f"Generating caption for: {image_path}")
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt").to(model.device)
    
    # Generate caption
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption