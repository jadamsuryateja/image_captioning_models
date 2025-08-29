# app.py
from flask import Flask, request, jsonify, render_template
import torch
from PIL import Image
import io
import os
import logging
from werkzeug.middleware.proxy_fix import ProxyFix

from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoProcessor, Kosmos2ForConditionalGeneration,
    CLIPProcessor, CLIPModel,
    GPT2LMHeadModel, GPT2Tokenizer
)
from sentence_transformers import SentenceTransformer
import huggingface_hub
from huggingface_hub import hf_hub_download  # Updated import

# Update the model loading to use environment cache
os.environ['TRANSFORMERS_CACHE'] = '/opt/render/project/src/.cache/huggingface'
os.environ['HF_HOME'] = '/opt/render/project/src/.cache/huggingface'

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder='templates')
app.wsgi_app = ProxyFix(app.wsgi_app)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load BLIP model directly from Hugging Face
print("Loading BLIP model...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model.to(device)
blip_model.eval()

# Load Kosmos-2 model directly from Hugging Face
print("Loading Kosmos-2 model...")
model_name = "microsoft/kosmos-2-patch14-224"
kosmos_processor = AutoProcessor.from_pretrained(model_name)
kosmos_model = Kosmos2ForConditionalGeneration.from_pretrained(model_name)
kosmos_model.to(device)
kosmos_model.eval()

# Load CLIP model
print("Loading CLIP model...")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.to(device)
clip_model.eval()

# Load GPT-2
print("Loading GPT-2 model...")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_model.to(device)
gpt2_model.eval()
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

# Load SentenceBERT
print("Loading SentenceBERT model...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
sbert_model.to(device)
sbert_model.eval()

# Function to generate Kosmos-2 caption with grounding
def generate_kosmos_caption(image_bytes, prompt="<grounding> Describe the image in one short generic sentence."):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = kosmos_processor(text=prompt, images=image, return_tensors="pt").to(device)

        output_ids = kosmos_model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            max_new_tokens=64,
            min_new_tokens=8,
            num_beams=3
        )

        generated_text = kosmos_processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        processed_text, _ = kosmos_processor.post_process_generation(generated_text)

        caption = (
            processed_text
            .replace("<image>", "")
            .replace("</image>", "")
            .replace("<grounding>", "")
            .replace("</grounding>", "")
            .strip()
        )
        return caption
    except Exception as e:
        print(f"Error generating Kosmos-2 caption: {e}")
        return "Could not generate caption."

def generate_blip(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt")
    output = blip_model.generate(**inputs)
    return blip_processor.decode(output[0], skip_special_tokens=True)

def generate_clip_ranking(image_bytes, captions):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = clip_processor(text=captions, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    return captions[torch.argmax(probs).item()]

def generate_gpt2_enhancement(caption):
    input_ids = gpt2_tokenizer.encode(caption, return_tensors="pt")
    output = gpt2_model.generate(
        input_ids, 
        max_length=50,
        num_beams=5,
        temperature=0.7,
        no_repeat_ngram_size=2
    )
    return gpt2_tokenizer.decode(output[0], skip_special_tokens=True)

def generate_sbert_fusion(captions):
    embeddings = sbert_model.encode(captions, convert_to_tensor=True)
    centroid = torch.mean(embeddings, dim=0)
    similarities = torch.nn.functional.cosine_similarity(embeddings, centroid.unsqueeze(0))
    best_idx = torch.argmax(similarities).item()
    return captions[best_idx]

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/trycaption')
def trycaption():
    return render_template('trycaption.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/caption', methods=['POST'])
def caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image_bytes = file.read()

    try:
        # Generate base captions
        cap_blip = generate_blip(image_bytes)
        cap_kosmos = generate_kosmos_caption(image_bytes)
        
        # Create list of base captions
        base_captions = [cap_blip, cap_kosmos]
        
        # Generate enhanced captions using other models
        cap_clip = generate_clip_ranking(image_bytes, base_captions)
        cap_gpt2 = generate_gpt2_enhancement(cap_clip)
        cap_sbert = generate_sbert_fusion(base_captions + [cap_gpt2])

        return jsonify({'captions': [
            {'model': 'BLIP', 'caption': cap_blip},
            {'model': 'Kosmos-2', 'caption': cap_kosmos},
            {'model': 'CLIP Fusion', 'caption': cap_clip},
            {'model': 'Naive Fusion', 'caption': cap_sbert},
            {'model': 'VAE Fusion', 'caption': cap_gpt2}
        ]})

    except Exception as e:
        print(f"Error in caption generation: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    # Use production server settings when not in debug
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))