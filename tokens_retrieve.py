import os
import sys
import argparse
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from src.HookedLVLM import HookedLVLM
from src.lvlm_lens import create_interactive_logit_lens


def prepend_cls_token(embeddings, position=0, hidden_dim=4096):
    device = embeddings[0].device
    b, d = embeddings[0].shape[0], embeddings[0].shape[2]
    print(f"b:{b}, d:{d}")
    with torch.no_grad():
        token_embeddings = torch.stack([layer[:,position,:] for layer in embeddings], dim=1).to(device) 
    cls_token = torch.zeros(1, 1, d, device=device).expand(b, 1, d)
    pos_embed = torch.zeros(1, token_embeddings.size(1) + 1, d, device=device)
    embd = torch.cat([cls_token, token_embeddings], dim=1)
    embd = embd + pos_embed[:,:embd.size(1), :]
    return embd

def filter_embeddings(embeddings, indices, prompt):
    text_embeddings = [layer[:, :len(prompt), :] for layer in embeddings]
    image_embeddings = [layer[:, len(prompt):, :] for layer in embeddings]
    indices = torch.tensor(indices, dtype=torch.long, device=embeddings[0].device)
    return [layer[indices] for layer in image_embeddings]
    

def process_images(image,image_folder, save_folder, device, quantize_type):
    # Import Model
    model = HookedLVLM(device=device, quantize=True, quantize_type=quantize_type)
    norm = model.model.language_model.model.norm
    lm_head = model.model.language_model.lm_head
    tokenizer = model.processor.tokenizer
    model_name = model.model.config._name_or_path.split("/")[-1]
    text_question = "Describe the image meow meow meow meow."
    prompt = f"USER: <image>\n{text_question} ASSISTANT:"
    hidden_states = model.forward(image, prompt, output_hidden_states=True).hidden_states
    print(len(hidden_states[0]))
    create_interactive_logit_lens(hidden_states, norm, lm_head, tokenizer, image, model_name, image_folder, prompt, save_folder)
    return hidden_states, prompt

def main():
    parser = argparse.ArgumentParser(description="Process images using HookedLVLM model")
    parser.add_argument("--image_folder", default="/home/uas-dtu/c_data/cropped_person1_frame17.jpg", help="Path to the folder containing images")
    parser.add_argument("--save_folder", default="./output3", help="Path to save the results")
    parser.add_argument("--device", default="cuda", help="Device to run the model on")
    parser.add_argument("--quantize_type", default="fp16", help="Quantization type")
    parser.add_argument("--num_images", type=int, help="Number of images to process (optional)")

    args = parser.parse_args()

    process_images(args.image_folder, args.save_folder, args.device, args.quantize_type, args.num_images)

if __name__ == "__main__":
    main()
