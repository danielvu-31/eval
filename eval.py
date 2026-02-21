
from imscore.hps.model import HPSv2
from imscore.mps.model import MPS
from imscore.imreward.model import ImageReward
from imscore.hpsv3.model import HPSv3

import torch
import numpy as np
from PIL import Image
from einops import rearrange
import argparse
import json
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel,StableDiffusionPipeline
from transformers import AutoTokenizer, CLIPTextModel
import hpsv2
import os
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import Dataset, DataLoader
import gc
import torchvision.transforms.functional as TF

    
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metrics", 
        type=str, 
        choices=["mps", "hpsv2", "imagereward", "hpsv3"],
        default="imagereward"
    )
    parser.add_argument(
        "--save_dir",  
        type=str,
        default="results.json"
    )
    return parser.parse_args()

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_text_embeddings(tokenizer, text_encoder, prompt):
    input_ids = tokenizer(prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to("cuda")
    encoder_hidden_states = text_encoder(input_ids)[0]
    return encoder_hidden_states
    
def tokenize_captions(captions, tokenizer):
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids

class SBV2Gen():
    def __init__(
        self, 
        path_ckpt_sbv2,
        model_name="/home/dungnt206/workspace/pretrain/huggingface/stable-diffusion-2-1-base"
    ):
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to("cuda")
        
        self.unet = UNet2DConditionModel.from_pretrained(path_ckpt_sbv2, subfolder="unet").to("cuda")
        self.unet.eval()
        
        self.last_timestep = torch.ones((1,), dtype=torch.int64, device="cuda")
        self.last_timestep = self.last_timestep * (self.noise_scheduler.config.num_train_timesteps - 1)
        self.first_timestep = torch.ones((1,), dtype=torch.int64, device="cuda")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")

        pipe = StableDiffusionPipeline.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder"
        ).to("cuda", dtype=torch.float32)
        
        # prepare stuff
        self.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to("cuda")
        self.alpha_t = (self.alphas_cumprod[self.last_timestep] ** 0.5).view(-1, 1, 1, 1)
        # self.sigma_t = ((1 - self.alphas_cumprod[self.last_timestep]) ** 0.5).view(-1, 1, 1, 1)

    def get_alpha_t(self, timesteps):
        return (self.alphas_cumprod[timesteps] ** 0.5).view(-1, 1, 1, 1)
    def get_sigma_t(self, timesteps):
        return ((1. - self.alphas_cumprod[timesteps]) ** 0.5).view(-1, 1, 1, 1)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


@torch.no_grad()
def generate_image(unet, vae, sbv2_gen, noise, encoder_hidden_state, timesteps):
    model_pred = unet(noise, timesteps, encoder_hidden_state).sample
    alpha_t = sbv2_gen.get_alpha_t(timesteps)
    sigma_t = sbv2_gen.get_sigma_t(timesteps)

    pred_original_sample = (noise - sigma_t * model_pred) / alpha_t
    if sbv2_gen.noise_scheduler.config.thresholding:
            pred_original_sample = sbv2_gen.noise_scheduler._threshold_sample(
            pred_original_sample
        )
    elif sbv2_gen.noise_scheduler.config.clip_sample:
        clip_sample_range = sbv2_gen.noise_scheduler.config.clip_sample_range
        pred_original_sample = pred_original_sample.clamp(
            -clip_sample_range, clip_sample_range
        )
    pred_original_sample = pred_original_sample / sbv2_gen.vae.config.scaling_factor
    refine_image = (sbv2_gen.vae.decode(pred_original_sample).sample + 1) / 2

    return refine_image


class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

def generate_for_prompts(prompts, sbv2_gen, accelerator, save_dir, batch_size=2):
    os.makedirs(save_dir, exist_ok=True)
    
    dataset = PromptDataset(prompts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataloader, text_encoder, unet, vae = accelerator.prepare(dataloader, sbv2_gen.text_encoder, sbv2_gen.unet, sbv2_gen.vae)

    local_prompts = []
    local_image_paths = []

    for batch_idx, batch_prompts in enumerate(tqdm(dataloader, disable=not accelerator.is_local_main_process)):
        prompt_embeds = get_text_embeddings(
            sbv2_gen.tokenizer, text_encoder, batch_prompts
        )
        shape = (prompt_embeds.shape[0], 4, 64, 64)
        noise = torch.randn(shape, device=accelerator.device, dtype=torch.float32)

        images = generate_image(
            unet, vae, sbv2_gen, noise, prompt_embeds, sbv2_gen.last_timestep
        ).detach().cpu().clamp(0, 1)

        for i, (prompt, image) in enumerate(zip(batch_prompts, images.unbind(0))):
            img_name = f"gpu{accelerator.process_index}_batch{batch_idx}_img{i}.png"
            img_path = os.path.join(save_dir, img_name)
            TF.to_pil_image(image).save(img_path)
            local_prompts.append(prompt)
            local_image_paths.append(img_path)

        del prompt_embeds, images, noise
        torch.cuda.empty_cache()

    all_prompts     = gather_object(local_prompts)      
    all_image_paths = gather_object(local_image_paths)   
    
    if accelerator.is_main_process:
        mapping = [
            {"prompt": p, "image_path": ip}
            for p, ip in zip(all_prompts, all_image_paths)
        ]
        save_json(mapping, os.path.join(save_dir, "mapping.json"))

    accelerator.wait_for_everyone()
    return os.path.join(save_dir, "mapping.json")


class Evaluator:
    def __init__(self):
        self.mps = MPS.from_pretrained("RE-N-Y/mpsv1").to("cuda")
        self.hpsv2 = HPSv2.from_pretrained("RE-N-Y/hpsv21").to("cuda")
        self.imagereward = ImageReward.from_pretrained("RE-N-Y/ImageReward").to("cuda")

    def evaluate_hpsv2(self,images, prompts):
        return self.hpsv2.score(images,prompts)

    def evaluate_mps(self,images, prompts):
        return self.mps.score(images,prompts)
    
    def evaluate_imagereward(self,images, prompts):
        return self.imagereward.score(images,prompts)
    
    def evaluate_hpsv3(self,images, prompts):
        return self.hpsv3.score(images,prompts)

class ImagePromptDataset(Dataset):
    def __init__(self, image_paths, prompts):
        assert len(image_paths) == len(prompts), "Images and prompts must have the same length"
        self.image_paths = image_paths
        self.prompts = prompts

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = TF.to_tensor(image)  # (C, H, W)
        return image, self.prompts[idx]


def evaluate_dataset(image_paths, prompts, evaluator, evaluate_fn, accelerator, batch_size=16):
    dataset = ImagePromptDataset(image_paths, prompts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataloader = accelerator.prepare(dataloader)

    total_score = torch.tensor(0.0, device=accelerator.device)
    total_count = torch.tensor(0,   device=accelerator.device)

    for images, batch_prompts in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        images = images.to(accelerator.device)  # (B, C, H, W)
        scores = evaluate_fn(images, batch_prompts)

        total_score += scores.sum() if torch.is_tensor(scores) else scores
        total_count += len(batch_prompts)

    # Sum across all GPUs
    total_score = accelerator.reduce(total_score, reduction="sum")
    total_count = accelerator.reduce(total_count, reduction="sum")

    average_score = total_score / total_count
    return average_score.item()


def main():
    args = parse_args()
    accelerator = Accelerator()

    sbv2_gen = SBV2Gen(
        path_ckpt_sbv2="/lustre/scratch/client/movian/research/users/anhnd72/viet/pretrained_models/model_fid8.1"
    )

    sbv2_gen.text_encoder = sbv2_gen.text_encoder.to(accelerator.device)
    sbv2_gen.unet = sbv2_gen.unet.to(accelerator.device)
    sbv2_gen.vae = sbv2_gen.vae.to(accelerator.device)


    hpsv2_dataset = hpsv2.benchmark_prompts('all')
    IR_dataset    = load_json("/lustre/scratch/client/movian/research/users/kiennt104/CrossDistill/imagereward.json")

    folder_path = "/lustre/scratch/client/movian/research/users/kiennt104/CrossDistill/HPDv3/benchmark"
    data_dict = {}
    with accelerator.main_process_first():         
        for file_name in tqdm(os.listdir(folder_path), disable=not accelerator.is_local_main_process):
            category  = file_name.split("_")[1]
            file_path = os.path.join(folder_path, file_name)
            data_dict[category] = load_json(file_path)

    hpsv2_prompts = [
        prompt
        for style, prompts in hpsv2_dataset.items()
        for prompt in prompts
    ]

    ir_prompts = [data['prompt'] for data in IR_dataset]

    hpsv3_prompts = [
        entry['caption']
        for category, json_data in data_dict.items()
        for entry in json_data
    ]
    
    accelerator.print("Generating HPSV2 images...")
    mapping_hpsv2 = generate_for_prompts(hpsv2_prompts, sbv2_gen, accelerator, save_dir = "HPSV2")

    gc.collect()
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()

    accelerator.print("Generating ImageReward images...")
    mapping_IR = generate_for_prompts(ir_prompts, sbv2_gen, accelerator, save_dir = "IR")

    accelerator.print("Generating HPSV3 images...")
    mapping_hpsv3 = generate_for_prompts(hpsv3_prompts, sbv2_gen, accelerator, save_dir = "HPSV3")

    if accelerator.is_main_process:
        print(f"HPSV2  → {len(prompt_hpsv2)} prompts, {len(images_hpsv2)} images")
        print(f"IR     → {len(prompts_IR)} prompts, {len(images_IR)} images")
        print(f"HPSV3  → {len(prompt_hpsv3)} prompts, {len(images_hpsv3)} images")

    # Evaluation
    data_HPSV2 = load_json(mapping_hpsv2)
    data_IR = load_json(mapping_IR)
    data_HPSV3 = load_json(mapping_hpsv3)

    hpsv2_prompts = [data['prompt'] for data in data_HPSV2]
    ir_prompts = [data['prompt'] for data in data_IR]
    hpsv3_prompts = [data['prompt'] for data in data_HPSV3]

    images_hpsv2 = [data['image_path'] for data in data_HPSV2]
    images_IR = [data['image_path'] for data in data_IR]
    images_hpsv3 = [data['image_path'] for data in data_HPSV3]


    evaluator = Evaluator()

    with torch.no_grad():
        # HPSv2
        accelerator.print("Evaluating HPSV2...")
        average_hpsv2_score = evaluate_dataset(
            images_hpsv2, hpsv2_prompts,
            evaluator, evaluator.evaluate_hpsv2,
            accelerator, batch_size=16
        )

        # MPS
        accelerator.print("Evaluating MPS...")
        # MPS for IR
        average_mps_score_IR = evaluate_dataset(
            images_IR, ir_prompts,
            evaluator, evaluator.evaluate_mps,
            accelerator, batch_size=16
        )
        # MPS for HPSV2
        average_mps_score_HPSV2 = evaluate_dataset(
            images_hpsv2, hpsv2_prompts,
            evaluator, evaluator.evaluate_mps,
            accelerator, batch_size=16
        )
        del evaluator
        gc.collect()
        torch.cuda.empty_cache()
        
        # HPSV3
        accelerator.print("Evaluating HPSV3...")
        hpsv3 = HPSv3.from_pretrained("RE-N-Y/hpsv3").to(accelerator.device)
        def evaluate_hpsv3(images, prompts):
            return hpsv3.score(images, prompts)

        average_hpsv3_score = evaluate_dataset(
            images_hpsv3, hpsv3_prompts,
            None, evaluate_hpsv3,
            accelerator, batch_size=4
        )
    results = {
        "hpsv2": average_hpsv2_score,
        "mps_IR": average_mps_score_IR,
        "mps_HPSV2": average_mps_score_HPSV2,
        "hpsv3": average_hpsv3_score
    }
    save_json(results, args.save_dir)
    
if __name__ == "__main__":
    main()