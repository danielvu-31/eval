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
    parser.add_argument(
        "--path_ckpt",
        type=str,
        default="/lustre/scratch/client/movian/research/users/anhnd72/viet/pretrained_models/model_fid8.1"
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
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        
        self.unet = UNet2DConditionModel.from_pretrained(path_ckpt_sbv2, subfolder="unet")
        self.unet.eval()
        
        self.last_timestep = torch.ones((1,), dtype=torch.int64, device="cuda")
        self.last_timestep = self.last_timestep * (self.noise_scheduler.config.num_train_timesteps - 1)
        self.first_timestep = torch.ones((1,), dtype=torch.int64, device="cuda")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")

        pipe = StableDiffusionPipeline.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder"
        ).to(dtype=torch.float32)
        
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

class Evaluator:
    def __init__(self):
        self.mps = MPS.from_pretrained("RE-N-Y/mpsv1")
        self.hpsv2 = HPSv2.from_pretrained("RE-N-Y/hpsv21")
        self.imagereward = ImageReward.from_pretrained("RE-N-Y/ImageReward")

    def evaluate_hpsv2(self,images, prompts):
        return self.hpsv2.score(images,prompts)

    def evaluate_mps(self,images, prompts):
        return self.mps.score(images,prompts)
    
    def evaluate_imagereward(self,images, prompts):
        return self.imagereward.score(images,prompts)
    
    def evaluate_hpsv3(self,images, prompts):
        return self.hpsv3.score(images,prompts)

def print_gpu_usage(label=""):
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved  = torch.cuda.memory_reserved(i) / 1024**3
        total     = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"[{label}] GPU {i}: {allocated:.2f}GB allocated | {reserved:.2f}GB reserved | {total:.2f}GB total")

@torch.no_grad()
def generate_and_evaluate(prompts, sbv2_gen, evaluator, evaluate_fns, accelerator, batch_size=8):
    dataset = PromptDataset(prompts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    unet = sbv2_gen.unet.to(accelerator.device)
    vae = sbv2_gen.vae.to(accelerator.device)
    text_encoder = sbv2_gen.text_encoder.to(accelerator.device)

    dataloader= accelerator.prepare(dataloader)

    total_scores = {name: torch.tensor(0.0, device=accelerator.device) for name in evaluate_fns}
    total_count  = torch.tensor(0, device=accelerator.device)

    for batch_prompts in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        prompt_embeds = get_text_embeddings(sbv2_gen.tokenizer, text_encoder, batch_prompts)
        shape = (prompt_embeds.shape[0], 4, 64, 64)
        noise = torch.randn(shape, device=accelerator.device, dtype=torch.float32)

        images = generate_image(
            unet, vae, sbv2_gen, noise, prompt_embeds, sbv2_gen.last_timestep
        ).clamp(0, 1)

        with torch.no_grad():
            for name, eval_fn in evaluate_fns.items():
                scores = eval_fn(images, batch_prompts)
                if torch.is_tensor(scores):
                    total_scores[name] += scores.sum() if scores.dim() > 0 else scores * len(batch_prompts)
                else:
                    total_scores[name] += float(scores) * len(batch_prompts)

        total_count += len(batch_prompts)

        del prompt_embeds, images, noise
        torch.cuda.empty_cache()

    for name in total_scores:
        total_scores[name] = accelerator.reduce(total_scores[name], reduction="sum")
    total_count = accelerator.reduce(total_count, reduction="sum")
    del dataloader, text_encoder, unet, vae
    torch.cuda.empty_cache()
    return {name: (score / total_count).item() for name, score in total_scores.items()}

def main():
    args = parse_args()
    accelerator = Accelerator()

    sbv2_gen = SBV2Gen(
        path_ckpt_sbv2=args.path_ckpt
    )

    hpsv2_dataset = hpsv2.benchmark_prompts('all')
    IR_dataset    = load_json("/lustre/scratch/client/movian/research/users/kiennt104/CrossDistill/imagereward.json")

    folder_path = "/lustre/scratch/client/movian/research/users/kiennt104/CrossDistill/HPDv3/benchmark"
    data_dict = {}
    with accelerator.main_process_first():
        for file_name in tqdm(os.listdir(folder_path), disable=not accelerator.is_local_main_process):
            category  = file_name.split("_")[1]
            file_path = os.path.join(folder_path, file_name)
            data_dict[category] = load_json(file_path)

    hpsv2_prompts = [p for style, prompts in hpsv2_dataset.items() for p in prompts]
    ir_prompts    = [data['prompt'] for data in IR_dataset]
    hpsv3_prompts = [entry['caption'] for category, json_data in data_dict.items() for entry in json_data]


    accelerator.print("Generating & Evaluating HPSV3 dataset...")
    hpsv3_model = HPSv3.from_pretrained("RE-N-Y/hpsv3").to(accelerator.device)
    
    hpsv3_results = generate_and_evaluate(
        hpsv3_prompts, sbv2_gen,
        None,
        evaluate_fns={
            "hpsv3": lambda imgs, prompts: hpsv3_model.score(imgs, prompts),
        },
        accelerator=accelerator,
        batch_size=2
    )
    del hpsv3_model
    torch.cuda.empty_cache(); gc.collect()

    evaluator = Evaluator()
    
    accelerator.print("Generating & Evaluating HPSV2 dataset...")
    evaluator.hpsv2 = evaluator.hpsv2.to(accelerator.device)
    evaluator.mps = evaluator.mps.to(accelerator.device)
    evaluator.imagereward = evaluator.imagereward.to(accelerator.device)
    
    hpsv2_results = generate_and_evaluate(
        hpsv2_prompts, sbv2_gen,
        evaluator,
        evaluate_fns={
            "hpsv2": evaluator.evaluate_hpsv2,
            "mps":   evaluator.evaluate_mps,
        },
        accelerator=accelerator,
        batch_size=8
    )
    torch.cuda.empty_cache(); gc.collect()

    accelerator.print("Generating & Evaluating IR dataset...")
    ir_results = generate_and_evaluate(
        ir_prompts, sbv2_gen,
        evaluator,
        evaluate_fns={
            "mps": evaluator.evaluate_mps,
        },
        accelerator=accelerator,
        batch_size=8
    )
    torch.cuda.empty_cache(); gc.collect()


    results = {
        "hpsv2":    hpsv2_results["hpsv2"],
        "mps_HPSV2": hpsv2_results["mps"],
        "mps_IR":   ir_results["mps"],
        "hpsv3":    hpsv3_results["hpsv3"],
    }

    if accelerator.is_main_process:
        print(results)
        save_json(results, args.save_dir)


if __name__ == "__main__":
    main()