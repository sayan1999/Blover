from PIL import Image
import math, random
from transformers import BartForConditionalGeneration, AutoTokenizer, set_seed
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionUpscalePipeline,
    StableDiffusionLatentUpscalePipeline,
)
from diffusers.image_processor import VaeImageProcessor
import torch, os

PROMPT_LEN, MAX_N_TOKEN = 56, 1024
WIDTH, HEIGHT = 960, 544
GUIDANCE = 7.5
EPOCHS = 35
UPSCALE_EPOCHS = 10
OUTPUT_DIR = "OUTPUTS"


class BLOVER:
    summary = None
    random = None

    def randomize(cls):
        cls.random = random.randint(2**24, 2**32 - 1)
        return cls.random

    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    def __init__(self):
        pass


class BLOG:
    def __init__(self):
        torch.cuda.empty_cache()
        print("Summarizing blog")
        self.model_ckpt = "facebook/bart-large-cnn"
        self.device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_ckpt).to(
            self.device
        )

    def summarize(self, text=""):
        set_seed(BLOVER().randomize())
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        if input_ids.shape[1] <= PROMPT_LEN:
            summary = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            BLOVER.summary = summary
            print("Summary of blog is ->>>>>>>>>>>>>>>>>>>>>>>>>\n" + summary)
            return summary

        n_split = math.ceil(input_ids.shape[1] / MAX_N_TOKEN)
        splits = [
            input_ids[:, MAX_N_TOKEN * i : MAX_N_TOKEN * i + MAX_N_TOKEN]
            for i in range(0, n_split)
        ]
        summarized_splits = [
            self.model.generate(split, max_length=PROMPT_LEN) for split in splits
        ]
        summary = " ".join(
            [
                self.tokenizer.decode(s[0], skip_special_tokens=True)
                for s in summarized_splits
            ]
        )
        return self.summarize(summary)


class COVER:
    def __init__(self):
        torch.cuda.empty_cache()
        print("Creating cover")
        self.device = "cpu"
        self.model_id = "stabilityai/stable-diffusion-2-1"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id, safety_checker=None, torch_dtype=torch.float32
        ).to(self.device)
        self.optimize()

    def optimize(self):
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_sequential_cpu_offload()
        # self.pipe.enable_attention_slicing()
        pass

    def create(self, seed=178327878873):
        generator = torch.Generator(device=self.device)
        generator = generator.manual_seed(BLOVER().randomize())
        BLOVER.low_res_latents = self.pipe(
            prompt=BLOVER.summary,
            height=HEIGHT,
            width=WIDTH,
            guidance_scale=GUIDANCE,
            num_inference_steps=EPOCHS,
            generator=generator,
            output_type="latent",
        ).images
        with torch.no_grad():
            image = self.pipe.decode_latents(BLOVER.low_res_latents)
            image = self.pipe.numpy_to_pil(image)[0]
        # improc = VaeImageProcessor()
        # image = improc.postprocess(BLOVER.low_res_latents, output_type='pil')[0]
        BLOVER.cover = os.path.join(OUTPUT_DIR, f"cover{BLOVER.random}.png")
        image.save(BLOVER.cover)

    def show(self, title="cover"):
        Image.open(f"cover{BLOVER.random}.png").show(title=BLOVER.summary)


class UPSCALAR:
    def __init__(self):
        torch.cuda.empty_cache()
        print("Scaling cover")
        self.device = "cpu"
        self.model_id = "stabilityai/sd-x2-latent-upscaler"
        self.pipe = StableDiffusionLatentUpscalePipeline.from_pretrained(
            self.model_id, torch_dtype=torch.float32
        ).to(self.device)
        self.optimize()

    def optimize(self):
        # self.pipe.enable_xformers_memory_efficient_attention()
        # self.pipe.enable_sequential_cpu_offload()
        # self.pipe.enable_attention_slicing()
        pass

    def scale(self, file=f"cover{BLOVER.random}.png", seed=178327878873):
        generator = torch.Generator(device=self.device)
        generator = generator.manual_seed(seed)
        low_res_img = Image.open(file).convert("RGB")
        upscaled_image = self.pipe(
            image=BLOVER.low_res_latents,
            num_inference_steps=UPSCALE_EPOCHS,
            prompt=BLOVER.summary,
            generator=generator,
        ).images[0]
        BLOVER.coverx = os.path.join(OUTPUT_DIR, f"coverX4{BLOVER.random}.png")
        upscaled_image.save(BLOVER.coverx)

    def show(self, title="coverX4"):
        Image.open(BLOVER.coverx).show(title=title)


if __name__ == "__main__":
    blog = BLOG()
    text = open("blog.txt").read()
    summary = blog.summarize(text)
    del blog

    cover = COVER()
    cover.create()
    cover.show()
    del cover

    upscalar = UPSCALAR()
    upscalar.scale()
    upscalar.show()
    del upscalar
