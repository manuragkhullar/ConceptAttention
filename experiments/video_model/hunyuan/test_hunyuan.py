
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, HunyuanVideoTransformer3DModel, HunyuanVideoPipeline
from diffusers.utils import export_to_video

from pipeline import ModifiedHunyuanVideoPipeline
from modified_dit import ModifiedHunyuanVideoTransformer3DModel

if __name__ == "__main__":
    # quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
    # transformer_8bit = HunyuanVideoTransformer3DModel.from_pretrained(
    #     "hunyuanvideo-community/HunyuanVideo",
    #     subfolder="transformer",
    #     quantization_config=quant_config,
    #     torch_dtype=torch.bfloat16,
    # )

    modified_transformer = ModifiedHunyuanVideoTransformer3DModel.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )
    pipeline = ModifiedHunyuanVideoPipeline.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        transformer=modified_transformer,
        torch_dtype=torch.bfloat16,
        # device_map="balanced",
    ).to("cuda")
    # pipeline.enable_sequential_cpu_offload()
    # pipeline.enable_vae_slicing()
    pipeline.vae.enable_tiling()

    prompt = "A man bounces a basketball in the park. Sky in background. "
    video = pipeline(prompt=prompt, num_frames=61, num_inference_steps=2).frames[0]
    export_to_video(video, "results/basketball.mp4", fps=15)