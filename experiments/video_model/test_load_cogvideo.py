"""
    Load the cogvideo model. 
"""
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from modified_dit import ModifiedCogVideoXTransformer3DModel
from pipeline import ModifiedCogVideoXPipeline

def make_concept_attention_video(concepts, concept_attention_maps):
    """
        For each concept, create a video using matplotlib where each frame is displayed as a heatmap.

        Inputs:
            concepts: List[str]
            concept_attention_maps: torch.Tensor of shape (num_concepts, num_frames, height, width)
    """
    num_concepts, num_frames, height, width = concept_attention_maps.shape

    fig, axes = plt.subplots(1, num_concepts, figsize=(len(concepts) * 7, 5))
    if num_concepts == 1:
        axes = [axes]

    def update(frame):
        for i, ax in enumerate(axes):
            ax.clear()
            ax.set_title(concepts[i])
            heatmap = concept_attention_maps[i, frame, :, :].cpu().numpy()
            ax.imshow(
                heatmap, 
                cmap='inferno', 
                interpolation='nearest',
                vmin=concept_attention_maps.min(),
                vmax=concept_attention_maps.max()
            )

    ani = animation.FuncAnimation(fig, update, frames=num_frames, repeat=False)
    ani.save('results/concept_attention_video.mp4', writer='ffmpeg', fps=2)

if __name__ == "__main__":
    transformer = ModifiedCogVideoXTransformer3DModel.from_pretrained(
        "THUDM/CogVideoX-2b",
        subfolder="transformer",
        torch_dtype=torch.float16
    )

    pipe = ModifiedCogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-2b", 
        transformer=transformer,
        torch_dtype=torch.float16
    )

    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    prompt = "A golden retriever dog running in a grassy park. Trees in the background. Blue sky as well."
    
    concepts = ["dog", "grass", "sky", "tree"]
    video, concept_attention_dict = pipe(
        prompt=prompt, 
        concepts=concepts,
        guidance_scale=6, 
        use_dynamic_cfg=True, 
        num_inference_steps=50,
        concept_attention_kwargs= {
            "timesteps": list(range(0, 50)),
            "layers": list(range(0, 30)),
        }
    )
    video = video.frames[0]

    print(concept_attention_dict["concept_attention_maps"].shape)

    concept_attention_maps = concept_attention_dict["concept_attention_maps"]
    make_concept_attention_video(concepts, concept_attention_maps)

    export_to_video(video, "results/output.mp4", fps=4)