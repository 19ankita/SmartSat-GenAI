import torch
from diffusers import StableDiffusionPipeline
import os

def generate_satellite_design(prompt=None, max_temp=80, weight_constraint=0.5):
    if prompt is None:
        prompt = f"3D printed satellite component optimized for thermal performance with max operating temperature of {max_temp}°C and weight constraint of {weight_constraint} kg."
    #pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    #pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    image = pipe(prompt).images[0]
    os.makedirs("generative_design/outputs", exist_ok=True)
    output_path = "generative_design/outputs/diffusion_sat_component.png"
    image.save(output_path)
    return output_path

if __name__ == "__main__":
    result = generate_satellite_design()
    print(f"Generated design image saved to: {result}")