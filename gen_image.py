from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

class ImageGenerator:
    def __init__(self, model_id="stabilityai/stable-diffusion-2-1-base", device="cuda"):
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            scheduler=scheduler, 
            torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to(device)
    
    def generate(self, prompt, negative_prompt=None, output_path=None):
        image = self.pipe(prompt, negative_prompt=negative_prompt).images[0]
        
        if output_path:
            image.save(output_path)
            
        return image


# Example usage
if __name__ == "__main__":
    generator = ImageGenerator()
    import time
    start_time = time.time()
    image = generator.generate(
        prompt="magenta trapezoids layered on a transluscent silver sheet, simple, icon",
        negative_prompt="3d, blurry, complex geometry, realistic",
        output_path="sheet.png"
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
