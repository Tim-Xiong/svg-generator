from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

class ImageGenerator:
    def __init__(self, model_id="stabilityai/stable-diffusion-2-1-base", device="cuda"):
        """
        Initialize the image generator with a specific model.
        
        Args:
            model_id (str): The model identifier for the stable diffusion model.
                Default is "stabilityai/stable-diffusion-2-1-base".
            device (str): The device to run the model on, either "cuda" or "cpu".
                Default is "cuda".
        """
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            scheduler=scheduler, 
            torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to(device)
        self.positive_prompt = "simple, icon"
        self.negative_prompt = "3d, blurry, complex geometry, realistic"
    
    def generate(self, prompt, negative_prompt=None, output_path=None, num_images=1, num_inference_steps=50):
        """
        Generate an image based on the provided prompt.
        
        Args:
            prompt (str): The text description to generate an image from.
            negative_prompt (str, optional): Elements to avoid in the generated image.
                If None, uses the default negative prompt.
            output_path (str, optional): Path to save the generated image.
                If None, the image is not saved to disk.
            num_images (int, optional): Number of images to generate.
                
        Returns:
            PIL.Image.Image: The generated image.
        """
        prompt = f"{prompt}, {self.positive_prompt}"
        if negative_prompt is None:
            negative_prompt = self.negative_prompt
        images = self.pipe(
            prompt, 
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            num_images_per_prompt=num_images
        ).images
        
        if output_path:
            for i, image in enumerate(images):
                image.save(f".cache/{output_path.replace('.png', f'_{i}.png')}")
            
        return image

# Example usage
if __name__ == "__main__":
    generator = ImageGenerator()
    import time
    start_time = time.time()
    image = generator.generate(
        prompt="magenta trapezoids layered on a transluscent silver sheet",
        output_path="sheet.png",
        num_images=4
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
