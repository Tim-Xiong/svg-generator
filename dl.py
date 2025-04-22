import logging
import re
import cairosvg
import torch
from transformers import AutoModelForCausalLM
from lxml import etree
import kagglehub
from gen_image import ImageGenerator
from starvector.data.util import process_and_rasterize_svg

svg_constraints = kagglehub.package_import('metric/svg-constraints')

class DLModel:
    def __init__(self, model_id="starvector/starvector-8b-im2svg", device="cuda"):
        """
        Initialize the SVG generation pipeline using StarVector.
        
        Args:
            model_id (str): The model identifier for the StarVector model.
            device (str): The device to run the model on, either "cuda" or "cpu".
        """
        self.image_generator = ImageGenerator(model_id="stabilityai/stable-diffusion-2-1-base", device=device)
        self.default_svg = """<svg width="256" height="256" viewBox="0 0 256 256"><circle cx="50" cy="50" r="40" fill="red" /></svg>"""
        self.constraints = svg_constraints.SVGConstraints()
        self.timeout_seconds = 90
        
        # Load StarVector model
        self.device = device
        self.starvector = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            trust_remote_code=True
        )
        self.processor = self.starvector.model.processor
        self.starvector.to(device)
        self.starvector.eval()
    
    def predict(self, description):
        """
        Generate an SVG from a text description.
        
        Args:
            description (str): The text description to generate an image from.
            
        Returns:
            str: The generated SVG content.
        """
        try:
            # Step 1: Generate image using diffusion model
            images = self.image_generator.generate(description)
            image = images[0]
            
            # Save the generated image
            image_path = "diff_image.png"
            image.save(image_path)
            logging.info(f"Intermediate image saved to {image_path}")
            
            # Step 2: Convert image to SVG using StarVector
            processed_image = self.processor(image, return_tensors="pt")['pixel_values'].to(self.device)
            if not processed_image.shape[0] == 1:
                processed_image = processed_image.squeeze(0)
            
            batch = {"image": processed_image}
            with torch.no_grad():
                raw_svg = self.starvector.generate_im2svg(batch, max_length=4000)[0]
                raw_svg, _ = process_and_rasterize_svg(raw_svg)
            
            if 'viewBox' not in raw_svg:
                raw_svg = raw_svg.replace('<svg', f'<svg viewBox="0 0 384 384"')

            # Step 3: Enforce constraints
            svg_content = self.enforce_constraints(raw_svg)
            
            return svg_content
        except Exception as e:
            logging.error(f"Error generating SVG: {e}")
            return self.default_svg

    def enforce_constraints(self, svg_string: str) -> str:
        """Enforces constraints on an SVG string, removing disallowed elements
        and attributes.

        Parameters
        ----------
        svg_string : str
            The SVG string to process.

        Returns
        -------
        str
            The processed SVG string, or the default SVG if constraints
            cannot be satisfied.
        """
        logging.info('Sanitizing SVG...')

        try:
            # Remove XML declaration if it exists
            svg_string = re.sub(r'<\?xml[^>]+\?>', '', svg_string).strip()
            
            parser = etree.XMLParser(remove_blank_text=True, remove_comments=True)
            root = etree.fromstring(svg_string, parser=parser)
        except etree.ParseError as e:
            logging.error('SVG Parse Error: %s. Returning default SVG.', e)
            logging.error('SVG string: %s', svg_string)
            return self.default_svg
    
        elements_to_remove = []
        for element in root.iter():
            tag_name = etree.QName(element.tag).localname
    
            # Remove disallowed elements
            if tag_name not in self.constraints.allowed_elements:
                elements_to_remove.append(element)
                continue  # Skip attribute checks for removed elements
    
            # Remove disallowed attributes
            attrs_to_remove = []
            for attr in element.attrib:
                attr_name = etree.QName(attr).localname
                if (
                    attr_name
                    not in self.constraints.allowed_elements[tag_name]
                    and attr_name
                    not in self.constraints.allowed_elements['common']
                ):
                    attrs_to_remove.append(attr)
    
            for attr in attrs_to_remove:
                logging.debug(
                    'Attribute "%s" for element "%s" not allowed. Removing.',
                    attr,
                    tag_name,
                )
                del element.attrib[attr]
    
            # Check and remove invalid href attributes
            for attr, value in element.attrib.items():
                 if etree.QName(attr).localname == 'href' and not value.startswith('#'):
                    logging.debug(
                        'Removing invalid href attribute in element "%s".', tag_name
                    )
                    del element.attrib[attr]

            # Validate path elements to help ensure SVG conversion
            if tag_name == 'path':
                d_attribute = element.get('d')
                if not d_attribute:
                    logging.warning('Path element is missing "d" attribute. Removing path.')
                    elements_to_remove.append(element)
                    continue # Skip further checks for this removed element
                # Use regex to validate 'd' attribute format
                path_regex = re.compile(
                    r'^'  # Start of string
                    r'(?:'  # Non-capturing group for each command + numbers block
                    r'[MmZzLlHhVvCcSsQqTtAa]'  # Valid SVG path commands (adjusted to exclude extra letters)
                    r'\s*'  # Optional whitespace after command
                    r'(?:'  # Non-capturing group for optional numbers
                    r'-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?'  # First number
                    r'(?:[\s,]+-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)*'  # Subsequent numbers with mandatory separator(s)
                    r')?'  # Numbers are optional (e.g. for Z command)
                    r'\s*'  # Optional whitespace after numbers/command block
                    r')+'  # One or more command blocks
                    r'\s*'  # Optional trailing whitespace
                    r'$'  # End of string
                )
                if not path_regex.match(d_attribute):
                    logging.warning(
                        'Path element has malformed "d" attribute format. Removing path.'
                    )
                    elements_to_remove.append(element)
                    continue
                logging.debug('Path element "d" attribute validated (regex check).')
        
        # Remove elements marked for removal
        for element in elements_to_remove:
            if element.getparent() is not None:
                element.getparent().remove(element)
                logging.debug('Removed element: %s', element.tag)

        try:
            cleaned_svg_string = etree.tostring(root, encoding='unicode', xml_declaration=False)
            return cleaned_svg_string
        except ValueError as e:
            logging.error(
                'SVG could not be sanitized to meet constraints: %s', e
            )
            return self.default_svg

# Example usage
if __name__ == "__main__":
    model = DLModel()
    svg = model.predict("a purple forest at dusk")
    # Convert SVG to PNG
    try:
        # Create a PNG in memory
        png_data = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
        
        # Save the PNG to a file
        with open("output.png", "wb") as f:
            f.write(png_data)
        print("SVG saved as output.png")
    except Exception as e:
        print(f"Error converting SVG to PNG: {e}")