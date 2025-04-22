import os
import tempfile
import logging
import re
import subprocess
import cairosvg
from lxml import etree
import kagglehub
from gen_image import ImageGenerator
import vtracer

svg_constraints = kagglehub.package_import('metric/svg-constraints')

class MLModel:
    def __init__(self, model_id="stabilityai/stable-diffusion-2-1-base", device="cuda"):
        """
        Initialize the SVG generation pipeline.
        
        Args:
            model_id (str): The model identifier for the stable diffusion model.
            device (str): The device to run the model on, either "cuda" or "cpu".
        """
        self.image_generator = ImageGenerator(model_id=model_id, device=device)
        self.default_svg = """<svg width="256" height="256" viewBox="0 0 256 256"><circle cx="50" cy="50" r="40" fill="red" /></svg>"""
        self.constraints = svg_constraints.SVGConstraints()
        self.timeout_seconds = 90
    
    def predict(self, description, simplify=True, color_precision=6, 
                gradient_step=10, filter_speckle=4, path_precision=8):
        """
        Generate an SVG from a text description.
        
        Args:
            description (str): The text description to generate an image from.
            simplify (bool): Whether to simplify the SVG paths.
            color_precision (int): Color quantization precision.
            gradient_step (int): Gradient step for color quantization (not used by vtracer).
            filter_speckle (int): Filter speckle size.
            path_precision (int): Path fitting precision.
            
        Returns:
            str: The generated SVG content.
        """
        try:
            # Step 1: Generate image using diffusion model
            images = self.image_generator.generate(description)
            image = images[0]
            
            # Step 2: Save image to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
                temp_img_path = temp_img.name
                image.save(temp_img_path)
            
            # Step 3: Convert image to SVG using vtracer
            with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as temp_svg:
                temp_svg_path = temp_svg.name
            
            # Process the image with vtracer using parameters directly
            vtracer.convert_image_to_svg_py(
                temp_img_path,
                temp_svg_path,
                colormode='color',
                hierarchical='stacked' if simplify else 'cutout',
                mode='spline',
                filter_speckle=filter_speckle,
                color_precision=color_precision,
                path_precision=path_precision,
                corner_threshold=60,
                length_threshold=4.0,
                max_iterations=10,
                splice_threshold=45
            )
            
            # Step 4: Read the generated SVG
            with open(temp_svg_path, 'r') as f:
                svg_content = f.read()
            
            # Clean up temporary files
            os.unlink(temp_img_path)
            os.unlink(temp_svg_path)
            
            # Step 5: Enforce constraints
            svg_content = self.enforce_constraints(svg_content)
            
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

    def optimize_svg(self, svg_content):
        """
        Optimize the SVG content using SVGO.
        
        Args:
            svg_content (str): The SVG content to optimize.
            
        Returns:
            str: The optimized SVG content.
        """
        try:
            with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as temp_svg:
                temp_svg_path = temp_svg.name
                temp_svg.write(svg_content.encode('utf-8'))
            
            with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as temp_out:
                temp_out_path = temp_out.name
            
            subprocess.run(["svgo", temp_svg_path, "-o", temp_out_path], check=True)
            
            with open(temp_out_path, 'r') as f:
                optimized_svg = f.read()
            
            os.unlink(temp_svg_path)
            os.unlink(temp_out_path)
            
            return optimized_svg
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("Warning: SVGO not found or failed. Returning unoptimized SVG.")
            return svg_content


# Example usage
if __name__ == "__main__":
    model = MLModel()
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