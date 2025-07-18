import os
import io
import base64
import json
import logging
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element, CompositeElement, Image, Table
from PIL import Image as PILImage

# Set environment variables for image extraction
os.environ["TABLE_IMAGE_CROP_PAD"] = "1"
os.environ["EXTRACT_IMAGE_BLOCK_CROP_HORIZONTAL_PAD"] = "20"
os.environ["EXTRACT_IMAGE_BLOCK_CROP_VERTICAL_PAD"] = "10"

logger = logging.getLogger(__name__)

def validate_image_data(image_data):
    try:
        # Check if image_data is already a PIL Image
        if isinstance(image_data, PILImage.Image):
            return image_data
        
        # If it's bytes, try to open directly
        if isinstance(image_data, bytes):
            image = PILImage.open(io.BytesIO(image_data))
            return image
        
        # If it's a string, it should be base64 encoded
        if isinstance(image_data, str):
            # Remove data URL prefix if present (data:image/png;base64,)
            if image_data.startswith('data:'):
                # Split on comma and take the second part (the actual base64 data)
                image_data = image_data.split(',', 1)[1]
            
            # Decode the base64 string to bytes
            image_bytes = base64.b64decode(image_data)
            
            # Open the image from bytes
            image = PILImage.open(io.BytesIO(image_bytes))
            return image
        
        logger.error(f"Unsupported image data type: {type(image_data)}")
        return None
        
    except Exception as e:
        logger.error(f"Error validating image data: {e}")
        return None

# Helper function to process a single PDF element and extract relevant data
# Returns a dictionary with type, content, and metadata, or None if not relevant
def _process_single_element(element: Element, pdf_filename: str):
    # Extract the category and metadata for the element
    category = getattr(element, 'category', 'Uncategorized')
    
    # Safely extract metadata
    raw_metadata = vars(element.metadata) if element.metadata else {}
    raw_metadata["source_file"] = pdf_filename
    raw_metadata["page_number"] = getattr(element.metadata, 'page_number', None) if hasattr(element, 'metadata') and element.metadata else None
    raw_metadata["category"] = category
    
    # Serialize metadata to handle non-JSON serializable objects
    metadata = serialize_metadata(raw_metadata)

    # Handle table elements
    if isinstance(element, Table) or category == "Table":
        table_text = element.text
        if table_text and table_text.strip():
            table_data = {"text": table_text}
            # Add HTML if available
            if hasattr(element.metadata, "text_as_html") and element.metadata.text_as_html is not None:
                table_data["html"] = element.metadata.text_as_html
            return {"type": "table", "content": table_data, "metadata": metadata}
            
    # Handle image elements
    elif isinstance(element, Image) or category == "Image":
        if hasattr(element, 'metadata') and element.metadata:
            # Try different ways to get image data
            image_payload = None
            
            # Check for image_base64 attribute
            if hasattr(element.metadata, 'image_base64') and element.metadata.image_base64:
                image_payload = element.metadata.image_base64
            
            # Check for image_path attribute
            elif hasattr(element.metadata, 'image_path') and element.metadata.image_path:
                try:
                    with open(element.metadata.image_path, 'rb') as img_file:
                        image_bytes = img_file.read()
                        image_payload = base64.b64encode(image_bytes).decode('utf-8')
                except Exception as e:
                    logger.warning(f"Could not read image file {element.metadata.image_path}: {e}")
            
            # Check for other possible image attributes
            elif hasattr(element.metadata, 'image') and element.metadata.image:
                image_payload = element.metadata.image
            
            if image_payload:
                try:
                    # Validate the image data
                    validated_image = validate_image_data(image_payload)
                    if validated_image:
                        logger.info(f"Successfully processed image from {pdf_filename}")
                        return {"type": "image", "content": image_payload, "metadata": metadata}
                    else:
                        logger.warning(f"Failed to validate image from {pdf_filename}")
                        return None
                except Exception as e:
                    logger.warning(f"Error processing image from {pdf_filename}: {e}")
                    return None
            else:
                logger.warning(f"No image data found in image element from {pdf_filename}")
        
    # Handle text elements
    elif hasattr(element, 'text') and element.text and element.text.strip():
        return {"type": "text", "content": element.text, "metadata": metadata}
        
    return None

def serialize_metadata(metadata):
    clean_metadata = {}
    for key, value in metadata.items():
        try:
            json.dumps(value)
            clean_metadata[key] = value
        except TypeError:
            if hasattr(value, "to_dict"):
                clean_metadata[key] = value.to_dict()
            else:
                clean_metadata[key] = str(value)
    return clean_metadata

# Main function to process a PDF file and extract structured elements
# Returns a list of dictionaries representing text, tables, and images
def process_pdf(file_path: str):
    if not os.path.exists(file_path):
        logger.error(f"PDF file not found: {file_path}")
        return []
        
    pdf_filename = os.path.basename(file_path)
    logger.info(f"Starting PDF extraction for: {pdf_filename}")
    
    try:
        # Partition the PDF into elements using the unstructured library
        elements = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            languages=["eng"],
            infer_table_structure=True,
            extract_image_block_types=["Image", "Table"], 
            extract_image_block_to_payload=True,
            extract_images_in_pdf=True, 
            chunking_strategy="by_title",
            max_characters=1500,
            new_after_n_chars=1000,
            combine_text_under_n_chars=500
        )
    except Exception as e:
        logger.error(f"Error partitioning PDF {pdf_filename}: {e}", exc_info=True)
        return []

    processed_elements = []
    image_count = 0
    
    for element in elements:
        try:
            # If the element is composite, process its original sub-elements
            if isinstance(element, CompositeElement):
                if hasattr(element.metadata, 'orig_elements') and element.metadata.orig_elements is not None:
                    for sub_element in element.metadata.orig_elements:
                        processed = _process_single_element(sub_element, pdf_filename)
                        if processed:
                            processed_elements.append(processed)
                            if processed["type"] == "image":
                                image_count += 1
                else:
                    # Process the composite element itself
                    processed = _process_single_element(element, pdf_filename)
                    if processed:
                        processed_elements.append(processed)
                        if processed["type"] == "image":
                            image_count += 1
            else:
                # Otherwise, process the element directly
                processed = _process_single_element(element, pdf_filename)
                if processed:
                    processed_elements.append(processed)
                    if processed["type"] == "image":
                        image_count += 1
        except Exception as e:
            logger.warning(f"Error processing element in {pdf_filename}: {e}")
            continue
                
    logger.info(f"Extracted {len(processed_elements)} total elements from {pdf_filename}.")
    logger.info(f"Found {image_count} images in {pdf_filename}.")
    return processed_elements