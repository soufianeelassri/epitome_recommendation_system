import os
import io
import base64
import json
import logging
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element, CompositeElement, Image, Table
from PIL import Image as PILImage

os.environ["TABLE_IMAGE_CROP_PAD"] = "1"
os.environ["EXTRACT_IMAGE_BLOCK_CROP_HORIZONTAL_PAD"] = "20"
os.environ["EXTRACT_IMAGE_BLOCK_CROP_VERTICAL_PAD"] = "10"

logger = logging.getLogger(__name__)

def validate_image_data(image_data):
    try:
        if isinstance(image_data, PILImage.Image):
            return image_data
        
        if isinstance(image_data, bytes):
            image = PILImage.open(io.BytesIO(image_data))
            return image
        
        if isinstance(image_data, str):
            if image_data.startswith('data:'):
                image_data = image_data.split(',', 1)[1]
            
            image_bytes = base64.b64decode(image_data)
            
            image = PILImage.open(io.BytesIO(image_bytes))
            return image
        
        logger.error(f"Unsupported image data type: {type(image_data)}")
        return None
        
    except Exception as e:
        logger.error(f"Error validating image data: {e}")
        return None

def _process_single_element(element: Element, pdf_filename: str):
    category = getattr(element, 'category', 'Uncategorized')
    
    raw_metadata = vars(element.metadata) if element.metadata else {}
    raw_metadata["source_file"] = pdf_filename
    raw_metadata["page_number"] = getattr(element.metadata, 'page_number', None) if hasattr(element, 'metadata') and element.metadata else None
    raw_metadata["category"] = category
    
    metadata = serialize_metadata(raw_metadata)

    if isinstance(element, Table) or category == "Table":
        table_text = element.text
        if table_text and table_text.strip():
            table_data = {"text": table_text}
            if hasattr(element.metadata, "text_as_html") and element.metadata.text_as_html is not None:
                table_data["html"] = element.metadata.text_as_html
            return {"type": "table", "content": table_data, "metadata": metadata}
            
    elif isinstance(element, Image) or category == "Image":
        if hasattr(element, 'metadata') and element.metadata:
            image_payload = None
            
            if hasattr(element.metadata, 'image_base64') and element.metadata.image_base64:
                image_payload = element.metadata.image_base64
            
            elif hasattr(element.metadata, 'image_path') and element.metadata.image_path:
                try:
                    with open(element.metadata.image_path, 'rb') as img_file:
                        image_bytes = img_file.read()
                        image_payload = base64.b64encode(image_bytes).decode('utf-8')
                except Exception as e:
                    logger.warning(f"Could not read image file {element.metadata.image_path}: {e}")
            
            elif hasattr(element.metadata, 'image') and element.metadata.image:
                image_payload = element.metadata.image
            
            if image_payload:
                try:
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

def process_pdf(file_path: str):
    if not os.path.exists(file_path):
        logger.error(f"PDF file not found: {file_path}")
        return []
        
    pdf_filename = os.path.basename(file_path)
    logger.info(f"Starting PDF extraction for: {pdf_filename}")
    
    try:
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
            if isinstance(element, CompositeElement):
                if hasattr(element.metadata, 'orig_elements') and element.metadata.orig_elements is not None:
                    for sub_element in element.metadata.orig_elements:
                        processed = _process_single_element(sub_element, pdf_filename)
                        if processed:
                            processed_elements.append(processed)
                            if processed["type"] == "image":
                                image_count += 1
                else:
                    processed = _process_single_element(element, pdf_filename)
                    if processed:
                        processed_elements.append(processed)
                        if processed["type"] == "image":
                            image_count += 1
            else:
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