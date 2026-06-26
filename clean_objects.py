import os
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path

# Target directories based on CollectionBuilder structure
INPUT_DIR = "objects"
OUTPUT_DIR = "objects"  # Overwrites/saves directly into objects folder

def clean_image(cv_img):
    """
    Cleans an image by deskewing, tightly cropping out empty white borders, 
    and boosting contrast for optimal text readability.
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # 2. Deskew (Fix slight text rotation)
    coords = np.column_stack(np.where(gray < 200))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) > 0.5:
        (h, w) = cv_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cv_img = cv2.warpAffine(cv_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # 3. Tight Crop (Find bounding box of actual text content)
    # Invert image so text is white and background is black
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours of the text block
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get largest text bounding container
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # Add a comfortable 20px padding so text doesn't touch the edge
        padding = 20
        img_h, img_w = cv_img.shape[:2]
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_w - x, w + (padding * 2))
        h = min(img_h - y, h + (padding * 2))
        
        # Crop to the text block
        cv_img = cv_img[y:y+h, x:x+w]
        gray = gray[y:y+h, x:x+w]

        # 4. Maximize Text Contrast (White paper background, crisp black text)
    # Rescale intensity values to clip faded colors
    xp = [0, 40, 200, 255]  # Input pixel intensity breakpoints
    fp = [0, 0, 255, 255]   # Output pixel intensity mappings (forces darks to black, lights to white)
    x_lookup = np.interp(np.arange(256), xp, fp).astype('uint8')
    cleaned_gray = cv2.LUT(gray, x_lookup)

    # Convert back to clean RGB format
    final_img = cv2.cvtColor(cleaned_gray, cv2.COLOR_GRAY2RGB)
    return final_img

def process_batch():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Could not find '{INPUT_DIR}' directory.")
        return

    for filename in os.listdir(INPUT_DIR):
        file_path = os.path.join(INPUT_DIR, filename)
        
        # Skip subdirectories like small/ and thumbs/
        if os.path.isdir(file_path):
            continue
            
        base_name, ext = os.path.splitext(filename)
        ext = ext.lower()
        
        # Process PDFs (convert first page to image, clean it, and overwrite as PDF)
        if ext == '.pdf':
            print(f"Processing PDF: {filename}")
            try:
                pages = convert_from_path(file_path, first_page=1, last_page=1, dpi=200)
                if pages:
                    # Convert PIL image to OpenCV format
                    cv_img = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
                    cleaned = clean_image(cv_img)
                    
                    # Convert back to PDF container and save over original
                    pil_img = Image.fromarray(cleaned)
                    pil_img.save(file_path, "PDF", resolution=200.0)
                    print(f" Successfully cleaned PDF: {filename}")
            except Exception as e:
                print(f" Error processing PDF {filename}: {e}")
                
        # Process Standard Images
        elif ext in ['.jpg', '.jpeg', '.png', '.tiff']:
            print(f"Processing Image: {filename}")
            try:
                cv_img = cv2.imread(file_path)
                if cv_img is not None:
                    cleaned = clean_image(cv_img)
                    cv2.imwrite(file_path, cv2.cvtColor(cleaned, cv2.COLOR_RGB2BGR))
                    print(f" Successfully cleaned Image: {filename}")
            except Exception as e:
                print(f" Error processing Image {filename}: {e}")

if __name__ == "__main__":
    process_batch()
