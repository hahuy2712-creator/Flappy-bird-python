import cv2
import numpy as np
import os

def remove_solid_background(image_path: str, output_path: str, bg_color: tuple = None, threshold: int = 10) -> bool:
    """
    Removes a solid background from an image and saves it as a transparent PNG.
    
    Args:
        image_path: Path to the input image.
        output_path: Path to save the output transparent image (should be .png).
        bg_color: A tuple (B, G, R) of the background color to remove. 
                  If None, it automatically samples the top-left pixel.
        threshold: The tolerance level for color matching (0-255).
        
    Returns:
        bool: True if successful, False otherwise.
    """
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return False
        
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read {image_path}.")
        return False
        
    # Get base background color if not provided
    if bg_color is None:
        bg_color = img[0, 0].tolist()
        
    bg_color_arr = np.array(bg_color, dtype=np.int16) # use int16 to prevent overflow in clip
    
    # Create lower and upper bounds for the background color using threshold
    lower_bound = np.clip(bg_color_arr - threshold, 0, 255).astype(np.uint8)
    upper_bound = np.clip(bg_color_arr + threshold, 0, 255).astype(np.uint8)
    
    # Create a mask: 255 where it matches background, 0 elsewhere
    mask = cv2.inRange(img, lower_bound, upper_bound)
    
    # Invert mask: 255 for foreground (what we want to keep), 0 for background
    alpha = cv2.bitwise_not(mask)
    
    # Convert original BGR image to BGRA (add alpha channel)
    b, g, r = cv2.split(img)
    rgba = [b, g, r, alpha]
    result = cv2.merge(rgba)
    
    # Save the resulting image
    success = cv2.imwrite(output_path, result)
    if success:
        print(f"Successfully saved background-removed image to {output_path}")
    else:
        print(f"Failed to save image to {output_path}")
        
    return success

def remove_ai_background(image_path: str, output_path: str) -> bool:
    """
    Removes the background from an image using the 'rembg' AI model.
    This works perfectly for complex images (people, objects, complex backgrounds).
    
    Note: Requires the 'rembg' package installed. 
    Run `pip install rembg` or `uv add rembg` before using.
    """
    try:
        from rembg import remove
    except ImportError:
        print("Error: The 'rembg' package is not installed.")
        print("Please install it using: uv add rembg")
        return False
        
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return False
        
    try:
        with open(image_path, 'rb') as i_file:
            input_data = i_file.read()
            output_data = remove(input_data)
            
        with open(output_path, 'wb') as o_file:
            o_file.write(output_data)
            
        print(f"Successfully saved AI background-removed image to {output_path}")
        return True
    except Exception as e:
        print(f"Error processing image with rembg: {e}")
        return False

# Example Usage
if __name__ == "__main__":
    # Test on a hypothetical image 'dataset/bee.png' 
    input_file = "dataset/bee.png"
    output_solid = "dataset/bee_nobg_solid.png"
    output_ai = "dataset/bee_nobg_ai.png"
    
    if os.path.exists(input_file):
        # 1. Method for Solid Backgrounds (Fast, uses existing OpenCV via numpy)
        print("Testing Solid Background Removal...")
        remove_solid_background(input_file, output_solid, threshold=15)
        
        # 2. Method for Complex Backgrounds (Requires rembg)
        # Uncomment below to test after installing rembg
        # print("\nTesting AI Background Removal...")
        # remove_ai_background(input_file, output_ai)
    else:
        print(f"Place an image at '{input_file}' to run the tests.")
