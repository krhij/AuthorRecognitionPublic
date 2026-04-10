import os
from PIL import Image

# --- CONFIGURATION ---
NUM_OF_AUTHORS = 8
# Standard size for the neural network: (Width, Height)
TARGET_SIZE = (128, 64) 
# Where the raw data lives
SOURCE_DIR = "." 
# Where the clean data will be saved
OUTPUT_DIR = "processed_dataset" 

def resize_and_pad(img, target_size):
    """
    Resizes an image while keeping its original proportions, 
    then pads the leftover space with white background.
    """
    # 1. Shrink the image to fit within the target box, keeping aspect ratio
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    # 2. Create a blank white canvas of the exact target size ("L" mode = Grayscale, 255 = White)
    new_img = Image.new("L", target_size, 255)
    
    # 3. Calculate exactly where to paste the image so it sits dead center
    paste_x = (target_size[0] - img.size[0]) // 2
    paste_y = (target_size[1] - img.size[1]) // 2
    
    # 4. Paste the shrunken image onto the white canvas
    new_img.paste(img, (paste_x, paste_y))
    return new_img

def build_dataset():
    # Create the main output folder if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for author_no in range(1, NUM_OF_AUTHORS + 1):
        print(f"Processing Author {author_no}...")
        
        # Setup paths for this specific author
        author_dir = os.path.join(SOURCE_DIR, f"author{author_no}")
        desc_file = os.path.join(author_dir, "word_places.txt")
        
        # Create a specific output folder for this author's processed words
        output_author_dir = os.path.join(OUTPUT_DIR, f"author_{author_no}")
        os.makedirs(output_author_dir, exist_ok=True)
        
        # Open the text file containing the coordinates
        try:
            with open(desc_file, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Could not find {desc_file}. Skipping.")
            continue
            
        word_count = 0
        current_image_name = ""
        current_image = None
        
        # Loop through every word listed in the file
        for line in lines:
            parts = line.split()
            
            # Skip empty lines or comment lines starting with %
            if not parts or parts[0] == '%':
                continue
                
            # Clean up the image filename (remove quotes and handle slashes)
            raw_image_name = parts[0].strip('\'"')
            # Extract just the filename itself (e.g., 'skany/page1.png' -> 'page1.png')
            image_basename = os.path.basename(raw_image_name.replace('\\', '/'))
            image_path = os.path.join(author_dir, "skany", image_basename)
            
            # To save memory, only load the big scanned page if we've moved to a new one
            if image_basename != current_image_name:
                try:
                    # Load the image and instantly convert to Grayscale ("L")
                    current_image = Image.open(image_path).convert("L")
                    current_image_name = image_basename
                except Exception as e:
                    print(f"  Error loading {image_path}: {e}")
                    current_image = None
                    continue
                    
            if current_image is None:
                continue
                
            # Extract coordinates
           # --- UPGRADED EXTRACTION LOGIC ---
            # If the line doesn't have enough data to contain 4 coordinates, skip it
            if len(parts) < 5:
                continue
                
            try:
                # FIX 1: Read from the END of the list using negative indices. 
                # This ignores spaces in the word text (e.g., "Z daleka").
                row1, col1 = int(parts[-4]), int(parts[-3])
                row2, col2 = int(parts[-2]), int(parts[-1])
                
                # FIX 2: Mathematical sorting. 
                # Ensures left is always smaller than right, upper smaller than lower.
                left = min(col1, col2)
                right = max(col1, col2)
                upper = min(row1, row2)
                lower = max(row1, row2)
                
                # Prevent crashing on 0-pixel wide/tall boxes
                if left == right or upper == lower:
                    print(f"  Skipping 0-pixel crop in {image_basename}")
                    continue

                # 1. Crop the word
                cropped_word = current_image.crop((left, upper, right, lower))
                
                # 2. Resize and pad
                processed_word = resize_and_pad(cropped_word, TARGET_SIZE)
                
                # 3. Save it as a standardized PNG file
                # The format :04d ensures filenames look like word_0001.png, word_0002.png
                save_path = os.path.join(output_author_dir, f"word_{word_count:04d}.png")
                processed_word.save(save_path)
                
                word_count += 1
                
            except Exception as e:
                print(f"  Failed to process a word in {image_basename}: {e}")
                
        print(f"✓ Author {author_no} complete: Saved {word_count} processed words.\n")

if __name__ == "__main__":
    print("Starting dataset extraction...\n")
    build_dataset()
    print("Dataset successfully built! You are ready for PyTorch.")