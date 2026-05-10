import os
from PIL import Image

NUM_OF_AUTHORS = 8
TARGET_SIZE = (128, 64) 
SOURCE_DIR = "." 
OUTPUT_DIR = "processed_dataset" 

def resize_and_pad(img, target_size):
    #Pomniejszenie slowa z zachowaniem proporcji 
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    # Tworzenie nowego obrazu o docelowym rozmiarze z białym tłem
    new_img = Image.new("L", target_size, 255)
    
    # wkljenianie pomniejszonego obrazu na białe tło, tak aby był wyśrodkowany
    paste_x = (target_size[0] - img.size[0]) // 2
    paste_y = (target_size[1] - img.size[1]) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img

def build_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for author_no in range(1, NUM_OF_AUTHORS + 1):
        print(f"Processing Author {author_no}...")
        
        author_dir = os.path.join(SOURCE_DIR, f"author{author_no}")
        desc_file = os.path.join(author_dir, "word_places.txt")
        output_author_dir = os.path.join(OUTPUT_DIR, f"author_{author_no}")
        os.makedirs(output_author_dir, exist_ok=True)
        
        # otwieranie word_places.txt i czytanie jego zawartości
        try:
            with open(desc_file, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Could not find {desc_file}. Skipping.")
            continue
            
        word_count = 0
        current_image_name = ""
        current_image = None
        
        # Petla przez kazda linie w word_places.txt, przetwarzanie slow i zapisywanie ich do processed_dataset
        for line in lines:
            parts = line.split()
            
            # pomijanie pustych linii i komentarzy
            if not parts or parts[0] == '%':
                continue
                
            # ekstrakcja nazwy obrazu i budowanie sciezki do niego
            raw_image_name = parts[0].strip('\'"')
            image_basename = os.path.basename(raw_image_name.replace('\\', '/'))
            image_path = os.path.join(author_dir, "skany", image_basename)
            
            # nowe zdjecie jest wgrywane tylko wtedy, gdy jego nazwa różni się od aktualnie załadowanego obrazu.
            if image_basename != current_image_name:
                try:
                    current_image = Image.open(image_path).convert("L")
                    current_image_name = image_basename
                except Exception as e:
                    print(f"  Error loading {image_path}: {e}")
                    current_image = None
                    continue
                    
            if current_image is None:
                continue
                
            if len(parts) < 5:
                continue
                
            try:

                word = " ".join(parts[1:-4])
                # wspolrzedne sa zawsze w ostatnich 4 elementach linii
                row1, col1 = int(parts[-4]), int(parts[-3])
                row2, col2 = int(parts[-2]), int(parts[-1])
                
                # Upewniamy się, że współrzędne są uporządkowane poprawnie
                left = min(col1, col2)
                right = max(col1, col2)
                upper = min(row1, row2)
                lower = max(row1, row2)
                
                # Jeśli współrzędne są takie same, oznacza to, że nie ma obszaru do wycięcia, pomijamy 
                if left == right or upper == lower:
                    print(f"  Skipping 0-pixel crop in {image_basename}")
                    continue

                cropped_word = current_image.crop((left, upper, right, lower))
                processed_word = resize_and_pad(cropped_word, TARGET_SIZE)

                #zapisywanie przetworzonego słowa do folderu docelowego z nazwą zawierającą oryginalne słowo
                clean_word = "".join(filter(str.isalnum, word)) # Usuwa znaki specjalne z nazwy słowa
                save_path = os.path.join(output_author_dir, f"{clean_word}_{word_count:04d}.png")
                processed_word.save(save_path)
                
                word_count += 1
                
            except Exception as e:
                print(f"  Failed to process a word in {image_basename}: {e}")
                
        print(f"Author {author_no} complete: Saved {word_count} processed words.\n")

if __name__ == "__main__":
    print("Starting dataset extraction...\n")
    build_dataset()
    print("Dataset successfully built!")