import os
import re
import spacy
import json

# Load SpaCy model
import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    doc = nlp(text)
    return " ".join(
        token.lemma_.lower()
        for token in doc
        if token.pos_ in {'PRON', 'ADP', 'CCONJ', 'SCONJ', 'AUX', 'DET'} 
        and token.is_alpha
    )

#Directories
base_directory = "Data"
preprocessed_path = os.path.join(base_directory, "preprocessed")
os.makedirs(preprocessed_path, exist_ok=True)
functionwd_folder = os.path.join(base_directory, "preprocessed", "function_wd")
os.makedirs(functionwd_folder, exist_ok=True)
movie_folder = os.path.join(functionwd_folder, "Movies")
os.makedirs(movie_folder, exist_ok=True) 


def preprocess_and_save_txt(base_dir):
    dataset_path = os.path.join(base_dir, "kagglehub", "datasets", "gufukuro", "movie-scripts-corpus", "versions", "1", 
                                "movie_characters", "data", "movie_character_texts", "movie_character_texts")
    
    if not os.path.isdir(dataset_path):
        print(f"Directory not found: {dataset_path}")
        return

    for movie_name in os.listdir(dataset_path):
        original_movie_path = os.path.join(dataset_path, movie_name)
        if not os.path.isdir(original_movie_path):
            continue

        
        new_movie_path = os.path.join(movie_folder, movie_name[:-8])
        os.makedirs(new_movie_path, exist_ok=True)


        for character_file in os.listdir(original_movie_path):
            if not character_file.endswith(".txt"):
                continue


            file_path = os.path.join(original_movie_path, character_file)

            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Filter and clean dialog lines
            cleaned_lines = []
            for line in lines:
                line = re.sub(r"[0-9]+\)+", "", line.strip())
                if line.startswith("  dialog: "):
                    line = re.sub("  dialog: ", "", line)
                    line = re.sub(r"[\[\]]+", "", line).strip()
                    cleaned_lines.append(line)

            combined_text = " ".join(cleaned_lines).lower().strip()
            ready_text = preprocess_text(combined_text)

            if ready_text:
                clean_filename = re.sub("_text.txt", ".txt", character_file)
                output_path = os.path.join(new_movie_path, clean_filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as out_f:
                    out_f.write(ready_text)
                print(f"✅ Saved {movie_name} in {output_path}")


def char_name_extractor(contentwd_folder):

    characters = {}
    movie_folder = os.path.join(contentwd_folder, "Movies")

    if not os.path.isdir(contentwd_folder):
        print(f"Directory not found: {contentwd_folder}")
        return characters

    for movie_title in os.listdir(movie_folder):
        movie = os.path.join(movie_folder, movie_title)
        if not os.path.isdir(movie):
            print(f"no {movie} path")
            continue

        character_names = []

        for character in os.listdir(movie) :
            if character.endswith(".txt"):
                character_names.append(character)

        if character_names :
            characters[movie_title] = character_names
            
    if characters :
        name_dir = os.path.join(preprocessed_path, "Keywords")
        os.makedirs(name_dir, exist_ok=True)
        characters_name_file = os.path.join(name_dir, "characters_function.txt")
        with open(characters_name_file, "w", encoding="utf-8") as out_f:
            json.dump(characters, out_f, indent=4)
        print(f"✅ Saved characters dictionary to {characters_name_file}")
    
    return characters


# ---- MAIN ----
base_directory = "Data"
preprocess_and_save_txt(base_directory)
print("✅ Preprocessing complete, and txts saved.")
char_name_extractor(contentwd_folder="Data/preprocessed/content_wd")# 
