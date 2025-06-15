import re
import os
import json

def keywords_retrieval(name_dir):
    #    Retrieves keywords from specified text files for female, male, and children.

    female_kw_path = os.path.join(name_dir, "Keywords", "females_keywords.txt")
    male_kw_path = os.path.join(name_dir, "Keywords", "males_keywords.txt")
    children_kw_path = os.path.join(name_dir, "Keywords", "children_keywords.txt")

    with open(female_kw_path, "r", encoding="utf-8") as w:
        female_kw = set([line.strip().lower() for line in w if line.strip()])

    with open(male_kw_path, "r", encoding="utf-8") as m:
        male_kw = set([line.strip().lower() for line in m if line.strip()])

    with open(children_kw_path, "r", encoding="utf-8") as c:
        children_kw = set([line.strip().lower() for line in c if line.strip()])
        
    return female_kw, male_kw, children_kw


def get_text(current_name):
    #  Reads and returns the entire content of a text file.

    with open(current_name, "r", encoding="utf-8") as f:
        text = f.read()
        return text


def determine_gender(name_dir):
    # Determines the gender of characters based on keywords, saves their texts
    #into gender-specific directories, and provides counts per movie and overall.
    

    female_kw, male_kw, children_kw = keywords_retrieval(name_dir)
   
    characters_file = os.path.join(name_dir, "Keywords", "characters_content.txt")
    preprocessed_texts_path = os.path.join(name_dir, "content_wd", "Movies")
    gender_base_path = os.path.join(name_dir, "content_wd", "Gender_based_texts")
    os.makedirs(gender_base_path, exist_ok=True)

    female_docs = os.path.join(gender_base_path, "Women")
    male_docs = os.path.join(gender_base_path, "Men")
    children_docs = os.path.join(gender_base_path, "Children")
    os.makedirs(female_docs, exist_ok=True)
    os.makedirs(male_docs, exist_ok=True)
    os.makedirs(children_docs, exist_ok=True)

    counter_women = 0
    counter_men = 0
    counter_children = 0
    counter_unknown = 0
    

    with open(characters_file, "r", encoding="utf-8") as f:
        characters = json.load(f)

        unknown_char = []
        for movie, character_list in characters.items() :
            
            for char in character_list :

                gender = "unknown"
                name = char[:-4].lower().strip()
                words_character = name.split()           
            
                for word in words_character:
                    if word in children_kw :
                        gender = 'child'
                        break
                    
                    elif word in female_kw :
                        gender = "woman"
                        break
                
                    elif word in male_kw :
                        gender = "man"
                        break
                
                
                if gender == "unknown" :
                    unknown_char.append(name.strip().lower())
                                        
        # Process character files

                current_name = os.path.join(preprocessed_texts_path, movie, char)

                try:

                    if gender == "woman" :
                        with open(os.path.join(female_docs, char), "w", encoding="utf-8") as v:
                            v.write(get_text(current_name))
                        counter_women += 1

                    elif gender == "man":
                        with open(os.path.join(male_docs, char), "w", encoding="utf-8") as f:
                            f.write(get_text(current_name))
                        counter_men += 1

                    elif gender == "child" :
                        with open(os.path.join(children_docs, char), "w", encoding="utf-8") as f:
                            f.write(get_text(current_name))
                        counter_children += 1


                    elif gender == "unknown" :
                        #os.remove(current_name) 
                        print(f"{current_name} is not classifiable")
                        counter_unknown += 1

                except FileNotFoundError:
                    print(f"⚠️ File not found, skipping: {current_name}")
                    continue         

            print(f"✅ Saved gender texts for: {movie}")
    print("\n--- Overall Character Gender Counts Across All Movies ---")
    print(f"Total Women: {counter_women}")
    print(f"Total Men: {counter_men}")
    print(f"Total Children: {counter_children}")
    print(f"Total Unknown: {counter_unknown}")
    print(f"Total characters:", counter_unknown + counter_children + counter_men + counter_women)

# Call the function
name_dir = "Data/preprocessed"
determine_gender(name_dir)
