from Utils.phonemize.cotlet_utils import *
import cutlet

katsu = cutlet.Cutlet(ensure_ascii=False)
katsu.use_foreign_spelling = False

def process_latin_text(ml):
    # Check for small characters and replace them

    # Initialize Cutlet for romaji conversion

    # Convert to romaji and apply transformations
    # output = katsu.romaji(ml, capitalize=False).lower()

    output = ml.lower()
    

    # Replace specific romaji sequences
    if 'j' in output:
        output = output.replace('j', "dʑ")
    if 'y' in output:
        output = output.replace('y', "j")
    if 'tt' in output:
        output = output.replace('tt', "ʔt")
    if 't t' in output:
        output = output.replace('t t', "ʔt")
    if ' ʔt' in output:
        output = output.replace(' ʔt', "ʔt")
    if 'ssh' in output:
        output = output.replace('ssh', "ɕɕ")

    # Convert romaji to IPA
    output = Roma2IPA(convert_numbers_in_string(output))

    
    output = hira2ipa(output)

    # Apply additional transformations
    output = replace_chars_2(output)
    output = replace_repeated_chars(replace_tashdid_2(output))
    output = nasal_mapper(output)

    # Final adjustments
    if " ɴ" in output:
        output = output.replace(" ɴ", "ɴ")
        
    if ' neɽitai ' in output:
        output = output.replace(' neɽitai ', "naɽitai")

    if 'harɯdʑisama' in output:
        output = output.replace('harɯdʑisama', "arɯdʑisama")


    if "ki ni ɕinai" in output:
        output = re.sub(r'(?<!\s)ki ni ɕinai', r' ki ni ɕinai', output)

    if 'ʔt' in output:
        output = re.sub(r'(?<!\s)ʔt', r'ʔt', output)

    if 'de aɽoɯ' in output:
        output = re.sub(r'(?<!\s)de aɽoɯ', r' de aɽoɯ', output)

        
    return output.lstrip()


def replace_repeating_a(output):
    # Define patterns and their replacements
    patterns = [
        (r'(aː)\s*\1+\s*', r'\1~'),  # Replace repeating "aː" with "aː~~"
        (r'(aːa)\s*aː', r'\1~'),     # Replace "aːa aː" with "aː~~"
        (r'aːa', r'aː~'),             # Replace "aːa" with "aː~"
        (r'naː\s*aː', r'naː~'),       # Replace "naː aː" with "naː~"
        (r'(oː)\s*\1+\s*', r'\1~'),  # Replace repeating "oː" with "oː~~"
        (r'(oːo)\s*oː', r'\1~'),     # Replace "oːo oː" with "oː~~"
        (r'oːo', r'oː~'),              # Replace "oːo" with "oː~"
        (r'(eː)\s*\1+\s*', r'\1~'),  
        (r'(e)\s*\1+\s*', r'\1~'),  
        (r'(eːe)\s*eː', r'\1~'),     
        (r'eːe', r'eː~'),             
        (r'neː\s*eː', r'neː~'),       
    ]

    
    # Apply each pattern to the output
    for pattern, replacement in patterns:
        output = re.sub(pattern, replacement, output)
    
    return output


def latn_phonemize(text):
    
    # if "っ" in text:
    #     text = text.replace("っ","ʔ")
        
    output = post_fix(process_latin_text(text))
    #output = text
    
    if " ɴ" in output:
        output = output.replace(" ɴ", "ɴ")
    if "y" in output:
        output = output.replace("y", "j")
    if "ɯa" in output:
        output = output.replace("ɯa", "wa")
        
    if "a aː" in output:
        output = output.replace("a aː","a~")
    if "a a" in output:
        output = output.replace("a a","a~")



        
          
    output = replace_repeating_a((output))
 
    output = re.sub(r'\s+~', '~', output)
    
    if "oː~o oː~ o" in output:
        output = output.replace("oː~o oː~ o","oː~~~~~~")
    if "aː~aː" in output:
        output = output.replace("aː~aː","aː~~~")
    if "oɴ naː" in output:
        output = output.replace("oɴ naː","onnaː")
    if "aː~~ aː" in output:
        output = output.replace("aː~~ aː","aː~~~~")
    if "oː~o" in output:
        output = output.replace("oː~o","oː~~")
    if "oː~~o o" in output:
        output = output.replace("oː~~o o","oː~~~~") # yeah I'm too tired to learn regex how did you know

    output = random_space_fix(output)
    output = random_sym_fix(output) # fixing some symbols, if they have a specific white space such as miku& sakura -> miku ando sakura
    output = random_sym_fix_no_space(output) # same as above but for those without white space such as miku&sakura -> miku ando sakura
    # if "ɯ" in output:
    #     output = output.replace("ɯ","U")ｓｓ
    # if "ʔ" in output:
    #     output = output.replace("ʔ","!")
    
    return  output.lstrip()
# def process_row(row):
#     return {'phonemes': [phonemize(word) for word in row['phonemes']]}
