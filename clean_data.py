with open('/path/to/your/text/data', 'r', encoding='utf-8') as f:
    text = f.read()
print("length of dataset in characters: ", len(text))

# here are all the unique characters that occur in this text, run this separately, copy the special chracters
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# Specify the file paths
input_file_path = '/path/to/original/text'
output_file_path = '/path/to/cleanedup/txt'  # Change this if you want to save the cleaned text as a new file

# Characters to be removed
chars_to_remove = "<<add any weird encoding characters if you see them"

# Read the original text
with open(input_file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Remove the specified characters
cleaned_text = text.translate({ord(c): None for c in chars_to_remove})

# Write the cleaned text
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(cleaned_text)

print(f"Cleaned text written to: {output_file_path}")

