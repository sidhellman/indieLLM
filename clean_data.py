# Specify the file paths
input_file_path = 'input.txt'
output_file_path = 'cleaned_input.txt'  # Change this if you want to save the cleaned text as a new file

# Characters to be removed
chars_to_remove = "~¢£¥§©«­°·»½ÂÆÜàâèéêîïôöü –—‘’‚“”•…∞ﬁ"

# Read the original text
with open(input_file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Remove the specified characters
cleaned_text = text.translate({ord(c): None for c in chars_to_remove})

# Write the cleaned text
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(cleaned_text)

print(f"Cleaned text written to: {output_file_path}")
