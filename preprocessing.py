import re 

def cleanup_string(string):
    string = string.encode('ascii', 'ignore').decode('ascii')
    return string

def cleanup_object(input):
    if isinstance(input, str):
        return cleanup_string(input)
    elif isinstance(input, list):
        return [cleanup_object(item) for item in input]
    else:
        return input

def cleanup_dataframe(df):
    for col in df.columns:
        df[col] = df[col].apply(cleanup_object)

# FORMAT CHECKER:
_LINE_PATTERN_A = re.compile('^\d+\t.*\t\w+$') # <id> <TAB> <text> <TAB> <class_label>

def check_format(input_file):
    with open(input_file, encoding='utf-8') as f:
        next(f)
        file_content = f.read().strip()
        for i, line in enumerate(file_content.split('\n')):
            if not _LINE_PATTERN_A.match(line): 
                print(f"Wrong line format: {line}")
                return False
    
    print(f"File '{input_file}' is in the correct format.")
    return True