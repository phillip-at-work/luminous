import os
import re

# To use, append references to comments in luminous:
# REF: [Author, Year, Title, {Key1: Value1, Key2: Value2 ...}]
# or
# REF: [Author, Year, Title]

def parse_additional_info(info_str):

    info_dict = {}
    if info_str:
        pairs = info_str.strip('{}').split(', ')
        for pair in pairs:
            key, value = pair.split(': ')
            info_dict[key] = value
    return info_dict

def compile_references(code_directory, output_file):

    ref_pattern = re.compile(r'# REF: \[(.*?)\]')

    func_pattern = re.compile(r'^\s*def\s+(\w+)\s*\(')
    class_pattern = re.compile(r'^\s*class\s+(\w+)\s*')

    references = {}

    for root, _, files in os.walk(code_directory):
        for file in files:
            if file.endswith('.py'):
                with open(os.path.join(root, file), 'r') as f:
                    current_function = None
                    current_class = None
                    for line_number, line in enumerate(f, start=1):
                        
                        # classes
                        class_match = class_pattern.match(line)
                        if class_match:
                            current_class = class_match.group(1)
                            current_function = None  # Reset function context

                        # functions
                        func_match = func_pattern.match(line)
                        if func_match:
                            current_function = func_match.group(1)

                        # refs
                        ref_match = ref_pattern.search(line)
                        if ref_match:
                            ref_content = ref_match.group(1)
                            ref_parts = re.split(r',\s*(?![^{}]*\})', ref_content)
                            if len(ref_parts) == 4:
                                author, year, title, additional_info_str = ref_parts
                                additional_info = parse_additional_info(additional_info_str)
                                additional_info_str = ', '.join(f"{key}: {value}" for key, value in additional_info.items())
                                ref = f"{author}, {year}, {title}, {additional_info_str}"
                            else:
                                author, year, title = ref_parts[:3]
                                ref = f"{author}, {year}, {title}"

                            location = f"{file}, line {line_number}"
                            if current_class:
                                location += f", class {current_class}"
                            if current_function:
                                location += f", function {current_function}"
                            
                            if ref not in references:
                                references[ref] = []
                            references[ref].append(location)

    # write to file
    print(f"Writing references to {output_file}")
    with open(output_file, 'w') as f:
        for ref, locations in sorted(references.items()):
            f.write(f"{ref}\n")
            for location in locations:
                f.write(f"  - Used in: {location}\n")
            f.write("\n")

    print(f"References compiled successfully! Output written to {output_file}")

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_directory = os.path.join(script_dir, '..')
    print(f"Code directory: {code_directory}")

    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    results_dir = os.path.join(project_root, 'results')
    print(f"Results directory: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, 'compiled_references.txt')
    print(f"Output file: {output_file}")

    compile_references(code_directory, output_file)