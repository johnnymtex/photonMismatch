int_vars = ["num_shots", "num_modes_per_shot", "angle", "num_pixels", "ADC_bits"]

class Config:
    def __init__(self, file_path):
        vars = read_variables_from_file(file_path)
        for key, value in vars.items():
            setattr(self, key, value)

def read_variables_from_file(file_path):
    variables = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Strip any leading/trailing whitespace and split the line into key and value
                line = line.strip()
                if line and line[0] != "#":  # Ensure the line is not empty
                    key, value = line.split(sep=' = ', maxsplit=1)  # Split only at the first space or comma
                    if key in int_vars:
                        variables[key] = int(value)
                    else:
                        variables[key] = float(value)
        return variables
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None