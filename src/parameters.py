import json
import os
import tempfile

root = os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.pardir))


def load_parameters(
    default_parameters={},
    default_output_directory=tempfile.tempdir,
    output_filenames={},
):
    configuration_file = os.environ.get("CONFIGURATION_FILE")
    if configuration_file:
        with open(configuration_file) as f:
            configuration = json.load(f)
            output_directory = configuration["output_directory"]
            parameters = {**default_parameters, **configuration["parameters"]}
            output_files = {
                key: os.path.join(output_directory, path)
                for key, path in output_filenames.items()
            }
    else:
        parameters = default_parameters
        output_files = {
            key: os.path.join(root, default_output_directory, path)
            for key, path in output_filenames.items()
        }
    return parameters, output_files
