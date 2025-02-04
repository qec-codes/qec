import importlib.resources as resources
import json

def test_package_data_correctly_installed():
    # Read and parse JSON file from package data
    with resources.files("qec").joinpath("code_instances/saved_codes/test.json").open("r") as f:
        data = json.load(f)

    # Print dictionary
    assert data['hello'] == 'world'