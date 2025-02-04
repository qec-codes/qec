import importlib.resources as resources
import json

### This test confirms that package data is correctly installed by pip 

def test_package_data_correctly_installed():
    # Read and parse JSON file from package data
    with resources.files("qec").joinpath("code_instances/saved_codes/test.json").open(
        "r"
    ) as f:
        data = json.load(f)

    # We can change this to an actual code file once we have decided on the save format.
    assert data["hello"] == "world"
