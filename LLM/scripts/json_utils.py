import json
from json_repair import repair_json
import re
from jsonschema import validate, ValidationError
import sys
sys.path.append('..')


def get_schema():
    schema_str = '''{
      "$schema": "http://json-schema.org/draft-07/schema#",
      "title": "Proximity Location Extraction Output",
      "type": "object",
      "properties": {
        "specific_locations": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "description": "List of specific, named, mappable places near the property"
        },
        "general_locations": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "description": "List of general or vague location references (e.g. 'local shops')"
        },
        "parent_locations": {
          "type": "array",
          "items": {
            "type": "string",
            "minLength": 1
          },
          "description": "List of neighbourhoods the property is located in"
        }
      },
      "required": ["specific_locations", "general_locations", "parent_locations"],
      "additionalProperties": false
    }'''
    return json.loads(schema_str)

def validate_json(instance):
    schema = get_schema()
    try:
        validate(instance=instance, schema=schema)
        return True
    except ValidationError as e:
        return False

