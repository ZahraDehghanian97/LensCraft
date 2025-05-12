system_prompt = """
You are a cinematography expert who understands camera movements and positioning.
Convert the incoming natural-language camera movement description into JSON that.

Example 1:
    Input: "The camera trucks right while the character remains stationary throughout the entire sequence."
    Output:
    
    {
      "cinematographyPrompts": [
        {
          "initial": {
            "cameraAngle": "eye",
            "shotSize": "mediumShot",
            "subjectView": "front",
            "subjectFraming": "center"
          },
          "movement": {
            "type": "truckRight",
            "speed": "constant"
          },
          "final": {
            "cameraAngle": "eye",
            "shotSize": "mediumShot",
            "subjectView": "front",
            "subjectFraming": "center"
          }
        }
      ]
    }


    Example 2:
    Input: "As the character moves down and backward, the camera pulls out to maintain a consistent framing while capturing their actions."
    Output:
    
    {
      "cinematographyPrompts": [
        {
          "initial": {
            "cameraAngle": "eye",
            "shotSize": "mediumShot",
            "subjectView": "front",
            "subjectFraming": "center"
          },
          "movement": {
            "type": "dollyOut",
            "speed": "constant"
          },
          "final": {
            "cameraAngle": "high",
            "shotSize": "longShot",
            "subjectView": "front",
            "subjectFraming": "center"
          }
        }
      ]
    }


    Example 3:
    Input: "The camera arcs around the subject who stands still, starting from the front and ending at a three-quarter back view."
    Output:
    
    {
      "cinematographyPrompts": [
        {
          "initial": {
            "cameraAngle": "eye",
            "shotSize": "mediumShot",
            "subjectView": "front",
            "subjectFraming": "center"
          },
          "movement": {
            "type": "arcRight",
            "speed": "smoothStartStop"
          },
          "final": {
            "cameraAngle": "eye",
            "shotSize": "mediumShot",
            "subjectView": "threeQuarterBackRight",
            "subjectFraming": "center"
          }
        }
      ]
    }


    Example 4:
    Input: "Starting with an extreme close-up, the camera gradually pulls back to reveal the character's surroundings."
    Output:
    
    {
      "cinematographyPrompts": [
        {
          "initial": {
            "cameraAngle": "eye",
            "shotSize": "extremeCloseUp",
            "subjectView": "front",
            "subjectFraming": "center"
          },
          "movement": {
            "type": "dollyOut",
            "speed": "slowToFast"
          },
          "final": {
            "cameraAngle": "eye",
            "shotSize": "longShot",
            "subjectView": "front",
            "subjectFraming": "center"
          }
        }
      ]
    }

**exactly** matches the schema you have been provided via the API call.
Do **not** add any keys, omit any required keys, or change field names.
"""


CINEMATOGRAPHY_JSON_SCHEMA = {
    "name": "cinematography_schema",
    "description": "Structured description of camera moves for the simulation engine.",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "cinematographyPrompts": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "initial": {
                            "type": "object",
                            "properties": {
                                "cameraAngle": {
                                    "type": "string",
                                    "enum": [
                                        "low", "eye", "high", "overhead", "birdsEye"
                                    ]
                                },
                                "shotSize": {
                                    "type": "string",
                                    "enum": [
                                        "extremeCloseUp", "closeUp", "mediumCloseUp",
                                        "mediumShot", "fullShot", "longShot",
                                        "veryLongShot", "extremeLongShot"
                                    ]
                                },
                                "subjectView": {
                                    "type": "string",
                                    "enum": [
                                        "front", "back", "left", "right",
                                        "threeQuarterFrontLeft", "threeQuarterFrontRight",
                                        "threeQuarterBackLeft", "threeQuarterBackRight"
                                    ]
                                },
                                "subjectFraming": {
                                    "type": "string",
                                    "enum": [
                                        "left", "right", "top", "bottom", "center",
                                        "topLeft", "topRight", "bottomLeft", "bottomRight",
                                        "outerLeft", "outerRight", "outerTop", "outerBottom"
                                    ]
                                },
                            },
                            "required": [
                                "cameraAngle", "shotSize", "subjectView", "subjectFraming"
                            ],
                            "additionalProperties": False
                        },
                        "movement": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "static", "follow", "track", "dollyIn", "dollyOut",
                                        "panLeft", "panRight", "tiltUp", "tiltDown",
                                        "truckLeft", "truckRight", "pedestalUp",
                                        "pedestalDown", "arcLeft", "arcRight",
                                        "craneUp", "craneDown", "dollyOutZoomIn",
                                        "dollyInZoomOut"
                                    ]
                                },
                                "speed": {
                                    "type": "string",
                                    "enum": [
                                        "slowToFast", "fastToSlow", "constant",
                                        "smoothStartStop"
                                    ]
                                },
                            },
                            "required": ["type", "speed"],
                            "additionalProperties": False
                        },
                        "final": {
                            "type": "object",
                            "properties": {
                                "cameraAngle": {
                                    "type": "string",
                                    "enum": [
                                        "low", "eye", "high", "overhead", "birdsEye"
                                    ]
                                },
                                "shotSize": {
                                    "type": "string",
                                    "enum": [
                                        "extremeCloseUp", "closeUp", "mediumCloseUp",
                                        "mediumShot", "fullShot", "longShot",
                                        "veryLongShot", "extremeLongShot"
                                    ]
                                },
                                "subjectView": {
                                    "type": "string",
                                    "enum": [
                                        "front", "back", "left", "right",
                                        "threeQuarterFrontLeft", "threeQuarterFrontRight",
                                        "threeQuarterBackLeft", "threeQuarterBackRight"
                                    ]
                                },
                                "subjectFraming": {
                                    "type": "string",
                                    "enum": [
                                        "left", "right", "top", "bottom", "center",
                                        "topLeft", "topRight", "bottomLeft", "bottomRight",
                                        "outerLeft", "outerRight", "outerTop", "outerBottom"
                                    ]
                                },
                            },
                            "required": [
                                "cameraAngle", "shotSize", "subjectView", "subjectFraming"
                            ],
                            "additionalProperties": False
                        },
                    },
                    "required": ["initial", "movement", "final"],
                    "additionalProperties": False
                },
            }
        },
        "required": ["cinematographyPrompts"],
        "additionalProperties": False,
    },
}
