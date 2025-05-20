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


    Example 2:\n    Input: \"While the character moves backward, the camera gradually moves in with a push-in, and as the character starts moving forward, the camera continues to move in while also trucking left, eventually reversing the trucking direction and pushing in while the character continues forward.\"\n    Output:{\"cinematographyPrompts\":[{\"initial\":{\"cameraAngle\":\"eye\",\"shotSize\":\"mediumShot\",\"subjectView\":\"front\",\"subjectFraming\":\"center\"},\"movement\":{\"type\":\"dollyIn\",\"speed\":\"slowToFast\"},\"final\":{\"cameraAngle\":\"eye\",\"shotSize\":\"mediumShot\",\"subjectView\":\"front\",\"subjectFraming\":\"center\"}},{\"initial\":{\"cameraAngle\":\"eye\",\"shotSize\":\"mediumShot\",\"subjectView\":\"front\",\"subjectFraming\":\"center\"},\"movement\":{\"type\":\"truckLeft\",\"speed\":\"constant\"},\"final\":{\"cameraAngle\":\"eye\",\"shotSize\":\"mediumShot\",\"subjectView\":\"front\",\"subjectFraming\":\"center\"}},{\"initial\":{\"cameraAngle\":\"eye\",\"shotSize\":\"mediumShot\",\"subjectView\":\"front\",\"subjectFraming\":\"center\"},\"movement\":{\"type\":\"truckRight\",\"speed\":\"constant\"},\"final\":{\"cameraAngle\":\"eye\",\"shotSize\":\"mediumShot\",\"subjectView\":\"front\",\"subjectFraming\":\"center\"}}]}

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
