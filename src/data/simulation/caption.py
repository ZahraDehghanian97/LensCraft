enum_descriptions = {
    "CameraVerticalAngle": {
        "low": "from a low angle",
        "eye": "from an eye-level angle",
        "high": "from a high angle",
        "overhead": "from an overhead angle",
        "birdsEye": "from a bird's-eye angle"
    },
    "ShotSize": {
        "extremeCloseUp": "in an extreme close-up",
        "closeUp": "in a close-up",
        "mediumCloseUp": "in a medium close-up",
        "mediumShot": "in a medium shot",
        "fullShot": "in a full shot",
        "longShot": "in a long shot",
        "veryLongShot": "in a very long shot",
        "extremeLongShot": "in an extreme long shot"
    },
    "Scale": {
        "small": "with small intensity",
        "medium": "with medium intensity",
        "large": "with large intensity",
        "full": "with full intensity"
    },
    "MovementEasing": {
        "linear": "with linear movement",
        "easeInSine": "with gradual acceleration using sine function",
        "easeOutSine": "with gradual deceleration using sine function",
        "easeInOutSine": "with gradual acceleration and deceleration using sine function",
        "easeInQuad": "with quadratic acceleration",
        "easeOutQuad": "with quadratic deceleration",
        "easeInOutQuad": "with quadratic acceleration and deceleration",
        "easeInCubic": "with cubic acceleration",
        "easeOutCubic": "with cubic deceleration",
        "easeInOutCubic": "with cubic acceleration and deceleration",
        "easeInQuart": "with quartic acceleration",
        "easeOutQuart": "with quartic deceleration",
        "easeInOutQuart": "with quartic acceleration and deceleration",
        "easeInQuint": "with quintic acceleration",
        "easeOutQuint": "with quintic deceleration",
        "easeInOutQuint": "with quintic acceleration and deceleration",
        "easeInExpo": "with exponential acceleration",
        "easeOutExpo": "with exponential deceleration",
        "easeInOutExpo": "with exponential acceleration and deceleration",
        "easeInCirc": "with circular acceleration",
        "easeOutCirc": "with circular deceleration",
        "easeInOutCirc": "with circular acceleration and deceleration",
        "easeInBack": "with overshooting acceleration",
        "easeOutBack": "with overshooting deceleration",
        "easeInOutBack": "with overshooting acceleration and deceleration",
        "easeInElastic": "with elastic acceleration",
        "easeOutElastic": "with elastic deceleration",
        "easeInOutElastic": "with elastic acceleration and deceleration",
        "easeInBounce": "with bouncing acceleration",
        "easeOutBounce": "with bouncing deceleration",
        "easeInOutBounce": "with bouncing acceleration and deceleration",
        "handHeld": "with handheld camera movement",
        "anticipation": "with anticipatory movement",
        "smooth": "with smooth movement"
    },
    "SubjectView": {
        "front": "from the front",
        "back": "from the back",
        "left": "from the left side",
        "right": "from the right side",
        "threeQuarterFrontLeft": "from the front-left three-quarter view",
        "threeQuarterFrontRight": "from the front-right three-quarter view",
        "threeQuarterBackLeft": "from the back-left three-quarter view",
        "threeQuarterBackRight": "from the back-right three-quarter view"
    },
    "SubjectInFramePosition": {
        "left": "positioned on the left",
        "right": "positioned on the right",
        "top": "positioned at the top",
        "bottom": "positioned at the bottom",
        "center": "positioned in the center",
        "topLeft": "positioned in the top-left",
        "topRight": "positioned in the top-right",
        "bottomLeft": "positioned in the bottom-left",
        "bottomRight": "positioned in the bottom-right",
        "outerLeft": "positioned on the far left",
        "outerRight": "positioned on the far right",
        "outerTop": "positioned at the very top",
        "outerBottom": "positioned at the very bottom"
    },
    "Direction": {
        "left": "towards the left",
        "right": "towards the right",
        "up": "upwards",
        "down": "downwards",
        "forward": "forwards",
        "backward": "backwards"
    },
    "MovementMode": {
        "transition": "using transitional movement",
        "rotation": "using rotational movement",
        "arc": "using arc movement"
    },
    "CameraMovementType": {
        "static": "remaining stationary",
        "follow": "following the subject",
        "track": "tracking alongside the subject",
        "dollyIn": "moving closer to the subject",
        "dollyOut": "moving away from the subject",
        "panLeft": "panning to the left",
        "panRight": "panning to the right",
        "tiltUp": "tilting upward",
        "tiltDown": "tilting downward",
        "truckLeft": "moving laterally left",
        "truckRight": "moving laterally right",
        "pedestalUp": "moving straight up",
        "pedestalDown": "moving straight down",
        "arcLeft": "moving in a leftward arc",
        "arcRight": "moving in a rightward arc",
        "craneUp": "craning upward",
        "craneDown": "craning downward",
        "dollyOutZoomIn": "moving back while zooming in",
        "dollyInZoomOut": "moving forward while zooming out",
        "dutchLeft": "tilting left on the camera axis",
        "dutchRight": "tilting right on the camera axis"
    },
    "MovementSpeed": {
        "slowToFast": "accelerating from slow to fast",
        "fastToSlow": "decelerating from fast to slow",
        "constant": "at a constant speed",
        "smoothStartStop": "smoothly starting and stopping"
    },
    "SetupKind": {
        "init": "from the initial setup",
        "end": "towards the final setup"
    }
}



def extract_text_prompt(cin_params, subject_movement):
    camera_movement_descriptions = {
        "static": "remains stationary",
        "follow": "follows the subject",
        "track": "tracks alongside the subject",
        "dollyIn": "pushes in",
        "dollyOut": "pulls out", 
        "panLeft": "pans to the left",
        "panRight": "pans to the right",
        "tiltUp": "tilts upward",
        "tiltDown": "tilts downward",
        "truckLeft": "trucks left",
        "truckRight": "trucks right",
        "pedestalUp": "moves straight up with a pedestal movement",
        "pedestalDown": "moves straight down with a pedestal movement",
        "arcLeft": "moves in a leftward arc",
        "arcRight": "moves in a rightward arc", 
        "craneUp": "cranes upward",
        "craneDown": "cranes downward",
        "dollyOutZoomIn": "pulls out while zooming in",
        "dollyInZoomOut": "pushes in while zooming out",
        "dutchLeft": "tilts left on the camera axis",
        "dutchRight": "tilts right on the camera axis"
    }
    
    subject_movement_descriptions = {
        "static": "remains stationary",
        "circular": "moves in a circular path",
        "linear": "moves back and forth in a straight line",
        "zigzag": "moves in a zigzag pattern",
        "spiral": "moves in a spiral pattern",
        "figureEight": "moves in a figure-eight pattern",
        "wave": "moves in a wave-like sinusoidal pattern",
        "pendulum": "swings back and forth like a pendulum",
        "orbital": "orbits around a center point",
        "bounce": "bounces while moving forward"
    }
    
    detailed_subject_descriptions = {
        "static": "remains stationary throughout the entire sequence",
        "circular": "moves in a continuous circular motion",
        "linear": "travels back and forth along a straight path",
        "zigzag": "follows a sharp zigzag trajectory",
        "spiral": "traces an expanding spiral path",
        "figureEight": "traces a figure-eight pattern",
        "wave": "follows a smooth wave-like path",
        "pendulum": "swings in a pendulum motion",
        "orbital": "orbits in a tilted elliptical path",
        "bounce": "bounces rhythmically while progressing forward"
    }
    
    shot_size_descriptions = {
        "extremeCloseUp": "extreme close-up",
        "closeUp": "close-up", 
        "mediumCloseUp": "medium close-up",
        "mediumShot": "medium shot",
        "fullShot": "full shot",
        "longShot": "wide shot",
        "veryLongShot": "very wide shot", 
        "extremeLongShot": "extreme wide shot"
    }
    
    camera_movement_type = cin_params.get('movement', {}).get('type')
    movement_speed = cin_params.get('movement', {}).get('speed')
    
    initial_shot = cin_params.get('initial', {}).get('shotSize')
    final_shot = cin_params.get('final', {}).get('shotSize')
    
    initial_angle = cin_params.get('initial', {}).get('cameraAngle')
    final_angle = cin_params.get('final', {}).get('cameraAngle')
    
    initial_framing = cin_params.get('initial', {}).get('subjectFraming')
    final_framing = cin_params.get('final', {}).get('subjectFraming')
    
    camera_desc = camera_movement_descriptions.get(camera_movement_type, "moves")
    subject_desc = subject_movement_descriptions.get(subject_movement, "remains stationary")
    detailed_subject_desc = detailed_subject_descriptions.get(subject_movement, "remains stationary")
    
    is_subject_stationary = subject_movement in ["static", None] or subject_movement == "static"
    
    shot_change_desc = ""
    if initial_shot != final_shot and initial_shot and final_shot:
        initial_shot_desc = shot_size_descriptions.get(initial_shot, initial_shot)
        final_shot_desc = shot_size_descriptions.get(final_shot, final_shot)
        shot_change_desc = f"transitioning from a {initial_shot_desc} to a {final_shot_desc}"
    
    angle_change_desc = ""
    if initial_angle != final_angle and initial_angle and final_angle:
        angle_map = {
            "low": "low angle",
            "eye": "eye level", 
            "high": "high angle",
            "overhead": "overhead angle",
            "birdsEye": "bird's-eye view"
        }
        initial_angle_desc = angle_map.get(initial_angle, initial_angle)
        final_angle_desc = angle_map.get(final_angle, final_angle)
        angle_change_desc = f"changing from {initial_angle_desc} to {final_angle_desc}"
    
    if camera_movement_type == "static":
        if is_subject_stationary:
            prompt = "Both the camera and character remain stationary throughout the sequence."
        else:
            prompt = f"While the camera remains stationary, the character {detailed_subject_desc}."
    
    elif is_subject_stationary:
        base_desc = f"The camera {camera_desc}"
        
        changes = []
        if shot_change_desc:
            changes.append(shot_change_desc)
        if angle_change_desc:
            changes.append(angle_change_desc)
        
        if changes:
            base_desc += f", {' and '.join(changes)},"
        
        prompt = f"{base_desc} while the character remains stationary throughout the entire sequence."
    
    else:
        subject_clause = f"the character {subject_desc}"
        camera_clause = f"the camera {camera_desc}"
        
        if shot_change_desc or angle_change_desc:
            changes = []
            if shot_change_desc:
                changes.append(shot_change_desc)
            if angle_change_desc:
                changes.append(angle_change_desc)
            
            change_desc = ' and '.join(changes)
            camera_clause += f", {change_desc}"
        
        if camera_movement_type in ["follow", "track"]:
            if subject_movement in ["circular", "linear", "spiral"]:
                prompt = f"The camera {camera_desc} as {subject_clause}."
            elif subject_movement in ["zigzag", "wave"]:
                prompt = f"The camera {camera_desc}, matching the rhythm as {subject_clause}."
            elif subject_movement == "bounce":
                prompt = f"The camera {camera_desc}, maintaining focus as {subject_clause}."
            else:
                prompt = f"The camera {camera_desc} while {subject_clause}."
        
        elif camera_movement_type in ["truckLeft", "truckRight"]:
            if subject_movement in ["linear", "wave", "zigzag"]:
                if ("Left" in camera_movement_type and "left" in subject_movement) or \
                   ("Right" in camera_movement_type and "right" in subject_movement):
                    prompt = f"The camera {camera_desc} in sync with {subject_clause}."
                else:
                    prompt = f"The camera {camera_desc} while {subject_clause}."
            elif subject_movement == "circular":
                prompt = f"The camera {camera_desc} as {subject_clause}, creating dynamic framing."
            elif subject_movement in ["pendulum", "orbital"]:
                prompt = f"The camera {camera_desc} alongside {subject_clause}."
            else:
                prompt = f"The camera {camera_desc} while {subject_clause}."
        
        elif camera_movement_type in ["dollyIn", "dollyOut"]:
            if subject_movement in ["spiral", "circular"]:
                prompt = f"As {subject_clause}, {camera_clause}, creating a dynamic relationship."
            elif subject_movement == "linear":
                prompt = f"While {subject_clause}, {camera_clause}."
            elif subject_movement in ["bounce", "wave"]:
                prompt = f"The camera {camera_desc} as {subject_clause}, emphasizing the motion."
            else:
                prompt = f"As {subject_clause}, {camera_clause}."
        
        elif camera_movement_type in ["panLeft", "panRight"]:
            if subject_movement in ["linear", "zigzag"]:
                pan_direction = "left" if "Left" in camera_movement_type else "right"
                if pan_direction in subject_movement.lower():
                    prompt = f"The camera {camera_desc} in tandem with {subject_clause}."
                else:
                    prompt = f"The camera {camera_desc} while {subject_clause}."
            elif subject_movement == "circular":
                prompt = f"The camera {camera_desc} to follow {subject_clause}."
            else:
                prompt = f"As {subject_clause}, {camera_clause}."
        
        elif camera_movement_type in ["arcLeft", "arcRight", "craneUp", "craneDown"]:
            if subject_movement in ["orbital", "spiral"]:
                prompt = f"The camera {camera_desc} complementing {subject_clause}."
            elif subject_movement == "figureEight":
                prompt = f"The camera {camera_desc} as {subject_clause}, creating complex choreography."
            else:
                prompt = f"While {subject_clause}, {camera_clause}."
        
        else:
            if subject_movement in ["circular", "spiral", "figureEight"]:
                prompt = f"As {subject_clause}, {camera_clause}."
            elif subject_movement in ["linear", "zigzag", "wave"]:
                prompt = f"While {subject_clause}, {camera_clause}."
            elif subject_movement in ["bounce", "pendulum"]:
                prompt = f"The camera {camera_desc} as {subject_clause}."
            else:
                prompt = f"While {subject_clause}, {camera_clause}."
    
    if movement_speed and movement_speed != "constant":
        speed_descriptions = {
            "slowToFast": "gradually accelerating",
            "fastToSlow": "gradually decelerating", 
            "smoothStartStop": "with smooth acceleration and deceleration"
        }
        speed_desc = speed_descriptions.get(movement_speed)
        if speed_desc:
            prompt = prompt.rstrip('.') + f", {speed_desc}."
    
    if initial_framing != final_framing and initial_framing and final_framing:
        framing_map = {
            "left": "left side",
            "right": "right side", 
            "center": "center",
            "topLeft": "top-left",
            "topRight": "top-right",
            "bottomLeft": "bottom-left",
            "bottomRight": "bottom-right",
            "outerLeft": "far left",
            "outerRight": "far right",
            "outerTop": "very top",
            "outerBottom": "very bottom"
        }
        initial_framing_desc = framing_map.get(initial_framing, initial_framing)
        final_framing_desc = framing_map.get(final_framing, final_framing)
        
        framing_change = f"repositioning from {initial_framing_desc} to {final_framing_desc} of the frame"
        prompt = prompt.rstrip('.') + f", {framing_change}."
    
    return prompt
