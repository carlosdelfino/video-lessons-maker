#!/usr/bin/env python3
"""
SVG Professor Generator
Creates animated SVG professor with talking head and gestures for educational videos.
"""

import os
import tempfile
from pathlib import Path
from typing import Literal, Tuple
import xml.etree.ElementTree as ET

try:
    import cairosvg
except ImportError:
    cairosvg = None


class ProfessorSVG:
    """Generates SVG professor with various expressions and gestures."""
    
    # Gender-specific appearance settings
    GENDER_APPEARANCE = {
        "male": {
            "hair_color": "#4a5568",
            "hair_style": "short",
            "face_color": "#fcd5b8",
            "clothing_color": "#3b82f6",
            "has_facial_hair": True
        },
        "female": {
            "hair_color": "#8b5cf6",
            "hair_style": "long",
            "face_color": "#fcd5b8",
            "clothing_color": "#ec4899",
            "has_facial_hair": False
        }
    }
    
    EXPRESSIONS = {
        "neutral": {
            "mouth": "M -30 30 Q 0 40 30 30",
            "eyebrows": "M -50 -30 Q -30 -35 -10 -30 M 10 -30 Q 30 -35 50 -30",
            "eyes": "circle",
            "head_tilt": 0
        },
        "enthusiastic": {
            "mouth": "M -40 25 Q 0 50 40 25",
            "eyebrows": "M -55 -35 Q -30 -45 -5 -35 M 5 -35 Q 30 -45 55 -35",
            "eyes": "circle",
            "head_tilt": 5
        },
        "thoughtful": {
            "mouth": "M -30 35 Q 0 30 30 35",
            "eyebrows": "M -50 -25 Q -30 -20 -10 -25 M 10 -25 Q 30 -20 50 -25",
            "eyes": "circle",
            "head_tilt": -3
        },
        "questioning": {
            "mouth": "M -30 25 Q 0 20 30 25",
            "eyebrows": "M -50 -35 Q -30 -25 -10 -35 M 10 -35 Q 30 -25 50 -35",
            "eyes": "circle",
            "head_tilt": 8
        }
    }
    
    GESTURES = {
        "none": {
            "left_arm_offset": (-80, 50),
            "right_arm_offset": (80, 50),
            "left_hand": "open",
            "right_hand": "open"
        },
        "pointing": {
            "left_arm_offset": (-60, 30),
            "right_arm_offset": (80, -10),
            "left_hand": "open",
            "right_hand": "pointing"
        },
        "explaining": {
            "left_arm_offset": (-100, 30),
            "right_arm_offset": (100, 30),
            "left_hand": "open",
            "right_hand": "open"
        },
        "nodding": {
            "left_arm_offset": (-80, 50),
            "right_arm_offset": (80, 50),
            "left_hand": "open",
            "right_hand": "open"
        }
    }
    
    def __init__(self, width: int = 800, height: int = 600, gender: Literal["male", "female"] = "female"):
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2
        self.gender = gender
        self.appearance = self.GENDER_APPEARANCE[gender]
    
    def generate_svg(
        self,
        expression: Literal["neutral", "enthusiastic", "thoughtful", "questioning"] = "neutral",
        gesture: Literal["none", "pointing", "explaining", "nodding"] = "none",
        mouth_open: float = 0.0,
        blink: bool = False
    ) -> str:
        """Generate SVG professor with specified expression and gesture."""
        
        expr_data = self.EXPRESSIONS[expression]
        gesture_data = self.GESTURES[gesture]
        
        svg = ET.Element("svg", {
            "width": str(self.width),
            "height": str(self.height),
            "xmlns": "http://www.w3.org/2000/svg"
        })
        
        # Background
        bg = ET.SubElement(svg, "rect", {
            "x": "0", "y": "0",
            "width": str(self.width),
            "height": str(self.height),
            "fill": "#f0f4f8"
        })
        
        # Professor group with head tilt
        tilt = expr_data["head_tilt"]
        professor = ET.SubElement(svg, "g", {
            "transform": f"rotate({tilt}, {self.center_x}, {self.center_y})"
        })
        
        # Legs (behind body)
        self._add_legs(professor)
        
        # Body
        self._add_body(professor)
        
        # Arms with gesture
        self._add_arms(professor, gesture_data)
        
        # Head
        self._add_head(professor, expr_data, mouth_open, blink)
        
        # Convert to string
        return ET.tostring(svg, encoding="unicode")
    
    def _add_body(self, parent: ET.Element):
        """Add professor body."""
        clothing_color = self.appearance["clothing_color"]
        # Torso
        torso = ET.SubElement(parent, "ellipse", {
            "cx": str(self.center_x),
            "cy": str(self.center_y + 100),
            "rx": "120",
            "ry": "150",
            "fill": clothing_color,
            "stroke": self._darken_color(clothing_color),
            "stroke-width": "3"
        })
        
        # Shirt collar
        collar = ET.SubElement(parent, "path", {
            "d": f"M {self.center_x - 60} {self.center_y - 20} L {self.center_x} {self.center_y + 30} L {self.center_x + 60} {self.center_y - 20}",
            "fill": self._lighten_color(clothing_color),
            "stroke": self._darken_color(clothing_color),
            "stroke-width": "2"
        })
    
    def _add_arms(self, parent: ET.Element, gesture_data: dict):
        """Add arms with gesture."""
        clothing_color = self.appearance["clothing_color"]
        shoulder_y = self.center_y + 50
        
        # Left arm
        left_offset = gesture_data["left_arm_offset"]
        left_end_x = self.center_x + left_offset[0]
        left_end_y = shoulder_y + left_offset[1]
        left_arm = ET.SubElement(parent, "path", {
            "d": f"M {self.center_x - 100} {shoulder_y} Q {self.center_x - 50} {shoulder_y + 50} {left_end_x} {left_end_y}",
            "stroke": clothing_color,
            "stroke-width": "25",
            "stroke-linecap": "round",
            "fill": "none"
        })
        
        # Right arm
        right_offset = gesture_data["right_arm_offset"]
        right_end_x = self.center_x + right_offset[0]
        right_end_y = shoulder_y + right_offset[1]
        right_arm = ET.SubElement(parent, "path", {
            "d": f"M {self.center_x + 100} {shoulder_y} Q {self.center_x + 50} {shoulder_y + 50} {right_end_x} {right_end_y}",
            "stroke": clothing_color,
            "stroke-width": "25",
            "stroke-linecap": "round",
            "fill": "none"
        })
        
        # Hands
        self._add_hand(parent, gesture_data["left_hand"], "left", left_end_x, left_end_y)
        self._add_hand(parent, gesture_data["right_hand"], "right", right_end_x, right_end_y)
    
    def _add_legs(self, parent: ET.Element):
        """Add legs to professor."""
        clothing_color = self.appearance["clothing_color"]
        # Left leg
        left_leg = ET.SubElement(parent, "rect", {
            "x": str(self.center_x - 50),
            "y": str(self.center_y + 200),
            "width": "40",
            "height": "120",
            "fill": clothing_color,
            "stroke": self._darken_color(clothing_color),
            "stroke-width": "3"
        })
        
        # Right leg
        right_leg = ET.SubElement(parent, "rect", {
            "x": str(self.center_x + 10),
            "y": str(self.center_y + 200),
            "width": "40",
            "height": "120",
            "fill": clothing_color,
            "stroke": self._darken_color(clothing_color),
            "stroke-width": "3"
        })
        
        # Feet
        for foot_x in [self.center_x - 60, self.center_x]:
            foot = ET.SubElement(parent, "ellipse", {
                "cx": str(foot_x + 20),
                "cy": str(self.center_y + 330),
                "rx": "30",
                "ry": "15",
                "fill": "#1e293b",
                "stroke": "#0f172a",
                "stroke-width": "2"
            })
    
    def _add_hand(self, parent: ET.Element, hand_type: str, side: str, x: float, y: float):
        """Add hand based on type."""
        
        if hand_type == "pointing":
            # Pointing hand (index finger extended)
            hand = ET.SubElement(parent, "ellipse", {
                "cx": str(x),
                "cy": str(y),
                "rx": "15",
                "ry": "20",
                "fill": "#fcd34d",
                "stroke": "#d97706",
                "stroke-width": "2"
            })
            # Finger
            finger = ET.SubElement(parent, "rect", {
                "x": str(x - 5),
                "y": str(y - 25),
                "width": "10",
                "height": "20",
                "fill": "#fcd34d",
                "stroke": "#d97706",
                "stroke-width": "2"
            })
        else:
            # Open hand
            hand = ET.SubElement(parent, "ellipse", {
                "cx": str(x),
                "cy": str(y),
                "rx": "25",
                "ry": "30",
                "fill": "#fcd34d",
                "stroke": "#d97706",
                "stroke-width": "2"
            })
    
    def _add_head(self, parent: ET.Element, expr_data: dict, mouth_open: float, blink: bool):
        """Add head with expression."""
        head_y = self.center_y - 50
        face_color = self.appearance["face_color"]
        hair_color = self.appearance["hair_color"]
        hair_style = self.appearance["hair_style"]
        
        # Neck
        neck = ET.SubElement(parent, "rect", {
            "x": str(self.center_x - 30),
            "y": str(head_y + 60),
            "width": "60",
            "height": "40",
            "fill": face_color,
            "stroke": self._darken_color(face_color),
            "stroke-width": "2"
        })
        
        # Face
        face = ET.SubElement(parent, "ellipse", {
            "cx": str(self.center_x),
            "cy": str(head_y),
            "rx": "90",
            "ry": "100",
            "fill": face_color,
            "stroke": self._darken_color(face_color),
            "stroke-width": "3"
        })
        
        # Hair based on gender and style
        self._add_hair(parent, head_y, hair_color, hair_style)
        
        # Facial hair for male
        if self.appearance["has_facial_hair"]:
            self._add_facial_hair(parent, head_y)
        
        # Eyebrows (relative to head center)
        for brow_offset in [-30, 30]:
            brow_x = self.center_x + brow_offset
            brow = ET.SubElement(parent, "path", {
                "d": f"M {brow_x - 20} {head_y - 30} Q {brow_x} {head_y - 35} {brow_x + 20} {head_y - 30}",
                "stroke": hair_color,
                "stroke-width": "4",
                "fill": "none"
            })
        
        # Eyes
        eye_color = "white" if not blink else hair_color
        for eye_offset in [-30, 30]:
            eye_x = self.center_x + eye_offset
            eye = ET.SubElement(parent, expr_data["eyes"], {
                "cx": str(eye_x),
                "cy": str(head_y - 10),
                "r": "15" if not blink else "3",
                "fill": eye_color,
                "stroke": hair_color,
                "stroke-width": "2"
            })
            if not blink:
                # Pupil
                pupil = ET.SubElement(parent, "circle", {
                    "cx": str(eye_x + 3),
                    "cy": str(head_y - 10),
                    "r": "7",
                    "fill": "#4a3728"
                })
        
        # Nose
        nose = ET.SubElement(parent, "path", {
            "d": f"M {self.center_x} {head_y + 20} L {self.center_x - 10} {head_y + 40} L {self.center_x + 10} {head_y + 40}",
            "stroke": self._darken_color(face_color),
            "stroke-width": "2",
            "fill": "none"
        })
        
        # Mouth (animated based on mouth_open)
        mouth_y = head_y + 60
        if mouth_open > 0:
            # Open mouth
            mouth_height = 10 + mouth_open * 20
            mouth = ET.SubElement(parent, "ellipse", {
                "cx": str(self.center_x),
                "cy": str(mouth_y + mouth_height / 2),
                "rx": "30",
                "ry": str(mouth_height),
                "fill": "#991b1b",
                "stroke": "#7f1d1d",
                "stroke-width": "2"
            })
        else:
            # Closed mouth (relative to head center)
            ET.SubElement(parent, "path", {
                "d": f"M {self.center_x - 30} {mouth_y} Q {self.center_x} {mouth_y + 10} {self.center_x + 30} {mouth_y}",
                "stroke": "#7f1d1d",
                "stroke-width": "3",
                "fill": "none"
            })
        
        # Glasses (professional look)
        for glass_offset in [-30, 30]:
            glass_x = self.center_x + glass_offset
            glass = ET.SubElement(parent, "circle", {
                "cx": str(glass_x),
                "cy": str(head_y - 10),
                "r": "25",
                "fill": "none",
                "stroke": "#1e293b",
                "stroke-width": "3"
            })
        
        # Glasses bridge
        bridge = ET.SubElement(parent, "line", {
            "x1": str(self.center_x - 10),
            "y1": str(head_y - 10),
            "x2": str(self.center_x + 10),
            "y2": str(head_y - 10),
            "stroke": "#1e293b",
            "stroke-width": "3"
        })
    
    def _add_hair(self, parent: ET.Element, head_y: int, hair_color: str, hair_style: str):
        """Add hair based on gender and style."""
        if hair_style == "short":
            # Short hair (male)
            hair = ET.SubElement(parent, "ellipse", {
                "cx": str(self.center_x),
                "cy": str(head_y - 70),
                "rx": "85",
                "ry": "50",
                "fill": hair_color,
                "stroke": self._darken_color(hair_color),
                "stroke-width": "2"
            })
        else:
            # Long hair (female)
            hair = ET.SubElement(parent, "path", {
                "d": f"M {self.center_x - 90} {head_y - 20} Q {self.center_x - 100} {head_y + 50} {self.center_x - 80} {head_y + 120} L {self.center_x + 80} {head_y + 120} Q {self.center_x + 100} {head_y + 50} {self.center_x + 90} {head_y - 20} Q {self.center_x} {head_y - 100} {self.center_x - 90} {head_y - 20}",
                "fill": hair_color,
                "stroke": self._darken_color(hair_color),
                "stroke-width": "2"
            })
    
    def _add_facial_hair(self, parent: ET.Element, head_y: int):
        """Add facial hair for male professor."""
        # Mustache
        mustache = ET.SubElement(parent, "path", {
            "d": f"M {self.center_x - 30} {head_y + 35} Q {self.center_x} {head_y + 45} {self.center_x + 30} {head_y + 35}",
            "stroke": "#4a3728",
            "stroke-width": "4",
            "fill": "none"
        })
    
    def _darken_color(self, hex_color: str) -> str:
        """Darken a hex color by 20%."""
        # Simple darkening by reducing each component
        hex_color = hex_color.lstrip('#')
        r = max(0, int(hex_color[0:2], 16) - 40)
        g = max(0, int(hex_color[2:4], 16) - 40)
        b = max(0, int(hex_color[4:6], 16) - 40)
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def _lighten_color(self, hex_color: str) -> str:
        """Lighten a hex color by 20%."""
        # Simple lightening by increasing each component
        hex_color = hex_color.lstrip('#')
        r = min(255, int(hex_color[0:2], 16) + 40)
        g = min(255, int(hex_color[2:4], 16) + 40)
        b = min(255, int(hex_color[4:6], 16) + 40)
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def save_svg_as_png(self, svg_content: str, output_path: Path) -> bool:
        """Convert SVG to PNG using cairosvg."""
        if cairosvg is None:
            raise ImportError("cairosvg is required for SVG to PNG conversion. Install with: pip install cairosvg")
        
        try:
            # Convert SVG to PNG
            cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), write_to=str(output_path))
            return True
        except Exception as e:
            print(f"Error converting SVG to PNG: {e}")
            return False


def generate_professor_frame(
    expression: str = "neutral",
    gesture: str = "none",
    mouth_open: float = 0.0,
    blink: bool = False,
    output_path: Path = None
) -> str:
    """Generate a single professor frame and optionally save as PNG."""
    professor = ProfessorSVG()
    svg_content = professor.generate_svg(expression, gesture, mouth_open, blink)
    
    if output_path:
        professor.save_svg_as_png(svg_content, output_path)
    
    return svg_content
