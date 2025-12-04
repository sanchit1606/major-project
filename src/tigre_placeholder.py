"""
Minimal TIGRE placeholder module for SAX-NeRF compatibility.
This provides the basic Geometry class needed without requiring the full TIGRE installation.
"""

import numpy as np

class Geometry:
    """
    Basic geometry class for CT reconstruction.
    This is a simplified version of the TIGRE Geometry class.
    """
    def __init__(self):
        pass

# Create a mock tigre module structure
class MockTigreModule:
    class utilities:
        class geometry:
            Geometry = Geometry

# Create the tigre module
tigre = MockTigreModule()

# Make it available for import
__all__ = ['tigre', 'Geometry']
