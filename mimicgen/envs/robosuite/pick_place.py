# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Slight variant of pick place task.
"""
import numpy as np
from robosuite.environments.manipulation.pick_place import PickPlace
from mimicgen.models.robosuite.objects.xml_objects import MujocoXMLObject
class PickPlace_D0(PickPlace):
    """
    Slightly easier task where we limit z-rotation to 0 to 90 degrees for all object initializations (instead of full 360).
    """
    def __init__(self, **kwargs):
        assert "z_rotation" not in kwargs
        super().__init__(
            z_rotation=(0., np.pi / 2.),
            **kwargs,
        )

class PickPlace_D1(PickPlace):
    def __init__(self, obj_xml_path, **kwargs):
        assert "z_rotation" not in kwargs
        self.obj_xml_path = obj_xml_path
        super().__init__(
            z_rotation=(np.pi / 2., np.pi),
            **kwargs,
        )
        self.object_to_id = {'milk': 0}
        self.obj_names = ['milk']
        print("Using object from:", self.obj_xml_path)
    def _construct_objects(self):
        """
        Function that can be overriden by subclasses to load different objects.
        """
        self.objects = []
        # obj = BlenderObject(name="target", mjcf_path=self.obj_xml_path, scale=1.0, density=500)
        obj = MujocoXMLObject(
            self.obj_xml_path,
            name="milk",
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )
        # obj = BlenderObject(name="target", mjcf_path=self.obj_xml_path, scale=1.0, density=500)
        self.objects.append(obj)
    def _construct_visual_objects(self):
        """
        Function that can be overriden by subclasses to load different objects.
        """
        self.visual_objects = []
        visual_obj = MujocoXMLObject(
            self.obj_xml_path.replace('physical.xml', 'visual.xml'),
            name="visual",
            joints=None,
            obj_type="visual",
            duplicate_collision_geoms=True
        )
        self.visual_objects.append(visual_obj)