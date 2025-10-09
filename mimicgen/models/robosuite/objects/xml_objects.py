# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Some objects based on MJCF models.
"""
import os
import time
import xml.etree.ElementTree as ET
import numpy as np

from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import string_to_array, array_to_string

import mimicgen

XML_ASSETS_BASE_PATH = os.path.join(mimicgen.__path__[0], "models/robosuite/assets")


class BlenderObject(MujocoXMLObject):
    """
    Blender object with support for changing the scaling 
    """
    def __init__(
        self,
        name,
        mjcf_path,
        scale=1.0,
        solimp=(0.998, 0.998, 0.001),
        solref=(0.001, 1),
        density=100,
        friction=(0.95, 0.3, 0.1),
        margin=None,
        rgba=None,
    ):
        # get scale in x, y, z
        if isinstance(scale, float):
            scale = [scale, scale, scale]
        elif isinstance(scale, tuple) or isinstance(scale, list):
            assert len(scale) == 3
            scale = tuple(scale)
        else:
            raise Exception("got invalid scale: {}".format(scale))
        scale = np.array(scale)

        self.solimp = solimp
        self.solref = solref
        self.density = density
        self.friction = friction
        self.margin = margin

        self.rgba = rgba

        # read default xml
        xml_path = mjcf_path
        folder = os.path.dirname(xml_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # modify mesh scales
        asset = root.find("asset")
        meshes = asset.findall("mesh")
        for mesh in meshes:
            # if a scale already exists, multiply the scales
            scale_to_set = scale
            existing_scale = mesh.get("scale")
            if existing_scale is not None:
                scale_to_set = string_to_array(existing_scale) * scale
            mesh.set("scale", array_to_string(scale_to_set))

        # modify sites for collision (assumes we can just scale up the locations - may or may not work)
        for n in ["bottom_site", "top_site", "horizontal_radius_site"]:
            site = root.find("worldbody/body/site[@name='{}']".format(n))
            pos = string_to_array(site.get("pos"))
            pos = scale * pos
            site.set("pos", array_to_string(pos))

        # write modified xml (and make sure to postprocess any paths just in case)
        xml_str = ET.tostring(root, encoding="utf8").decode("utf8")
        # xml_str = postprocess_model_xml(xml_str)
        time_str = str(time.time()).replace(".", "_")
        new_xml_path = os.path.join(folder, "{}_{}.xml".format(time_str, os.getpid()))
        f = open(new_xml_path, "w")
        f.write(xml_str)
        f.close()
        # print(f"Write to {new_xml_path}")

        # initialize object with new xml we wrote
        super().__init__(
            fname=new_xml_path,
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )

        # clean up xml - we don't need it anymore
        if os.path.exists(new_xml_path):
            os.remove(new_xml_path)

    def _get_geoms(self, root, _parent=None):
        """
        Helper function to recursively search through element tree starting at @root and returns
        a list of (parent, child) tuples where the child is a geom element

        Args:
            root (ET.Element): Root of xml element tree to start recursively searching through
            _parent (ET.Element): Parent of the root element tree. Should not be used externally; only set
                during the recursive call

        Returns:
            list: array of (parent, child) tuples where the child element is a geom type
        """
        geom_pairs = super(BlenderObject, self)._get_geoms(root=root, _parent=_parent)

        # modify geoms according to the attributes
        for i, (parent, element) in enumerate(geom_pairs):
            element.set("solref", array_to_string(self.solref))
            element.set("solimp", array_to_string(self.solimp))
            element.set("density", str(self.density))
            element.set("friction", array_to_string(self.friction))
            if self.margin is not None:
                element.set("margin", str(self.margin))

            if (self.rgba is not None) and (element.get("group") == "1"):
                element.set("rgba", array_to_string(self.rgba))
        
        return geom_pairs


class CoffeeMachinePodObject(MujocoXMLObject):
    """
    Coffee pod object (used in Coffee task).
    """
    def __init__(self, name):
        super().__init__(os.path.join(XML_ASSETS_BASE_PATH, "objects/coffee_pod.xml"),
                         # name=name, joints=[dict(type="free", damping="0.0005")],
                         name=name, joints=[dict(type="free")],
                         obj_type="all", duplicate_collision_geoms=True)


class CoffeeMachineBodyObject(MujocoXMLObject):
    """
    Coffee machine body piece (used in Coffee task). Note that the piece is rigid (no joint is added).
    """
    def __init__(self, name):
        super().__init__(os.path.join(XML_ASSETS_BASE_PATH, "objects/coffee_body.xml"),
                         name=name, joints=None,
                         obj_type="all", duplicate_collision_geoms=True)


class CoffeeMachineLidObject(MujocoXMLObject):
    """
    Coffee machine lid piece (used in Coffee task).
    """
    def __init__(self, name):
        super().__init__(os.path.join(XML_ASSETS_BASE_PATH, "objects/coffee_lid.xml"),
                         name=name, joints=None,
                         obj_type="all", duplicate_collision_geoms=True)


class CoffeeMachineBaseObject(MujocoXMLObject):
    """
    Coffee machine base piece (used in Coffee task). Note that the piece is rigid (no joint is added).
    """
    def __init__(self, name):
        super().__init__(os.path.join(XML_ASSETS_BASE_PATH, "objects/coffee_base.xml"),
                         name=name, joints=None,
                         obj_type="all", duplicate_collision_geoms=True)


class DrawerObject(MujocoXMLObject):
    """
    Custom version of cabinet object that differs from BUDs. It has manually specified top, bottom, and horizontal sites,
    a slightly different material for the handle, and changed the group for the cabinet geoms from 1 to 0 because
    robosuite v1.4 enforces that geom groups with 0 participate in physics and 1 do not.
    """
    def __init__(
            self,
            name,
            joints=None):
        path_to_cabinet_xml = os.path.join(XML_ASSETS_BASE_PATH, "objects/drawer.xml")
        super().__init__(path_to_cabinet_xml,
                         name=name, joints=None, obj_type="all", duplicate_collision_geoms=True)

    # NOTE: had to manually set these to get placement sampler working okay
    @property
    def bottom_offset(self):
        return np.array([0, 0, -0.065])

    @property
    def top_offset(self):
        return np.array([0, 0, 0.065])
        
    @property
    def horizontal_radius(self):
        return 0.15


class LongDrawerObject(MujocoXMLObject):
    """
    Drawer that has longer platform for easier grasping of objects inside.
    """
    def __init__(
            self,
            name,
            joints=None):
        # our custom cabinet xml - has some longer geoms for the drawer platform
        path_to_cabinet_xml = os.path.join(XML_ASSETS_BASE_PATH, "objects/drawer_long.xml")
        super().__init__(path_to_cabinet_xml,
                         name=name, joints=None, obj_type="all", duplicate_collision_geoms=True)

    # NOTE: had to manually set these to get placement sampler working okay
    @property
    def bottom_offset(self):
        return np.array([0, 0, -0.065])

    @property
    def top_offset(self):
        return np.array([0, 0, 0.065])
        
    @property
    def horizontal_radius(self):
        return 0.15
class PickPlaceObject(MujocoXMLObject):
    """
    Custom version of pick and place object.
    """
    def __init__(
            self,
            name,
            path_to_xml,
            joints=None):
        super().__init__(path_to_xml,
                         name=name, joints=None, obj_type="all", duplicate_collision_geoms=True)

def array_to_string(a):
    a = np.asarray(a).reshape(-1)
    return " ".join(str(float(x)) for x in a)

def text_contains_any(s, tokens):
    s = (s or "").lower()
    return any(t in s for t in tokens)

def parse_semantic_file(semantic_path):
    """
    Parse semantic.txt with lines:
      <link_name> <category> <role>
    Returns dict: link_name -> (category, role)
    """
    mapping = {}
    if not semantic_path or not os.path.exists(semantic_path):
        return mapping
    with open(semantic_path, 'r') as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            parts = ln.split()
            if len(parts) < 3:
                # tolerate missing tokens by skipping line
                continue
            link_name = parts[0]
            category = parts[1]
            role = parts[2]
            mapping[link_name] = (category, role)
    return mapping

class ArticulatedObject(MujocoXMLObject):
    """
    Detect base_body and rotation_door using semantic.txt when present.
    If semantic file not present or incomplete, fall back to heuristics.
    """

    def __init__(self, xml_path, name=None, semantic_path=None, friction=None, damping=None):
        """
        xml_path: MJCF/XML file produced earlier
        semantic_path: optional path to semantic.txt; if None, will look for <xml_dir>/semantic.txt
        friction/damping: optional overrides for the rotation joint
        """
        super().__init__(xml_path, name=name, joints=None, obj_type="all", duplicate_collision_geoms=False)

        # store parsed worldbody info
        self.body_elems = {}
        self.parent_of = {}
        self.joint_records = []

        # semantic mapping: link -> (category, role)
        if semantic_path is None:
            xml_dir = os.path.dirname(os.path.abspath(xml_path))
            semantic_path = os.path.join(xml_dir, "semantic.txt")
        self.semantics = parse_semantic_file(semantic_path)

        # selected fields
        self.base_body = None
        self.rotation_body = None
        self.rotation_joint = None

        # perform analysis
        self._analyze_worldbody()

        # try semantic-based selection first
        selected = self._choose_by_semantics()
        if not selected:
            # fallback to heuristics
            self._choose_base_and_rotation_heuristic()

        # apply friction/damping overrides if provided
        self.friction = friction
        self.damping = damping
        if self.friction is not None:
            self._set_rotation_friction(self.friction)
        if self.damping is not None:
            self._set_rotation_damping(self.damping)

    # -------------------------
    # XML analysis
    # -------------------------
    def _analyze_worldbody(self):
        wb = getattr(self, "worldbody", None)
        if wb is None:
            xml_root = getattr(self, "xml", None)
            if xml_root is not None:
                wb = xml_root.find("worldbody")
        if wb is None:
            raise RuntimeError("Could not find worldbody element in XML")

        # DFS collect bodies and parent relationships
        self.body_elems = {}
        self.parent_of = {}
        stack = [(wb, None)]
        while stack:
            elem, parent_name = stack.pop()
            for child in list(elem):
                if child.tag == "body":
                    body_name = child.get("name")
                    if body_name is None:
                        body_name = f"<anon_body_{len(self.body_elems)}>"
                        child.set("name", body_name)
                    self.body_elems[body_name] = child
                    self.parent_of[body_name] = parent_name
                    stack.append((child, body_name))
                else:
                    # continue searching deeper (some XMLs nest non-body tags)
                    stack.append((child, parent_name))

        # collect joint elements (joint elements are typically children of the child body)
        self.joint_records = []
        for body_name, body_elem in self.body_elems.items():
            for joint_elem in body_elem.findall("joint"):
                jname = joint_elem.get("name") or f"<anon_joint_{len(self.joint_records)}>"
                jtype = (joint_elem.get("type") or "").lower()
                axis = None
                if joint_elem.get("axis"):
                    try:
                        axis = tuple(float(x) for x in joint_elem.get("axis").strip().split())
                    except Exception:
                        axis = None
                range_attr = joint_elem.get("range")
                rng = None
                if range_attr:
                    try:
                        rng = tuple(float(x) for x in range_attr.strip().split())
                    except Exception:
                        rng = None
                rec = {"elem": joint_elem, "name": jname, "type": jtype, "parent": self.parent_of.get(body_name), "child": body_name, "axis": axis, "range": rng}
                self.joint_records.append(rec)

    # -------------------------
    # semantic selection
    # -------------------------
    def _choose_by_semantics(self):
        """
        Use parsed semantic.txt to select rotation_body and base_body.
        Returns True if successful, False otherwise.
        """
        if not self.semantics:
            return False

        # Find rotation_door entry
        rotation_candidates = [ln for ln, (_, role) in self.semantics.items() if role == "rotation_door"]
        if not rotation_candidates:
            # no rotation door specified
            return False
        # assume exactly one; pick first otherwise
        rotation_link = rotation_candidates[0]
        if rotation_link not in self.body_elems:
            # semantic mentions a link that is not in XML; failover to heuristics
            return False

        # find base candidates: roles that end with '_body'
        base_candidates = [ln for ln, (_, role) in self.semantics.items() if role.endswith("_body")]
        # choose one that is parent of rotation_link if possible
        chosen_base = None
        if base_candidates:
            # prefer the one equal to the parent of rotation in world tree
            parent_of_rotation = self.parent_of.get(rotation_link)
            if parent_of_rotation and parent_of_rotation in base_candidates:
                chosen_base = parent_of_rotation
            else:
                # else pick first base candidate that exists in body_elems
                for b in base_candidates:
                    if b in self.body_elems:
                        chosen_base = b
                        break

        # if no candidate found in semantics that exists in XML, fallback to heuristics
        if chosen_base is None:
            # try pick the parent body in the XML as base
            parent = self.parent_of.get(rotation_link)
            if parent:
                chosen_base = parent
            else:
                # pick some top-level body
                top_level = [b for b, p in self.parent_of.items() if p is None]
                chosen_base = top_level[0] if top_level else next(iter(self.body_elems.keys()))

        # now set fields
        self.rotation_body = rotation_link
        self.base_body = chosen_base

        # find rotation joint whose child equals rotation_body
        found_joint = None
        for rec in self.joint_records:
            if rec["child"] == self.rotation_body:
                found_joint = rec
                break
        if found_joint:
            self.rotation_joint = found_joint["name"]
        else:
            # no joint element found in XML with child==rotation_body -> fallback False
            # (we still return True because semantics guided selection of bodies)
            self.rotation_joint = None

        return True

    # -------------------------
    # fallback heuristic selection (from previous implementation)
    # -------------------------
    def _choose_base_and_rotation_heuristic(self):
        # choose base: body with name containing base/frame/root or top-level
        base_tokens = ["base", "frame", "root"]
        base_candidates = [b for b in self.body_elems.keys() if text_contains_any(b, base_tokens)]
        if base_candidates:
            base_candidates = sorted(base_candidates, key=lambda b: (0 if self.parent_of.get(b) is None else 1))
            self.base_body = base_candidates[0]
        else:
            top_level = [b for b, p in self.parent_of.items() if p is None]
            self.base_body = top_level[0] if top_level else next(iter(self.body_elems.keys()))

        # pick rotation joint by scoring hinge/revolute and mesh presence and parent==base preference
        best = None; best_score = -1.0
        name_hint_tokens = ["link", "door", "panel", "leaf"]
        for rec in self.joint_records:
            score = 0.0
            if rec["type"] in ("hinge", "revolute"):
                score += 10.0
            if text_contains_any(rec["child"], name_hint_tokens):
                score += 5.0
            child_elem = self.body_elems.get(rec["child"])
            if child_elem is not None:
                mesh_geoms = sum(1 for g in child_elem.findall("geom") if (g.get("mesh") or g.get("type") == "mesh"))
                if mesh_geoms:
                    score += 3.0 + mesh_geoms * 0.5
            if rec["parent"] == self.base_body:
                score += 6.0
            if rec.get("range"):
                try:
                    lo, hi = rec["range"]; score += min(abs(hi - lo), 3.0)
                except Exception:
                    pass
            if score > best_score:
                best_score = score; best = rec
        if best is not None:
            self.rotation_joint = best["name"]
            self.rotation_body = best["child"]
            if best["parent"]:
                self.base_body = best["parent"]
        else:
            # fallback: first joint
            if self.joint_records:
                rec = self.joint_records[0]
                self.rotation_joint = rec["name"]
                self.rotation_body = rec["child"]
                if rec["parent"]:
                    self.base_body = rec["parent"]

    # -------------------------
    # helpers to find rotation joint element
    # -------------------------
    def _find_rotation_joint_elem(self):
        if self.rotation_joint is None:
            return None
        for rec in self.joint_records:
            if rec["name"] == self.rotation_joint:
                return rec["elem"]
        # final fallback search
        wb = getattr(self, "worldbody", None)
        if wb is not None:
            for je in wb.findall(".//joint"):
                if je.get("name") == self.rotation_joint:
                    return je
        return None

    def _set_rotation_friction(self, friction):
        j = self._find_rotation_joint_elem()
        if j is None:
            print("[warn] rotation joint element not found; cannot set friction")
            return
        j.set("frictionloss", array_to_string(np.array([friction])))

    def _set_rotation_damping(self, damping):
        j = self._find_rotation_joint_elem()
        if j is None:
            print("[warn] rotation joint element not found; cannot set damping")
            return
        j.set("damping", str(float(damping)))

    def set_rotation_range_in_xml(self, lower=None, upper=None, convert_radians_to_degrees=True):
        j = self._find_rotation_joint_elem()
        if j is None:
            print("[warn] rotation joint element not found; cannot set range")
            return
        if lower is None or upper is None:
            return
        if convert_radians_to_degrees:
            import math
            lower = math.degrees(lower); upper = math.degrees(upper)
        j.set("range", f"{float(lower)} {float(upper)}")

    # -------------------------
    # runtime mujoco helpers (mujoco_py style)
    # -------------------------
    def get_rotation_qpos(self, sim):
        if self.rotation_joint is None:
            raise RuntimeError("rotation joint not detected")
        try:
            jid = sim.model.joint_name2id(self.rotation_joint)
        except Exception:
            # fallback: pick first hinge joint
            for i in range(sim.model.njnt):
                if sim.model.jnt_type[i] == 0:  # hinge
                    jid = i; break
            else:
                raise
        qpos_addr = sim.model.get_joint_qpos_addr(jid)
        return float(sim.data.qpos[qpos_addr])

    def set_rotation_qpos(self, sim, v):
        if self.rotation_joint is None:
            raise RuntimeError("rotation joint not detected")
        try:
            jid = sim.model.joint_name2id(self.rotation_joint)
        except Exception:
            for i in range(sim.model.njnt):
                if sim.model.jnt_type[i] == 0:
                    jid = i; break
            else:
                raise
        addr = sim.model.get_joint_qpos_addr(jid)
        sim.data.qpos[addr] = float(v)
        try:
            sim.forward()
        except Exception:
            pass

    # -------------------------
    # debug
    # -------------------------
    def debug_print(self):
        print("Semantics parsed:", self.semantics)
        print("Bodies found:", list(self.body_elems.keys()))
        print("Parent map:", self.parent_of)
        print("Joint records:")
        for r in self.joint_records:
            print("  ", r["name"], "type=", r["type"], "parent=", r["parent"], "child=", r["child"], "range=", r["range"])
        print("Chosen base_body:", self.base_body)
        print("Chosen rotation_body:", self.rotation_body)
        print("Chosen rotation_joint:", self.rotation_joint)
