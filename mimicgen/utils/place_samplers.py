import collections
from copy import copy

import numpy as np

from robosuite.models.objects import MujocoObject
from robosuite.utils import RandomizationError
from robosuite.utils.transform_utils import quat_multiply
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

def _rotmat_from_axis_angle(axis, angle):
    if isinstance(axis, str):
        axis = {'x':[1,0,0],'y':[0,1,0],'z':[0,0,1]}[axis.lower()]
    axis = np.asarray(axis, float)
    axis /= (np.linalg.norm(axis) + 1e-12)
    x,y,z = axis
    K = np.array([[0,-z, y],[z,0,-x],[-y,x,0]], float)
    s, c = np.sin(angle), np.cos(angle)
    return np.eye(3) + s*K + (1-c)*(K@K)

def recompute_sites_for_rotation(
    bottom_offset,                 # np.array(3,) -> [0,0,zmin]
    top_offset,                    # np.array(3,) -> [0,0,zmax]
    bbox_half_size_vec,            # np.array(3,) -> [rx, ry, -zmin]  (from get_bounding_box_half_size)
    rotation_axis, rotation_angle, # baked mesh rotation you applied
    circular_radius=False,
    clearance=0.0,
):
    """
    Returns (bottom_site_new, top_site_new, horizontal_radius_site_new),
    all in the body frame, after applying the baked rotation to the bbox.
    """
    bottom_offset = np.asarray(bottom_offset, float)
    top_offset    = np.asarray(top_offset, float)
    half_vec      = np.asarray(bbox_half_size_vec, float)

    zmin = float(bottom_offset[2])
    zmax = float(top_offset[2])
    rx   = float(abs(half_vec[0]))
    ry   = float(abs(half_vec[1]))
    # half_vec[2] should be -zmin; we rely on bottom/top for zmin/zmax

    # Original body-frame bbox corners (before baked rotation)
    xs = np.array([-rx, rx])
    ys = np.array([-ry, ry])
    zs = np.array([zmin, zmax])
    corners = np.array([[x, y, z] for x in xs for y in ys for z in zs], float)

    # Apply baked rotation (geometry rotated; body frame fixed)
    R = _rotmat_from_axis_angle(rotation_axis, rotation_angle)
    corners_r = corners @ R.T

    # Recompute sites
    zmin_r = float(corners_r[:, 2].min())
    zmax_r = float(corners_r[:, 2].max())

    if circular_radius:
        r_xy = float(np.linalg.norm(corners_r[:, :2], axis=1).max()) + float(clearance)
        hrs_new = np.array([r_xy, 0.0, 0.0], float)
    else:
        rx_r = float(np.abs(corners_r[:, 0]).max()) + float(clearance)
        ry_r = float(np.abs(corners_r[:, 1]).max()) + float(clearance)
        hrs_new = np.array([rx_r, ry_r, 0.0], float)

    bottom_new = np.array([0.0, 0.0, zmin_r], float)
    top_new    = np.array([0.0, 0.0, zmax_r], float)
    return bottom_new, top_new, hrs_new
class MyUniformRandomSampler(UniformRandomSampler):

    def sample(self, fixtures=None, reference=None, on_top=True):
        """
        Uniformly sample relative to this sampler's reference_pos or @reference (if specified).

        Args:
            fixtures (dict): dictionary of current object placements in the scene as well as any other relevant
                obstacles that should not be in contact with newly sampled objects. Used to make sure newly
                generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)

            reference (str or 3-tuple or None): if provided, sample relative placement. Can either be a string, which
                corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
                relative to this sampler's `'reference_pos'` value.

            on_top (bool): if True, sample placement on top of the reference object. This corresponds to a sampled
                z-offset of the current sampled object's bottom_offset + the reference object's top_offset
                (if specified)

        Return:
            dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
                placements specified in @fixtures. Note quat is in (w,x,y,z) form

        Raises:
            RandomizationError: [Cannot place all objects]
            AssertionError: [Reference object name does not exist, invalid inputs]
        """
        # Standardize inputs
        placed_objects = {} if fixtures is None else copy(fixtures)
        if reference is None:
            base_offset = self.reference_pos
        elif type(reference) is str:
            assert (
                reference in placed_objects
            ), "Invalid reference received. Current options are: {}, requested: {}".format(
                placed_objects.keys(), reference
            )
            ref_pos, _, ref_obj = placed_objects[reference]
            base_offset = np.array(ref_pos)
            if on_top:
                base_offset += np.array((0, 0, ref_obj.top_offset[-1]))
        else:
            base_offset = np.array(reference)
            assert (
                base_offset.shape[0] == 3
            ), "Invalid reference received. Should be (x,y,z) 3-tuple, but got: {}".format(base_offset)

        # Sample pos and quat for all objects assigned to this sampler
        for obj in self.mujoco_objects:
            # First make sure the currently sampled object hasn't already been sampled
            assert obj.name not in placed_objects, "Object '{}' has already been sampled!".format(obj.name)

            # horizontal_radius = obj.horizontal_radius
            # bottom_offset = obj.bottom_offset
            bottom_offset, _, horizontal_radius = recompute_sites_for_rotation(
                obj.bottom_offset,
                obj.top_offset,
                obj.get_bounding_box_half_size(),
                rotation_axis=self.rotation_axis,
                rotation_angle=self.rotation
            )
            success = False
            for i in range(5000):  # 5000 retries
                object_x = self._sample_x(horizontal_radius) + base_offset[0]
                object_y = self._sample_y(horizontal_radius) + base_offset[1]
                object_z = self.z_offset + base_offset[2]
                if on_top:
                    object_z -= bottom_offset[-1]

                # objects cannot overlap
                location_valid = True
                if self.ensure_valid_placement:
                    for (x, y, z), _, other_obj in placed_objects.values():
                        if (
                            np.linalg.norm((object_x - x, object_y - y))
                            <= other_obj.horizontal_radius + horizontal_radius
                        ) and (object_z - z <= other_obj.top_offset[-1] - bottom_offset[-1]):
                            location_valid = False
                            break

                if location_valid:
                    # random rotation
                    quat = self._sample_quat()

                    # multiply this quat by the object's initial rotation if it has the attribute specified
                    if hasattr(obj, "init_quat"):
                        quat = quat_multiply(quat, obj.init_quat)
                    # quat = np.array([1, 0, 0, 0])

                    # location is valid, put the object down
                    pos = (object_x, object_y, object_z)
                    placed_objects[obj.name] = (pos, quat, obj)
                    success = True
                    break

            if not success:
                raise RandomizationError("Cannot place all objects ):")

        return placed_objects

class MultiAxisUniformRandomSampler(UniformRandomSampler):
    def _sample_quat(self):
        """
        Samples the orientation for a given object allowing multiple rotation axes.

        Supports:
          - self.rotation_axis: "x"/"y"/"z" or iterable of axes like ["z","x"]
          - self.rotation: a list of None, scalar, (min,max), or iterable matching rotation_axis length,
                          or a single (min,max)/scalar reused for all axes.

        Returns:
            np.array: sampled quaternion in (w, x, y, z) form
        """
        # normalize axes to a list
        if isinstance(self.rotation_axis, str):
            axes = [self.rotation_axis.lower()]
        elif isinstance(self.rotation_axis, collections.abc.Iterable):
            axes = [a.lower() for a in self.rotation_axis]
        else:
            raise ValueError(f"Invalid rotation_axis: {self.rotation_axis}")

        # normalize rotation spec to a list of same length as axes
        def _is_range_like(v):
            return isinstance(v, collections.abc.Iterable) and len(v) == 2

        # rot_specs = self.rotation
        rot_specs = None

        # quaternion helpers (w, x, y, z)
        def quat_mul(q1, q2):
            # Hamilton product: q = q1 * q2
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            return np.array([w, x, y, z], dtype=float)

        def quat_from_axis_angle(axis, angle):
            ca = np.cos(angle / 2.0)
            sa = np.sin(angle / 2.0)
            if axis == "x":
                return np.array([ca, sa, 0.0, 0.0], dtype=float)
            elif axis == "y":
                return np.array([ca, 0.0, sa, 0.0], dtype=float)
            elif axis == "z":
                return np.array([ca, 0.0, 0.0, sa], dtype=float)
            else:
                raise ValueError(f"Invalid axis '{axis}' (must be 'x','y','z')")

        # Compose per-axis quaternions in order: first axis applied first.
        q_total = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        for ax, rot_spec in zip(axes, rot_specs):
            if rot_spec is None:
                rot_angle = np.random.uniform(high=2 * np.pi, low=0)
            elif isinstance(rot_spec, collections.abc.Iterable):
                rot_angle = np.random.uniform(high=max(rot_spec), low=min(rot_spec))
            else:
                rot_angle = rot_spec

            q_axis = quat_from_axis_angle(ax, rot_angle)
            # Compose: q_total = q_total * q_axis  (apply q_axis after q_total)
            q_total = quat_mul(q_total, q_axis)

        # normalize to avoid numerical drift
        q_total = q_total / np.linalg.norm(q_total)
        return q_total