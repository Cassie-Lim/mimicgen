

from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from mimicgen.utils.place_samplers import MyUniformRandomSampler
from robosuite.utils.transform_utils import convert_quat
from mimicgen.models.robosuite.objects.xml_objects import MujocoXMLObject
from robosuite.models.base import MujocoModel
from robosuite.models.grippers import GripperModel
import mujoco

class Lift_D0(SingleArmEnv):
    """
    This class corresponds to the lifting task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (target) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        obj_xml_path=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        self.obj_xml_path = obj_xml_path

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided if the target is lifted

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arm to reach the target
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the target
            - Lifting: in {0, 1}, non-zero if arm has lifted the target

        The sparse reward only consists of the lifting component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 2.25

        # use a shaping reward
        elif self.reward_shaping:

            # reaching reward
            target_pos = self.sim.data.body_xpos[self.target_body_id]
            gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
            dist = np.linalg.norm(gripper_site_pos - target_pos)
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward

            # grasping reward
            if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.target):
                reward += 0.25

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "target",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.target = MujocoXMLObject(
            self.obj_xml_path,
            name="target",
            joints=[dict(type="free", damping="0")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.target)
        else:
            self.placement_initializer = MyUniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.target,
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                rotation=np.pi/2,
                rotation_axis="x",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.target,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.target_body_id = self.sim.model.body_name2id(self.target.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # target-related observables
            @sensor(modality=modality)
            def target_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.target_body_id])

            @sensor(modality=modality)
            def target_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.target_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_target_pos(obs_cache):
                return (
                    obs_cache[f"{pf}eef_pos"] - obs_cache["target_pos"]
                    if f"{pf}eef_pos" in obs_cache and "target_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors = [target_pos, target_quat, gripper_to_target_pos]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the target.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the target
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.target)

    def _check_success(self):
        """
        Check if target has been lifted.

        Returns:
            bool: True if target has been lifted
        """
        target_height = self.sim.data.body_xpos[self.target_body_id][2]
        table_height = self.model.mujoco_arena.table_offset[2]

        # target is higher than the table top above a margin
        return target_height > table_height + 0.15

    def edit_model_xml(self, xml_str):
        """
        This function edits the model xml with custom changes, including resolving relative paths,
        applying changes retroactively to existing demonstration files, and other custom scripts.
        Environment subclasses should modify this function to add environment-specific xml editing features.
        Args:
            xml_str (str): Mujoco sim demonstration XML file as string
        Returns:
            str: Edited xml file as string
        """
        return xml_str

    # def get_contact_point(self, gripper, object_geoms):
    #     """
    #     Return a list of 4x4 contact poses expressed in the OBJECT frame.
    #     Each 4x4 has rotation (from contact.frame) and translation (contact point).

    #     Args:
    #         gripper (GripperModel | list[str] | str)
    #         object_geoms (MujocoModel | list[str] | str)

    #     Returns:
    #         list[np.ndarray]: each shape (4,4), object-frame contact pose matrices.
    #     """
    #     model = self.sim.model
    #     data  = self.sim.data

    #     # --- standardize inputs ---
    #     if isinstance(object_geoms, MujocoModel):
    #         o_geoms = object_geoms.contact_geoms
    #         object_key = getattr(object_geoms, "name", None) or tuple(sorted(o_geoms))
    #     elif isinstance(object_geoms, str):
    #         o_geoms = [object_geoms]
    #         object_key = tuple(sorted(o_geoms))
    #     else:
    #         o_geoms = object_geoms
    #         object_key = tuple(sorted(o_geoms))

    #     if isinstance(gripper, GripperModel):
    #         g_geoms = gripper.contact_geoms
    #     elif isinstance(gripper, str):
    #         g_geoms = [gripper]
    #     else:
    #         g_geoms = gripper

    #     # Map names -> ids
    #     g_ids = {model.geom_name2id(n) for n in g_geoms}
    #     o_ids = {model.geom_name2id(n) for n in o_geoms}

    #     # --- choose object reference body (unique per object) ---
    #     ref_geom_id = next(iter(o_ids))
    #     obj_body_id = int(model.geom_bodyid[ref_geom_id])

    #     # Object pose in world
    #     Rwo = data.xmat[obj_body_id].reshape(3, 3).copy()   # world←object rotation
    #     pwo = data.xpos[obj_body_id].copy()                 # world position of object origin

    #     # # Cache transforms (optional)
    #     # if not hasattr(self, "_obj_frame_cache"):
    #     #     self._obj_frame_cache = {}
    #     # T_obj_world = np.eye(4); T_obj_world[:3,:3] = Rwo; T_obj_world[:3,3] = pwo
    #     # T_world_obj = np.eye(4); T_world_obj[:3,:3] = Rwo.T; T_world_obj[:3,3] = -Rwo.T @ pwo
    #     # self._obj_frame_cache[object_key] = {
    #     #     "body_id": obj_body_id,
    #     #     "T_obj_world": T_obj_world,
    #     #     "T_world_obj": T_world_obj,
    #     # }

    #     # --- collect contacts and convert to object frame (full 4x4 pose) ---
    #     poses_obj = []
    #     for i in range(data.ncon):
    #         c = data.contact[i]
    #         if not ((c.geom1 in g_ids and c.geom2 in o_ids) or (c.geom2 in g_ids and c.geom1 in o_ids)):
    #             continue

    #         # world-space contact point and frame
    #         p_w = c.pos.copy()                              # (3,)
    #         R_wc = np.array(c.frame).reshape(3, 3).copy()   # world←contact (cols: [normal, t1, t2])

    #         # express in object frame
    #         p_o  = Rwo.T @ (p_w - pwo)
    #         R_oc = Rwo.T @ R_wc

    #         T_oc = np.eye(4)
    #         T_oc[:3, :3] = R_oc
    #         T_oc[:3, 3]  = p_o
    #         poses_obj.append(T_oc)

    #     return poses_obj

    def get_contact_points(self, gripper, object_geoms, k=2):
        """
        Return up to k strongest contacts (by normal force) between gripper and object.

        Args:
            gripper (GripperModel | list[str] | str)
            object_geoms (MujocoModel | list[str] | str)
            k (int): number of contacts to keep

        Returns:
            contacts: list of dicts (length <= k), each with:
                {
                "T_obj_contact": (4,4) contact pose in OBJECT frame,
                "p_obj": (3,) contact position in OBJECT frame,
                "R_obj_contact": (3,3) rotation (object<-contact),
                "fn": float  # normal force (>=0) in contact frame
                "ft": float  # tangential force magnitude
                "condim": int
                "pair": (geom1_name, geom2_name)
                }
        Notes:
            - Call this AFTER the solver runs (e.g., after a simulation step) so forces are valid.
            - Contact frame columns are [normal, tangent1, tangent2, ...].
        """
        model = self.sim.model
        data  = self.sim.data

        # object geoms
        if isinstance(object_geoms, MujocoModel):
            o_geoms = object_geoms.contact_geoms
        elif isinstance(object_geoms, str):
            o_geoms = [object_geoms]
        else:
            o_geoms = list(object_geoms)

        # gripper geoms
        if isinstance(gripper, GripperModel):
            g_geoms = gripper.contact_geoms
        elif isinstance(gripper, str):
            g_geoms = [gripper]
        else:
            g_geoms = list(gripper)

        g_ids = {model.geom_name2id(n) for n in g_geoms}
        o_ids = {model.geom_name2id(n) for n in o_geoms}

        # choose reference body for object frame
        ref_gid    = next(iter(o_ids))
        obj_bodyid = int(model.geom_bodyid[ref_gid])

        Rwo = data.xmat[obj_bodyid].reshape(3, 3).copy()  # world<-object
        pwo = data.xpos[obj_bodyid].copy()                # world pos of object origin

        results = []
        # scratch for mj_contactForce
        f_contact = np.zeros(6, dtype=float)

        for i in range(data.ncon):
            c = data.contact[i]
            pair_ok = ((c.geom1 in g_ids and c.geom2 in o_ids) or
                    (c.geom2 in g_ids and c.geom1 in o_ids))
            if not pair_ok:
                continue

            # Extract force in contact frame
            f_contact[:] = 0.0
            mujoco.mj_contactForce(model, data, i, f_contact)

            # normal force (>=0 if pushing)
            fn = max(float(f_contact[0]), 0.0)
            if fn <= 0.0:
                # ignore separating / near-zero contacts
                continue

            # tangential magnitude (uses condim to know how many tangential components apply)
            condim = int(c.dim) if hasattr(c, "dim") else int(model.geom_condim[c.geom1])
            tangential = f_contact[1:min(condim, 3)]  # at most 2 tangential terms are used in typical setups
            ft = float(np.linalg.norm(tangential))

            # world-space contact pose
            p_w  = c.pos.copy()
            R_wc = np.array(c.frame).reshape(3, 3).copy()  # world<-contact

            # convert to object frame
            p_o  = Rwo.T @ (p_w - pwo)
            R_oc = Rwo.T @ R_wc

            T_oc = np.eye(4)
            T_oc[:3, :3] = R_oc
            T_oc[:3, 3]  = p_o

            results.append({
                "T_obj_contact": T_oc,
                "p_obj": p_o,
                "R_obj_contact": R_oc,
                "fn": fn,
                "ft": ft,
                "condim": condim,
                "pair": (model.geom_id2name(c.geom1), model.geom_id2name(c.geom2)),
            })

        # sort by descending normal force and keep top-k
        results.sort(key=lambda d: d["fn"], reverse=True)
        if k is not None and k > 0:
            results = results[:k]

        return [r["T_obj_contact"] for r in results]
