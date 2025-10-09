from collections import OrderedDict
import os

import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import DoorObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.environments.manipulation.door import Door
from robosuite.models.objects import MujocoXMLObject
from mimicgen.models.robosuite.objects import ArticulatedObject
from mimicgen.utils.place_samplers import MultiAxisUniformRandomSampler


from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
# removed: from robosuite.models.objects import DoorObject  # no longer used
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
# removed: from robosuite.environments.manipulation.door import Door
from robosuite.models.objects import MujocoXMLObject
from mimicgen.models.robosuite.objects import ArticulatedObject  # your new object
from robosuite.models.base import MujocoModel
from robosuite.models.grippers import GripperModel
import mujoco

class Open_D0(SingleArmEnv):
    def __init__(
        self,
        robots,
        obj_xml_path,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        use_latch=True,  # we keep the flag but treat latch as optional (may be missing)
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
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
    ):
        self.table_full_size = (0.8, 0.5, 0.05)
        self.table_offset = (-0.2, -0.4, 0.8)

        self.use_latch = use_latch
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        self.use_object_obs = use_object_obs
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
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 1.0
        # elif self.reward_shaping:
        #     # Reaching: use rotation body position as proxy for handle position
        #     dist = np.linalg.norm(self._gripper_to_rotation_body)
        #     reaching_reward = 0.25 * (1 - np.tanh(10.0 * dist))
        #     reward += reaching_reward

        #     # Rotating: use hinge qpos (if present) if use_latch was intended for rotation reward
        #     if self.use_latch and getattr(self, "hinge_qpos_addr", None) is not None:
        #         hinge_qpos = self.sim.data.qpos[self.hinge_qpos_addr]
        #         reward += np.clip(0.25 * np.abs(hinge_qpos / (0.5 * np.pi)), -0.25, 0.25)

        if self.reward_scale is not None:
            reward *= self.reward_scale / 1.0

        return reward

    def _load_model(self):
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        xpos = (xpos[0], self.table_offset[1] + self.table_full_size[1] + 0.2, 0)
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(table_full_size=self.table_full_size, table_offset=self.table_offset)
        mujoco_arena.set_origin([0, 0, 0])
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5986131746834771, -4.392035683362857e-09, 1.5903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349],
        )

        # Instantiate ArticulatedObject (looks for semantic.txt in same dir)
        xml_dir = os.path.dirname(os.path.abspath(self.obj_xml_path))
        semantic_path = os.path.join(xml_dir, "semantics.txt").replace('processed_', '')
        if not os.path.exists(semantic_path):
            semantic_path = None  # object will fall back to heuristics

        self.target = ArticulatedObject(
            self.obj_xml_path,
            name="target",
            semantic_path=semantic_path,  # pass semantic file
            # joints=None,
        )

        # Placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.target)
        else:
            self.placement_initializer = MultiAxisUniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.target,
                x_range=[0.07, 0.09],
                y_range=[-0.01, 0.01],
                # rotation=(-np.pi / 2.0 - 0.25, -np.pi / 2.0),
                rotation=[-np.pi / 2.0, np.pi],
                rotation_axis=["x", "z"],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.target,
        )

    def _setup_references(self):
        super()._setup_references()

        # Map object bodies
        self.object_body_ids = {}
        # base_body: the static frame / main body in semantic.txt
        base_name = getattr(self.target, "base_body", None)
        if base_name is None:
            # fallback: use whatever the object exposes as root_body or first body
            base_name = getattr(self.target, "root_body", None) or list(self.sim.model.body_names)[0]
        self.object_body_ids["base"] = self.sim.model.body_name2id(base_name)

        # rotation_body: the moving door panel
        rotation_name = getattr(self.target, "rotation_body", None)
        if rotation_name is not None:
            self.object_body_ids["rotation"] = self.sim.model.body_name2id(rotation_name)
        else:
            # fallback to base if missing (degenerate)
            self.object_body_ids["rotation"] = self.object_body_ids["base"]

        # hinge joint qpos address (if rotation_joint present)
        rotation_joint_name = getattr(self.target, "rotation_joint", None)
        self.hinge_qpos_addr = None
        if rotation_joint_name is not None:
            try:
                self.hinge_qpos_addr = self.sim.model.get_joint_qpos_addr(rotation_joint_name)
            except Exception:
                # joint may be missing in sim if names differ; leave None
                self.hinge_qpos_addr = None

        # NOTE: removed latch/handle-specific references:
        # - self.object_body_ids["latch"], self.target.latch_body, self.target_handle_site_id, handle_qpos addr, etc.
        # If your semantic file provides a specific handle site name you can add it here.

    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            @sensor(modality=modality)
            def door_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.object_body_ids["base"]])

            @sensor(modality=modality)
            def rotation_body_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.object_body_ids["rotation"]])

            @sensor(modality=modality)
            def door_to_eef_pos(obs_cache):
                if "door_pos" in obs_cache and f"{pf}eef_pos" in obs_cache:
                    return obs_cache["door_pos"] - obs_cache[f"{pf}eef_pos"]
                return np.zeros(3)

            @sensor(modality=modality)
            def rotation_to_eef_pos(obs_cache):
                if "rotation_body_pos" in obs_cache and f"{pf}eef_pos" in obs_cache:
                    return obs_cache["rotation_body_pos"] - obs_cache[f"{pf}eef_pos"]
                return np.zeros(3)

            @sensor(modality=modality)
            def hinge_qpos(obs_cache):
                if self.hinge_qpos_addr is not None:
                    return np.array([self.sim.data.qpos[self.hinge_qpos_addr]])
                else:
                    return np.array([0.0])

            sensors = [door_pos, rotation_body_pos, door_to_eef_pos, rotation_to_eef_pos, hinge_qpos]
            names = [s.__name__ for s in sensors]

            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            door_pos, door_quat, _ = object_placements[self.target.name]

            # Use base_body for positioning the object instance
            base_name = getattr(self.target, "base_body", None)
            if base_name is None:
                base_name = getattr(self.target, "root_body", None) or list(self.sim.model.body_names)[0]
            base_id = self.sim.model.body_name2id(base_name)
            self.sim.model.body_pos[base_id] = door_pos
            self.sim.model.body_quat[base_id] = door_quat

    def _check_success(self):
        # success if hinge rotated beyond threshold (if hinge exists)
        return False
        if self.hinge_qpos_addr is None:
            return False
        hinge_qpos = self.sim.data.qpos[self.hinge_qpos_addr]
        return hinge_qpos > 0.3

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)

        # if vis_settings.get("grippers", False):
        #     # visualize using rotation body as proxy target
        #     self._visualize_gripper_to_target(
        #         gripper=self.robots[0].gripper, target=self.object_body_ids["rotation"], target_type="target_main"
        #     )
        
    @property
    def _rotation_body_xpos(self):
        return self.sim.data.body_xpos[self.object_body_ids["rotation"]]

    @property
    def _gripper_to_rotation_body(self):
        return self._rotation_body_xpos - self._eef_xpos
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
            mujoco.mj_contactForce(model._model, data._data, i, f_contact)

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

if __name__ == "__main__":
    env = Open_D0(
        robots="Panda",
        # obj_xml_path="partnet-mobility/processed_dataset/10849/mobility_coacd.xml",
        obj_xml_path="partnet-mobility/processed_dataset/12540/mobility_coacd.xml",
        use_latch=False,  # D0 has no latch
        has_renderer=True,
        ignore_done=True,
        reward_shaping=True,
        control_freq=20,
    )

    for ep in range(5):
        obs = env.reset()
        done = False
        total_reward = 0.0
        low, high = env.action_spec
        while not done:
            env.render()
            action = np.random.uniform(low, high)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        print(f"Episode {ep} ended with total reward {total_reward}")
    env.close()