import numpy as np
import mujoco

def mat44_from_R_p(R, p):
    T = np.eye(4)
    T[:3, :3] = np.asarray(R)
    T[:3, 3]  = np.asarray(p)
    return T

def invert44(T):
    R = T[:3, :3]; t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3]  = -R.T @ t
    return Ti

def snapshot_initial_kinematics(sim, object_root_body_id):
    """
    Capture t=0 world transforms for the object root and (optionally) its subtree bodies.
    Returns:
      - T_w_o0: 4x4 world<-objectRoot at t=0
      - T_w_b0: dict {body_id -> 4x4 world<-body at t=0}
    """
    data = sim.data
    T_w_o0 = mat44_from_R_p(
        data.xmat[object_root_body_id].reshape(3,3).copy(),
        data.xpos[object_root_body_id].copy()
    )

    # collect bodies
    body_ids = [k for k,v in sim.model._body_id2name.items() if 'target' in v]

    T_w_b0 = {}
    for b in body_ids:
        T_w_b0[b] = mat44_from_R_p(
            data.xmat[b].reshape(3,3).copy(),
            data.xpos[b].copy()
        )
    return T_w_o0, T_w_b0

def canonicalize_contact_to_initial(sim, model, contact, g_ids, o_ids,
                                   object_root_body_id, T_w_o0, T_w_b0):
    """
    Given an mj_contact and initial snapshots, return the contact pose in:
      - initial body frame (T_b0_c)
      - initial object frame (T_o0_c)
    Also returns which body was touched.
    """
    c = contact
    is_g_o = (c.geom1 in g_ids and c.geom2 in o_ids)
    is_o_g = (c.geom2 in g_ids and c.geom1 in o_ids)
    if not (is_g_o or is_o_g):
        return None

    # which object geom/body is in contact?
    obj_geom = c.geom2 if is_g_o else c.geom1
    obj_body = int(model.geom_bodyid[obj_geom])

    # current world<-contact
    R_wc = np.array(c.frame).reshape(3,3).copy()
    p_w  = c.pos.copy()
    T_w_c = mat44_from_R_p(R_wc, p_w)

    # initial body frame (t=0)
    T_b0_w  = invert44(T_w_b0[obj_body])
    T_b0_c  = T_b0_w @ T_w_c

    # initial object frame (t=0)
    T_w_o0_inv = invert44(T_w_o0)
    T_o0_b0    = T_w_o0_inv @ T_w_b0[obj_body]
    T_o0_c     = T_o0_b0 @ T_b0_c

    return {
        "body_id": obj_body,
        "T_b0_contact": T_b0_c,   # contact in that body's initial frame
        "T_o0_contact": T_o0_c,   # contact in the object's initial frame (canonical)
    }

def get_contacts_canonical(
    sim,
    gripper,                  # GripperModel | list[str] | str
    object_geoms,             # MujocoModel | list[str] | str
    object_root_body_id,      # body id of object root
    T_w_o0=None,              # from snapshot_initial_kinematics (optional if mode='current')
    T_w_b0=None,              # from snapshot_initial_kinematics (optional if mode='current')
    k=2,
    mode="current",           # 'current' -> like your original; 'initial' -> canonical (t=0 object frame)
):
    """
    Returns top-k contacts expressed in the OBJECT frame.
    - mode='current': object frame is the *current* object pose (matches your original function).
    - mode='initial': object frame is the *initial* (t=0) object pose (articulation-invariant).

    """
    model, data = sim.model, sim.data

    # normalize inputs
    def _as_list(x):
        if hasattr(x, "contact_geoms"): return x.contact_geoms
        if isinstance(x, str): return [x]
        return list(x)

    g_geoms = _as_list(gripper)
    o_geoms = _as_list(object_geoms)
    g_ids = {model.geom_name2id(n) for n in g_geoms}
    o_ids = {model.geom_name2id(n) for n in o_geoms}

    # CURRENT object frame (like your original)
    Rwo_t = data.xmat[object_root_body_id].reshape(3,3).copy()  # world<-object(now)
    pwo_t = data.xpos[object_root_body_id].copy()

    # INITIAL object snapshot (optional)
    if mode == "initial":
        if T_w_o0 is None or T_w_b0 is None:
            raise ValueError("mode='initial' requires T_w_o0 and T_w_b0 from snapshot_initial_kinematics().")
        T_o0_w = invert44(T_w_o0)  # object0<-world

    out = []
    f6 = np.zeros(6, dtype=float)

    for i in range(data.ncon):
        c = data.contact[i]
        is_g_o = (c.geom1 in g_ids and c.geom2 in o_ids)
        is_o_g = (c.geom2 in g_ids and c.geom1 in o_ids)
        if not (is_g_o or is_o_g):
            continue

        # contact force
        f6[:] = 0.0
        mujoco.mj_contactForce(model._model, data._data, i, f6)
        fn = max(float(f6[0]), 0.0)
        if fn <= 0.0:
            continue

        # world-space contact pose
        p_w  = c.pos.copy()
        R_wc = np.array(c.frame).reshape(3,3).copy()
        T_w_c = mat44_from_R_p(R_wc, p_w)

        # --- OBJECT-relative transform we will return ---
        if mode == "current":
            # exactly your original: object(now) <- contact
            R_oc = Rwo_t.T @ R_wc
            p_o  = Rwo_t.T @ (p_w - pwo_t)
            T_obj_contact = mat44_from_R_p(R_oc, p_o)

        elif mode == "initial":
            if T_w_o0 is None or T_w_b0 is None:
                raise ValueError("mode='initial' needs T_w_o0, T_w_b0 from snapshot_initial_kinematics().")

            # figure out which object geom/body was touched
            obj_geom = c.geom2 if (c.geom1 in g_ids and c.geom2 in o_ids) else c.geom1
            obj_body = int(model.geom_bodyid[obj_geom])

            # current world<-body_t
            R_wb_t = data.xmat[obj_body].reshape(3,3).copy()
            p_wb_t = data.xpos[obj_body].copy()
            T_w_bt = mat44_from_R_p(R_wb_t, p_wb_t)

            # 1) contact in CURRENT body frame:  T^{b_t}_{c} = (T^{w}_{b_t})^{-1} T^{w}_{c}
            T_bt_c = invert44(T_w_bt) @ T_w_c

            # 2) same local pose but in INITIAL body frame:  T^{w}_{c0} = T^{w}_{b_0} T^{b_t}_{c}
            T_w_b0_body = T_w_b0[obj_body]
            T_w_c0 = T_w_b0_body @ T_bt_c

            # 3) express in INITIAL object frame:  T^{o_0}_{c} = (T^{w}_{o_0})^{-1} T^{w}_{c0}
            T_o0_c = invert44(T_w_o0) @ T_w_c0
            T_obj_contact = T_o0_c
        else:
            raise ValueError("mode must be 'current' or 'initial'.")

        out.append({
            "T_obj_contact": T_obj_contact,
            "p_obj": T_obj_contact[:3, 3],
            "R_obj_contact": T_obj_contact[:3, :3],
            "fn": fn,
            "pair": (model.geom_id2name(c.geom1), model.geom_id2name(c.geom2)),
            "mode": mode,
        })

    out.sort(key=lambda d: d["fn"], reverse=True)
    if k is not None and k > 0:
        out = out[:k]
    return out