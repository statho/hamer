from typing import Dict

import numpy as np

from .geometry import aa_to_rotmat, rotmat_to_aa

NUM_BODY_JOINTS = 21
COMMON_CHAIN = [0, 3, 6, 9]
RIGHT_ELBOW_CHAIN = [14, 17, 19]
LEFT_ELBOW_CHAIN = [13, 16, 18]


def flip_pose(rpose: np.ndarray) -> np.ndarray:
    """
    Flip MANO parameters (right hand to left hand).
    :rpose: np.ndarray of shape (B, C), where C=num_joints*3, containing the joint rotations of the right hand MANO model as rotation vectors.
    Returns:
    lpose: np.ndarray of the same shape, where the joint rotations are flipped so we can be directly used with the left hand MANO model.
    """
    lpose = rpose.copy()
    lpose[:, 1::3] *= -1
    lpose[:, 2::3] *= -1
    return lpose.astype(np.float32)


def combine_body_and_hands(
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    rh_global_orient: np.ndarray,
    lh_global_orient: np.ndarray,
    rh_valid: np.ndarray,
    lh_valid: np.ndarray,
) -> np.ndarray:
    """
    Transform the global orinetation of the hands to a the local coordinate system of the elbow.
    This is needed to appropriately merge the model parameters from a body-only and hand-only models.
    Args:
    :global_orient : np.ndarray of shape (B, 3), contatining the global orientation of the root joint as rotations vector.
    :body_pose : np.ndarray of shape (B, BODY_JOINTS, 3), containing the rotation of the BODY_JOINTS. If BODY_JOINTS > 21, we keep only the rotation of the 21 first joints.
    :rh_global_orient : np.ndarray of shape (B, 3) containing the global orientation of the right hand as rotation vector.
    :lh_global_orient : np.ndarray of shape (B, 3) containing the global orientation of the left hand as rotation vector.
    :rh_valid : np.ndarray of shape (B,) containing the visibility of the right hand.
    :lh_valid : np.ndarray of shape (B,) containing the visibility of the left hand.
    Returns:
    - body_pose: np.ndarray of shape (B, BODY_JOINTS, 3) containing the rotations of the BODY_JOINTS.
        The rotation of the hands is used from the hand models, we express the hand_global_orientation wrt to the local coordinate frame of the elbow.
    """
    bs = global_orient.shape[0]
    lh_global_orient = flip_pose(lh_global_orient)
    pose = np.concatenate(
        (global_orient[:, None], body_pose[:, :NUM_BODY_JOINTS]), axis=1
    )

    # Convert rotation vectors to rotation matrices.
    pose_rotmat = aa_to_rotmat(
        torch.from_numpy(pose).reshape(bs * (NUM_BODY_JOINTS + 1), 3)
    ).reshape(bs, NUM_BODY_JOINTS + 1, 3, 3)
    rh_global_orient_rotmat = aa_to_rotmat(torch.from_numpy(rh_global_orient))
    lh_global_orient_rotmat = aa_to_rotmat(torch.from_numpy(lh_global_orient))

    # Compute the global rotation of the common joints in the SMPL kinematic tree.
    init_rotmat = torch.eye(3)
    init_rotmat = init_rotmat.unsqueeze(0).repeat(bs, 1, 1)
    for joint_idx in COMMON_CHAIN:
        init_rotmat = init_rotmat @ pose_rotmat[:, joint_idx]
    right_elbow_rotmat = init_rotmat.clone()
    left_elbow_rotmat = init_rotmat.clone()

    # Compute the global rotation of the right elbow.
    for joint_idx in RIGHT_ELBOW_CHAIN:
        right_elbow_rotmat = right_elbow_rotmat @ pose_rotmat[:, joint_idx]
    # Compute the global rotation of the left elbow.
    for joint_idx in LEFT_ELBOW_CHAIN:
        left_elbow_rotmat = left_elbow_rotmat @ pose_rotmat[:, joint_idx]

    right_wrist_rotmat_rh = (
        right_elbow_rotmat.transpose(-1, -2) @ rh_global_orient_rotmat
    )
    left_wrist_rotmat_lh = left_elbow_rotmat.transpose(-1, -2) @ lh_global_orient_rotmat

    # Overwrite the hand rotation from HMR 2.0 with the one from Hamer.
    pose_rotmat[rh_valid, 21] = right_wrist_rotmat_rh[rh_valid]
    pose_rotmat[lh_valid, 20] = left_wrist_rotmat_lh[lh_valid]
    pose_out = rotmat_to_aa(pose_rotmat)
    body_pose_out = pose_out[:, 1:].numpy()  # bs, 21, 3 : return only the new body_pose
    return body_pose_out


def smpl_mano_to_smplh(
    smpl_params: Dict[np.ndarray],
    lh_params: Dict[np.ndarray],
    rh_params: Dict[np.ndarray],
) -> Dict[np.ndarray]:
    """
    :smpl_params : dictionary with SMPL parameters
    :lh_params : dictionary with left hand parameters from Hamer
    :rh_params : dictionary with right hand parameters from Hamer
    Returns:
    - smplh_params : dictionary with SMPL-H parameters
    """
    body_pose = combine_body_and_hands(
        smpl_params["global_orient"],
        smpl_params["body_pose"],
        rh_params["global_orient"],
        lh_params["global_orient"],
        rh_params["is_valid"],
        lh_params["is_valid"],
    )
    is_valid = smpl_params["is_valid"] & rh_params["is_valid"] & lh_params["is_valid"]
    bs = is_valid.shape[0]
    left_hand_pose = flip_pose(lh_params["hand_pose"].reshape(bs, -1)).reshape(
        bs, 15, 3
    )
    smplh_params = {
        "betas": smpl_params["betas"],
        "global_orient": smpl_params["global_orient"],
        "body_pose": body_pose,
        "left_hand_pose": left_hand_pose,
        "right_hand_pose": right_hand_pose,
        "is_valid": is_valid,
    }
    return smplh_params
