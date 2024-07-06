import numpy as np
from scipy.spatial.transform import Rotation as R


def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i + 1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1, -1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_names = []
    joint_parents = []
    joint_offsets = np.array([[0, 0, 0]])

    # Initialize the stack
    stack = []

    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
    # Find the ROOT joint
    root_idx = lines.index('ROOT RootJoint\n')
    # Parse the rest of the hierarchy structure
    joint_hierarchy = lines[root_idx + 4:]

    # Initialize Root joint
    joint_names.append('RootJoint')
    joint_parents.append(-1)

    parent_name = 'RootJoint'
    # Find the child joints and add them to the stack
    for i in range(len(joint_hierarchy)):
        line = joint_hierarchy[i].strip()
        if (line.startswith("JOINT")) | (line.startswith("End")):
            if line.startswith("End"):
                # The end node is named after the parent node + "_end"
                child_name = f'{parent_name}_end'
            else:
                child_name = line.split()[1]

            joint_names.append(child_name)
            joint_parents.append(joint_names.index(parent_name))

            # if line.startswith("JOINT"):
            stack.append((child_name, parent_name))
            parent_name = child_name
        elif line.startswith("}"):
            if len(stack) == 0:
                parent_name = 'RootJoint'
            else:
                tmp, parent_name = stack.pop()
        elif line.startswith("OFFSET"):
            # Split the string by space, get the offset in X, Y, Z.
            XYZ = line.split()[1:]
            # Convert to float
            offset_values = [float(offset) for offset in XYZ]
            # Add offset to joint_offsets
            joint_offsets = np.append(joint_offsets, [offset_values], axis=0)
    return joint_names, joint_parents, joint_offsets


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    # Initialization
    joint_positions = np.zeros((len(joint_offset), 3))
    joint_orientations = np.zeros((len(joint_offset), 4))
    idx_offset = 0
    # Iterate offsets (i.e. all nodes)
    for idx, offset in enumerate(joint_offset):
        cur_joint_name = joint_name[idx]
        parent_idx = joint_parent[idx]

        if cur_joint_name.startswith('RootJoint'):
            # For root，there are three more position data
            joint_positions[idx] = motion_data[frame_id, :3]
            joint_orientations[idx] = R.from_euler('XYZ', motion_data[frame_id, 3:6],
                                                   degrees=True).as_quat()
        elif cur_joint_name.endswith('_end'):
            # Perform Quaternion Multiplication
            q_result = joint_orientations[parent_idx] * np.concatenate(([0], offset)) * \
                       joint_orientations[parent_idx].conj()
            joint_positions[idx] = joint_positions[parent_idx] + q_result[1:]
            idx_offset += 1
        else:
            # Normal Node
            # Get the rotation data (Euler angles) for the current joint and convert it to a rotation matrix
            rotation = R.from_euler('XYZ', motion_data[frame_id, 3 * (idx - idx_offset + 1):3 * (idx - idx_offset + 2)],
                                    degrees=True).as_matrix()
            # Get the rotation matrix of the parent joint
            rot_matrix_p = R.from_quat(joint_orientations[parent_idx]).as_matrix()
            # Calculate the global rotation of the current joint
            tmp = rot_matrix_p.dot(rotation)
            # Convert the result to quaternions
            joint_orientations[idx] = R.from_matrix(tmp).as_quat()
            # Calculate the offset in the coordinate system of the parent node
            joint_positions[idx] = joint_positions[parent_idx] + rot_matrix_p.dot(offset)
    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """

    def index_bone_to_channel(index, flag):
        if flag == 't':
            end_bone_index = end_bone_index_t
        else:
            end_bone_index = end_bone_index_a
        for i in range(len(end_bone_index)):
            if end_bone_index[i] > index:
                return index - i
        return index - len(end_bone_index)

    def get_t2a_offset(bone_name):
        l_bone = ['lShoulder', 'lElbow', 'lWrist']
        r_bone = ['rShoulder', 'rElbow', 'rWrist']
        if bone_name in l_bone:
            return R.from_euler('XYZ', [0., 0., 45.], degrees=True)
        if bone_name in r_bone:
            return R.from_euler('XYZ', [0., 0., -45.], degrees=True)
        return R.from_euler('XYZ', [0., 0., 0.], degrees=True)

    # Load motion data, np.ndarray of shape (N,X), where N is the number of frames and X is the number of Channels
    motion_data = load_motion_data(A_pose_bvh_path)

    # Load pose data
    t_name, t_parent, t_offset = part1_calculate_T_pose(T_pose_bvh_path)
    a_name, a_parent, a_offset = part1_calculate_T_pose(A_pose_bvh_path)

    end_bone_index_t = []
    for i in range(len(t_name)):
        if t_name[i].endswith('_end'):
            end_bone_index_t.append(i)

    end_bone_index_a = []
    for i in range(len(a_name)):
        if a_name[i].endswith('_end'):
            end_bone_index_a.append(i)

    # Iterate motion frames
    for m_i in range(len(motion_data)):
        frame = motion_data[m_i]
        cur_frame = np.empty(frame.shape[0])
        cur_frame[:3] = frame[:3]
        # Iterate joints of T-pose
        for t_i in range(len(t_name)):
            # Find the corresponding A-pose bone index based on the T-pose bone name
            cur_bone = t_name[t_i]
            a_i = a_name.index(t_name[t_i])
            # Exclusion of end bones
            if cur_bone.endswith('_end'):
                continue
            # Compute the channel index
            channel_t_i = index_bone_to_channel(t_i, 't')
            channel_a_i = index_bone_to_channel(a_i, 'a')

            # retarget
            local_rotation = frame[3 + channel_a_i * 3: 6 + channel_a_i * 3]
            if cur_bone in ['lShoulder', 'lElbow', 'lWrist', 'rShoulder', 'rElbow', 'rWrist']:
                p_bone_name = t_name[t_parent[t_i]]
                Q_pi = get_t2a_offset(p_bone_name)
                Q_i = get_t2a_offset(cur_bone)
                local_rotation = (Q_pi * R.from_euler('XYZ', local_rotation, degrees=True) * Q_i.inv()) \
                    .as_euler('XYZ', degrees=True)
            cur_frame[3 + channel_t_i * 3: 6 + channel_t_i * 3] = local_rotation

        motion_data[m_i] = cur_frame

    return motion_data
