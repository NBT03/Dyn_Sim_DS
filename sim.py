import pybullet as p
import pybullet_data
import numpy as np
import time
import math
import os
from typing import Type, Dict, List, Optional, Tuple, Any, Set
import threading

class PyBulletSim:
    def __init__(self, use_random_objects=False, object_shapes=None, gui=True):
        self._workspace1_bounds = np.array([
            [-0.16, -0.17],
            [-0.55, -0.55],
            [0.78, 0.80]
        ])
        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._plane_id = p.loadURDF("assets/doosan/plane.urdf")
        p.setGravity(0, 0, -9.8)

        # load Doosan robot
        self.robot_body_id = p.loadURDF(
            "assets/doosan/doosan_origin.urdf", [0, 0, 0.8], p.getQuaternionFromEuler([0, 0, 0]))
        self._base_id = p.loadURDF(
            "assets/doosan/base_doosan.urdf", [0.75,0.3,0], p.getQuaternionFromEuler([0,0,np.pi]),useFixedBase=True)
        self._cabin_id = p.loadURDF(
            "assets/doosan/Cabin.urdf",[-0.75,-1,0], p.getQuaternionFromEuler([np.pi/2, 0, np.pi/2]),useFixedBase=True)
        self._gripper_body_id = None
        self.robot_end_effector_link_index = 6
        self._robot_tool_offset = [0, 0, 0]
        self._tool_tip_to_ee_joint = np.array([0, 0, 0.15])
        robot_joint_info = [p.getJointInfo(self.robot_body_id, i) for i in range(
            p.getNumJoints(self.robot_body_id))]
        self._robot_joint_indices = [
            x[0] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]
        self._joint_epsilon = 1e-3
        self.robot_home_joint_config = [np.pi*(-94.06/180),
                                        np.pi*(-13.65/180),
                                        np.pi*(101.87/180),
                                        np.pi*(1.48/180),
                                        np.pi * (89.39/180),
                                        np.pi * (3.73/180)]

        self.robot_goal_joint_config = [ np.pi * (94.06/180),
                                        np.pi * (30 / 180),
                                        np.pi * (60.87 / 180),
                                        np.pi * (1.48 / 180),
                                        np.pi * (89.39 / 180),
                                        np.pi * (3.73 / 180)]

        self.move_joints(self.robot_home_joint_config, speed=0.05)
        self._tote_id = p.loadURDF(
            "assets/tote/tote_bin.urdf",[-0.3,-0.35,0.80], p.getQuaternionFromEuler([np.pi/2, 0, 0]), useFixedBase=True)
        self._object_colors = get_tableau_palette()
        if object_shapes is not None:
            self._object_shapes = object_shapes
        else:
            self._object_shapes = [
                "assets/objects/cube.urdf",
                "assets/objects/rod.urdf",
                "assets/objects/custom.urdf",
            ]
        self._num_objects = len(self._object_shapes)
        self._object_shape_ids = [
            i % len(self._object_shapes) for i in range(self._num_objects)]
        self._objects_body_ids = []
        for i in range(self._num_objects):
            object_body_id = p.loadURDF(self._object_shapes[i], [-0.6,-0.35,0.05], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=False)
            self._objects_body_ids.append(object_body_id)
            p.changeVisualShape(object_body_id, -1, rgbaColor=[*self._object_colors[i], 1])
        self.reset_objects()
        self.obstacles = [
            # p.loadURDF('assets/obstacles/block.urdf',
            #            basePosition=[0.5, -0.6, 1.0],
            #            useFixedBase=True
            #            ),
            p.loadURDF('assets/obstacles/block.urdf',
                       basePosition=[0.35, -0.35, 0.85],
                       useFixedBase=False
                       ),
            # p.loadURDF('assets/obstacles/block.urdf',
            #            basePosition=[0.4, 0.4, 0.95],
            #            useFixedBase=True
            #            ),
            # p.loadURDF('assets/obstacles/block.urdf',
            #            basePosition=[-0.4, 0.5, 1.0],
            #            useFixedBase=True
            #            ),
        ]
        
        # Lưu ID của vật cản di chuyển
        self.moving_obstacle_id = self.obstacles[0]
        
        # Thay đổi màu để dễ nhận biết
        p.changeVisualShape(self.moving_obstacle_id, -1,) # Red color for moving obstacle
        
        # Vô hiệu hóa trọng lực cho vật cản
        p.changeDynamics(self.moving_obstacle_id, -1, mass=0)  # Mass=0 khiến vật không bị ảnh hưởng bởi trọng lực
        
        # Lưu vị trí ban đầu của vật cản
        self.obstacle_initial_position, self.obstacle_initial_orientation = p.getBasePositionAndOrientation(self.moving_obstacle_id)
        self.moving_obstacle_initial_pos = list(self.obstacle_initial_position)  # Thêm thuộc tính này
        
        # Tham số di chuyển cho vật cản
        self.obstacle_velocity = 0.00005  # Vận tốc cực kỳ chậm (giảm đi 6 lần)
        self.obstacle_direction = 1  # 1 cho hướng dương (lên), -1 cho hướng âm (xuống)
        
        # Giới hạn di chuyển (sai số 0.05 quanh vị trí ban đầu)
        self.obstacle_path_limit = [
            self.obstacle_initial_position[2] - 0.05,  # Giới hạn dưới
            self.obstacle_initial_position[2] + 0.18   # Giới hạn trên (tăng lên để dễ nhìn thấy sự khác biệt)
        ]

    def get_distance_to_obstacle(self, q_nearest, obstacle):
        self.set_joint_positions(q_nearest)
        end_effector_pos = p.getLinkState(self.robot_body_id, self.robot_end_effector_link_index)[0]
        obstacle_pos, _ = p.getBasePositionAndOrientation(obstacle)
        distance = np.linalg.norm(np.array(end_effector_pos) - np.array(obstacle_pos))
        return distance

    def get_obstacles_positions(self):
        obstacle_positions = []
        for obstacle_id in self._objects_body_ids:  # assuming you store the obstacle IDs in _objects_body_ids
            pos, _ = p.getBasePositionAndOrientation(obstacle_id)
            obstacle_positions.append(pos)
        return obstacle_positions

    def load_gripper(self):
        if self._gripper_body_id is not None:
            print("Gripper already loaded")
            return

        self._gripper_body_id = p.loadURDF("assets/gripper/robotiq_2f_85.urdf")
        p.resetBasePositionAndOrientation(self._gripper_body_id, [
                                          0.5, 0.1, 0.2], p.getQuaternionFromEuler([np.pi/2, 0, 0]))

        p.createConstraint(self.robot_body_id, self.robot_end_effector_link_index, self._gripper_body_id, -1, jointType=p.JOINT_FIXED, jointAxis=[
                           0, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=self._robot_tool_offset, childFrameOrientation=p.getQuaternionFromEuler([0, 0, 0]))

        for i in range(p.getNumJoints(self._gripper_body_id)):
            p.changeDynamics(self._gripper_body_id, i, lateralFriction=1.0, spinningFriction=1.0,
                             rollingFriction=0.0001, frictionAnchor=True)
        self.step_simulation(1e3)
    def move_joints(self, target_joint_state, speed=0.01):
        assert len(self._robot_joint_indices) == len(target_joint_state)
        p.setJointMotorControlArray(
            self.robot_body_id, self._robot_joint_indices,
            p.POSITION_CONTROL, target_joint_state,
            positionGains=speed * np.ones(len(self._robot_joint_indices))
        )

        timeout_t0 = time.time()
        while True:
            current_joint_state = [
                p.getJointState(self.robot_body_id, i)[0]
                for i in self._robot_joint_indices
            ]
            if all([
                np.abs(
                    current_joint_state[i] - target_joint_state[i]) < self._joint_epsilon
                for i in range(len(self._robot_joint_indices))
            ]):
                break
            if time.time()-timeout_t0 > 10:
                print(
                    "Timeout: robot is taking longer than 10s to reach the target joint state. Skipping...")
                p.setJointMotorControlArray(
                    self.robot_body_id, self._robot_joint_indices,
                    p.POSITION_CONTROL, self.robot_home_joint_config,
                    positionGains=np.ones(len(self._robot_joint_indices))
                )
                break
            self.step_simulation(1)
    def move_tool(self, position, orientation, speed=0.03):
        target_joint_state = np.zeros((6,))
        target_joint_state = p.calculateInverseKinematics(self.robot_body_id,
                                                          self.robot_end_effector_link_index,
                                                          position, orientation,
                                                          maxNumIterations=100, residualThreshold=1e-4)
        self.move_joints(target_joint_state)

    def robot_go_home(self, speed=0.1):
        self.move_joints(self.robot_home_joint_config, speed)
    def robot_go_after_grip(self,speed = 0.1):
        self.move_joints(self.robot_after_grip, speed)

    def close_gripper(self):
        p.setJointMotorControl2(
            self._gripper_body_id, 1, p.VELOCITY_CONTROL, targetVelocity=5, force=10000)
        self.step_simulation(4e2)

    def open_gripper(self):
        p.setJointMotorControl2(
            self._gripper_body_id, 1, p.VELOCITY_CONTROL, targetVelocity=-5, force=10000)
        self.step_simulation(4e2)

    def check_grasp_success(self):
        return p.getJointState(self._gripper_body_id, 1)[0] < 0.834 - 0.001

    def execute_grasp(self, grasp_position, grasp_angle):
        grasp_position = grasp_position + self._tool_tip_to_ee_joint
        gripper_orientation = p.getQuaternionFromEuler(
            [np.pi, 0, grasp_angle+np.pi/2])
        pre_grasp_position_over_bin = grasp_position+np.array([0, 0, 0.3])
        pre_grasp_position_over_object = grasp_position+np.array([0, 0, 0.1])
        post_grasp_position = grasp_position+np.array([0, 0, 0.3])
        grasp_success = False
        self.open_gripper()
        self.move_tool(pre_grasp_position_over_bin, None)
        self.move_tool(pre_grasp_position_over_object, gripper_orientation)
        self.move_tool(grasp_position, gripper_orientation)
        self.close_gripper()
        self.move_tool(post_grasp_position, None)
        self.robot_go_home(speed=0.03)
        grasp_success = self.check_grasp_success()
        return grasp_success
    def execute_place(self, place_angle=90.):
        gripper_orientation = p.getQuaternionFromEuler(
            [np.pi, 0, ((place_angle+180.) % 360.-180.)*np.pi/180.])
        place_position = np.array([0.4, -0.65, 0.4])
        self.move_tool(place_position, gripper_orientation, speed=0.01)
        self.open_gripper()
    def step_simulation(self, num_steps):
        for i in range(int(num_steps)):
            # Di chuyển vật cản
            self.update_moving_obstacle()
            
            p.stepSimulation()
            if self._gripper_body_id is not None:
                gripper_joint_positions = np.array([p.getJointState(self._gripper_body_id, i)[
                                                0] for i in range(p.getNumJoints(self._gripper_body_id))])
                p.setJointMotorControlArray(
                    self._gripper_body_id, [6, 3, 8, 5, 10], p.POSITION_CONTROL,
                    [
                        gripper_joint_positions[1], -gripper_joint_positions[1],
                        -gripper_joint_positions[1], gripper_joint_positions[1],
                        gripper_joint_positions[1]
                    ],
                    positionGains=np.ones(5)
                )
                
    def update_moving_obstacle(self):
        # Kiểm tra nếu moving_obstacle_id tồn tại
        if not hasattr(self, 'moving_obstacle_id') or not hasattr(self, 'obstacle_initial_position'):
            return
            
        # Lấy vị trí hiện tại của vật cản
        pos, orn = p.getBasePositionAndOrientation(self.moving_obstacle_id)
        
        # Cập nhật vị trí dựa trên vận tốc và hướng (theo trục Z)
        new_z = pos[2] + self.obstacle_velocity * self.obstacle_direction
        
        # Kiểm tra nếu đạt đến giới hạn và cần thay đổi hướng
        if new_z > self.obstacle_path_limit[1]:
            self.obstacle_direction = -1
            new_z = self.obstacle_path_limit[1]
        elif new_z < self.obstacle_path_limit[0]:
            self.obstacle_direction = 1
            new_z = self.obstacle_path_limit[0]
        
        # Đặt vị trí mới (giữ nguyên X và Y, chỉ thay đổi Z)
        p.resetBasePositionAndOrientation(
            self.moving_obstacle_id, 
            [pos[0], pos[1], new_z], 
            orn
        )

    def reset_objects(self):
        for object_body_id in self._objects_body_ids:
            random_position = [-0.163733, -0.46024903,0.92727434]
            random_orientation = [-0.41378682, -0.47447575, 0.07145692]
            p.resetBasePositionAndOrientation(
                object_body_id, random_position, p.getQuaternionFromEuler(random_orientation))
        self.step_simulation(2e2)
    def set_joint_positions(self, values):
        assert len(self._robot_joint_indices) == len(values)
        for joint, value in zip(self._robot_joint_indices, values):
            p.resetJointState(self.robot_body_id, joint, value)

    def check_collision(self, q, distance=0.18):
        self.set_joint_positions(q)
        for obstacle_id in self.obstacles:
            closest_points = p.getClosestPoints(
                self.robot_body_id, obstacle_id, distance)
            if closest_points is not None and len(closest_points) != 0:
                return True
        return False
        
    def get_moving_obstacle_info(self):
        """
        Trả về thông tin về vật cản di chuyển, bao gồm vị trí, kích thước và giới hạn Z
        """
        # Kiểm tra cả hai cách triển khai
        if hasattr(self, 'moving_obstacle_id'):
            # Lấy vị trí hiện tại của vật cản
            pos, orn = p.getBasePositionAndOrientation(self.moving_obstacle_id)
            
            # Lấy kích thước vật cản (AABB) nếu có thể
            try:
                aabb_min, aabb_max = p.getAABB(self.moving_obstacle_id)
                size = [aabb_max[i] - aabb_min[i] for i in range(3)]
            except:
                size = [0.2, 0.2, 0.2]  # Kích thước mặc định nếu không thể lấy AABB
            
            # Xác định các giới hạn Z
            if hasattr(self, 'obstacle_path_limit'):
                min_z = self.obstacle_path_limit[0]
                max_z = self.obstacle_path_limit[1]
            elif hasattr(self, 'min_z') and hasattr(self, 'max_z'):
                min_z = self.min_z
                max_z = self.max_z
            else:
                # Giá trị mặc định nếu không có giới hạn được xác định
                min_z = pos[2] - 0.05
                max_z = pos[2] + 0.15
            
            # Lấy vị trí ban đầu
            if hasattr(self, 'obstacle_initial_position'):
                initial_pos = self.obstacle_initial_position
            elif hasattr(self, 'moving_obstacle_initial_pos'):
                initial_pos = self.moving_obstacle_initial_pos
            else:
                initial_pos = [pos[0], pos[1], min_z]  # Ước tính nếu không có
            
            return {
                'id': self.moving_obstacle_id,
                'position': pos,
                'size': size,
                'initial_z': initial_pos[2],
                'current_z': pos[2],
                'min_z': min_z,
                'max_z': max_z,
                'direction': self.obstacle_direction if hasattr(self, 'obstacle_direction') else self.direction_z if hasattr(self, 'direction_z') else 0
            }
        
        return None

    def reset_moving_obstacle(self):
        """
        Đặt lại vật cản về vị trí ban đầu
        """
        if hasattr(self, 'moving_obstacle_id'):
            if hasattr(self, 'obstacle_initial_position'):
                # Phiên bản 1
                p.resetBasePositionAndOrientation(
                    self.moving_obstacle_id,
                    self.obstacle_initial_position,
                    self.obstacle_initial_orientation
                )
                self.obstacle_direction = 1  # Đặt lại hướng di chuyển
                
                # Cập nhật thêm thuộc tính cho phiên bản 2
                if hasattr(self, 'moving_obstacle_current_pos'):
                    self.moving_obstacle_current_pos = list(self.obstacle_initial_position)
                    self.direction_z = 1
            elif hasattr(self, 'moving_obstacle_initial_pos'):
                # Phiên bản 2
                self.moving_obstacle_current_pos = self.moving_obstacle_initial_pos.copy()
                self.direction_z = 1
                
                # Áp dụng vị trí cho vật cản
                p.resetBasePositionAndOrientation(
                    self.moving_obstacle_id,
                    self.moving_obstacle_initial_pos,
                    [0, 0, 0, 1]
                )
            
            print("Đã đặt lại vật cản về vị trí ban đầu")

    def set_obstacle_highest_position(self):
        """
        Đặt vật cản ở vị trí cao nhất
        """
        if hasattr(self, 'moving_obstacle_id'):
            current_pos, current_orn = p.getBasePositionAndOrientation(self.moving_obstacle_id)
            
            # Xác định vị trí Z cao nhất
            if hasattr(self, 'obstacle_path_limit'):
                max_z = self.obstacle_path_limit[1]
            elif hasattr(self, 'max_z'):
                max_z = self.max_z
            else:
                max_z = current_pos[2] + 0.15
                
            # Tạo vị trí cao nhất
            highest_pos = list(current_pos)
            highest_pos[2] = max_z
            
            # Đặt vật cản
            p.resetBasePositionAndOrientation(
                self.moving_obstacle_id,
                highest_pos,
                current_orn
            )
            
            # Cập nhật hướng và vị trí hiện tại (nếu có)
            if hasattr(self, 'obstacle_direction'):
                self.obstacle_direction = -1
            if hasattr(self, 'direction_z'):
                self.direction_z = -1
            if hasattr(self, 'moving_obstacle_current_pos'):
                self.moving_obstacle_current_pos = highest_pos.copy()
                
            print(f"Đã đặt vật cản ở vị trí cao nhất: {highest_pos}")

    def add_moving_obstacles(self):
        # Thêm vật cản di chuyển
        obstacle_id = p.loadURDF(
            "cube_small.urdf",
            basePosition=[0.25, 0.0, 0.025],
            globalScaling=0.4
        )
        
        # Lưu trữ thông tin về vị trí ban đầu của vật cản
        self.moving_obstacle_id = obstacle_id
        self.moving_obstacle_initial_pos = [0.25, 0.0, 0.025]
        self.moving_obstacle_current_pos = self.moving_obstacle_initial_pos.copy()
        
        # Đặt màu cho vật cản (đỏ)
        p.changeVisualShape(obstacle_id, -1, rgbaColor=[1, 0, 0, 1])
        
        # Thông số cho chuyển động của vật cản
        self.velocity = 0.05  # Tốc độ di chuyển rất chậm
        self.direction_z = 1  # Hướng di chuyển ban đầu (lên)
        self.min_z = self.moving_obstacle_initial_pos[2] - 0.05  # Giới hạn dưới
        self.max_z = self.moving_obstacle_initial_pos[2] + 0.15  # Giới hạn trên
        
        print(f"Đã thêm vật cản di chuyển tại vị trí {self.moving_obstacle_initial_pos}")
        
        # Bắt đầu một luồng riêng để cập nhật vị trí của vật cản
        self.stop_obstacle_thread = False
        self.obstacle_thread = threading.Thread(target=self.update_obstacle_position)
        self.obstacle_thread.daemon = True
        self.obstacle_thread.start()

    def update_obstacle_position(self):
        """Cập nhật vị trí của vật cản di chuyển theo thời gian thực"""
        while not self.stop_obstacle_thread:
            if hasattr(self, 'moving_obstacle_id'):
                # Lấy vị trí và hướng hiện tại
                current_pos = self.moving_obstacle_current_pos
                
                # Tính toán vị trí mới trên trục Z
                new_z = current_pos[2] + self.velocity * self.direction_z
                
                # Kiểm tra và đảo hướng nếu đạt đến giới hạn
                if new_z >= self.max_z:
                    new_z = self.max_z
                    self.direction_z = -1
                elif new_z <= self.min_z:
                    new_z = self.min_z
                    self.direction_z = 1
                
                # Cập nhật vị trí
                new_pos = [current_pos[0], current_pos[1], new_z]
                self.moving_obstacle_current_pos = new_pos
                
                # Áp dụng vị trí mới cho vật cản
                p.resetBasePositionAndOrientation(
                    self.moving_obstacle_id,
                    new_pos,
                    [0, 0, 0, 1]
                )
            
            # Tạm dừng một khoảng thời gian ngắn
            

class SphereMarker:
    def __init__(self, position, radius=0.05, rgba_color=(1, 0, 0, 0.8), text=None, orientation=None, p_id=0):
        self.p_id = p_id
        position = np.array(position)
        vs_id = p.createVisualShape(
            p.GEOM_SPHERE, radius=radius, rgbaColor=rgba_color, physicsClientId=self.p_id)

        self.marker_id = p.createMultiBody(
            baseMass=0,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vs_id,
            basePosition=position,
            useMaximalCoordinates=False
        )

        self.debug_item_ids = list()
        if text is not None:
            self.debug_item_ids.append(
                p.addUserDebugText(text, position + radius)
            )

        if orientation is not None:
            # x axis
            axis_size = 2 * radius
            rotation_mat = np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3,3)

            # x axis
            x_end = np.array([[axis_size, 0, 0]]).transpose()
            x_end = np.matmul(rotation_mat, x_end)
            x_end = position + x_end[:, 0]
            self.debug_item_ids.append(
                p.addUserDebugLine(position, x_end, lineColorRGB=(1, 0, 0))
            )
            # y axis
            y_end = np.array([[0, axis_size, 0]]).transpose()
            y_end = np.matmul(rotation_mat, y_end)
            y_end = position + y_end[:, 0]
            self.debug_item_ids.append(
                p.addUserDebugLine(position, y_end, lineColorRGB=(0, 1, 0))
            )
            # z axis
            z_end = np.array([[0, 0, axis_size]]).transpose()
            z_end = np.matmul(rotation_mat, z_end)
            z_end = position + z_end[:, 0]
            self.debug_item_ids.append(
                p.addUserDebugLine(position, z_end, lineColorRGB=(0, 0, 1))
            )

    def __del__(self):
        p.removeBody(self.marker_id, physicsClientId=self.p_id)
        for debug_item_id in self.debug_item_ids:
            p.removeUserDebugItem(debug_item_id)


def get_tableau_palette():
    palette = np.array(
        [
            [89, 169, 79],  # green
            [156, 117, 95],  # brown
            [237, 201, 72],  # yellow
            [78, 121, 167],  # blue
            [255, 87, 89],  # red
            [242, 142, 43],  # orange
            [176, 122, 161],  # purple
            [255, 157, 167],  # pink
            [118, 183, 178],  # cyan
            [186, 176, 172]  # gray
        ],
        dtype=np.cfloat
    )
    return palette / 255.
