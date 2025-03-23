from __future__ import division
import sim
import pybullet as p
import random
import numpy as np
import math
import time
import threading
MAX_ITERS = 10000
delta_q = 0.1

def visualize_path(q_1, q_2, env, color=[0, 1, 0]):
    env.set_joint_positions(q_1)
    point_1 = list(p.getLinkState(env.robot_body_id, 6)[0])
    env.set_joint_positions(q_2)
    point_2 = list(p.getLinkState(env.robot_body_id, 6)[0])
    p.addUserDebugLine(point_1, point_2, color, 1.5)

def rrt(q_init, q_goal, MAX_ITERS, delta_q, steer_goal_p, env, distance=0.12):
    V, E = [q_init], []
    path, found = [], False
    for i in range(MAX_ITERS):
        q_rand = semi_random_sample(steer_goal_p, q_goal)
        q_nearest = nearest(V, q_rand)
        q_new = steer(q_nearest, q_rand, delta_q)
        if not env.check_collision(q_new, 0.18):
            if q_new not in V:
                V.append(q_new)
            if (q_nearest, q_new) not in E:
                E.append((q_nearest, q_new))
                visualize_path(q_nearest, q_new, env)
            if get_euclidean_distance(q_goal, q_new) < delta_q:
                V.append(q_goal)
                E.append((q_new, q_goal))
                visualize_path(q_new, q_goal, env)
                found = True
                break

    if found:
        current_q = q_goal
        path.append(current_q)
        while current_q != q_init:
            for edge in E:
                if edge[1] == current_q:
                    current_q = edge[0]
                    path.append(edge[0])
        path.reverse()
        return path
    else:
        return None

def execute_path_with_obstacle_awareness(env, normal_path, elevated_path, switch_distance=0.5):
    """
    Thực thi đường đi cao nhất để tránh vật cản ở vị trí cao nhất
    
    Args:
        env: Môi trường mô phỏng
        normal_path: Đường đi khi vật cản ở vị trí thấp (không sử dụng)
        elevated_path: Đường đi khi vật cản ở vị trí cao
        switch_distance: Khoảng cách kích hoạt chuyển đổi đường đi (không sử dụng)
    """
    if not elevated_path:
        print("Không có đường đi ở vị trí cao, không thể thực thi")
        return
    
    print(f"Bắt đầu thực thi đường đi ở vị trí cao nhất...")
    print(f"Quỹ đạo cao: {len(elevated_path)} điểm")
    
    # Chỉ hiển thị đường đi cao hơn màu xanh dương
    for i in range(len(elevated_path) - 1):
        visualize_path(elevated_path[i], elevated_path[i+1], env, color=[0, 0, 1])
    
    # Đặt robot về vị trí bắt đầu
    env.set_joint_positions(env.robot_home_joint_config)
    
    # Thực thi đường đi cao
    print("Bắt đầu di chuyển theo quỹ đạo cao nhất")
    
    for i, config in enumerate(elevated_path):
        print(f"Di chuyển đến điểm {i+1}/{len(elevated_path)}")
        
        # Di chuyển đến điểm tiếp theo trên đường đi cao
        env.move_joints(config, speed=0.03)
        
        # Đợi một chút để đảm bảo robot hoàn thành di chuyển
        time.sleep(0.1)
    
    print("Hoàn thành thực thi đường đi cao nhất")

def find_closest_config(path, query_config):
    """
    Tìm cấu hình gần nhất với query_config trong path
    
    Args:
        path: Danh sách các cấu hình
        query_config: Cấu hình cần tìm điểm gần nhất
        
    Returns:
        Chỉ số của cấu hình gần nhất
    """
    min_distance = float('inf')
    closest_index = 0
    
    for i, config in enumerate(path):
        distance = get_euclidean_distance(config, query_config)
        if distance < min_distance:
            min_distance = distance
            closest_index = i
    
    return closest_index

def semi_random_sample(steer_goal_p, q_goal):
    prob = random.random()

    if prob < steer_goal_p:
        return q_goal
    else:
        # Uniform sample over reachable joint angles
        q_rand = [random.uniform(-np.pi, np.pi) for i in range(len(q_goal))]
    return q_rand

def get_euclidean_distance(q1, q2):
    distance = 0
    for i in range(len(q1)):
        distance += (q2[i] - q1[i])**2
    return math.sqrt(distance)

def nearest(V, q_rand):
    distance = float("inf")
    q_nearest = None
    for idx, v in enumerate(V):
        if get_euclidean_distance(q_rand, v) < distance:
            q_nearest = v
            distance = get_euclidean_distance(q_rand, v)
    return q_nearest

def steer(q_nearest, q_rand, delta_q):
    q_new = None
    if get_euclidean_distance(q_rand, q_nearest) <= delta_q:
        q_new = q_rand
    else:
        q_hat = [(q_rand[i] - q_nearest[i]) / get_euclidean_distance(q_rand, q_nearest) for i in range(len(q_rand))]
        q_new = [q_nearest[i] + q_hat[i] * delta_q for i in range(len(q_hat))]
    return q_new

def get_grasp_position_angle(object_id):
    position, grasp_angle = np.zeros((3, 1)), 0
    position, orientation = p.getBasePositionAndOrientation(object_id)
    grasp_angle = p.getEulerFromQuaternion(orientation)[2]
    return position, grasp_angle

def run():
    num_trials = 20  
    path_lengths = []
    env.load_gripper()
    passed = 0
    for trial in range(num_trials):
        # Đặt lại vật cản về vị trí ban đầu
        env.reset_moving_obstacle()
        
        object_id = env._objects_body_ids[0]
        position, grasp_angle = get_grasp_position_angle(object_id)
        grasp_success = env.execute_grasp(position, grasp_angle)
        if grasp_success:
            # Đặt vật cản ở vị trí cao nhất
            env.set_obstacle_highest_position()
            
            # Tạo đường đi với vật cản ở vị trí cao nhất
            print("Tạo đường đi với vật cản ở vị trí cao nhất...")
            elevated_path = rrt(env.robot_home_joint_config,
                              env.robot_goal_joint_config, MAX_ITERS, delta_q, 0.5, env)
            
            if elevated_path is None:
                print("Không thể tìm đường đi với vật cản ở vị trí cao nhất. Bỏ qua lượt này.")
                path_lengths.append(None)
                continue
            
            # Thực thi đường đi cao nhất
            execute_path_with_obstacle_awareness(env, None, elevated_path)
            
            print("Path executed. Dropping the object")
            env.open_gripper()
            env.step_simulation(num_steps=5)
            env.close_gripper()
            
            # Đưa robot về vị trí ban đầu
            env.robot_go_home(speed=0.1)
            
            p.removeAllUserDebugItems()
        env.robot_go_home()
        object_pos, _ = p.getBasePositionAndOrientation(object_id)
        if object_pos[0] >= -0.8 and object_pos[0] <= -0.2 and\
            object_pos[1] >= -0.3 and object_pos[1] <= 0.3 and\
            object_pos[2] <= 0.2:
            passed += 1
        env.reset_objects()

def draw():
    print("Starting draw function")
    line_ids = [None, None, None]
    current_object_id = None
    current_obstacle_id = None
    
    def get_distance(a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)
    
    while True:
        try:
            if len(env._objects_body_ids) == 0 or len(env.obstacles) == 0:
                print("No objects or obstacles found, waiting...")
                time.sleep(0.1)
                continue
            object_id = env._objects_body_ids[0]
            obstacles_id = env.obstacles[0]
            if object_id != current_object_id or obstacles_id != current_obstacle_id:
                line_ids = [None, None, None]
                current_object_id = object_id
                current_obstacle_id = obstacles_id
                print(f"Detected object or obstacles change. Updated object_id: {object_id}, obstacle_id: {obstacles_id}")
            try:
                p.getBodyInfo(object_id)
                p.getBodyInfo(obstacles_id)
            except p.error as e:
                print(f"Body ID error: {e}")
                time.sleep(0.1)
                continue
            getlink1 = p.getLinkState(object_id, 0)[0]
            getlink2 = p.getLinkState(object_id, 1)[0]
            midpoint = np.add(getlink1, getlink2) / 2
            closest_points = p.getClosestPoints(obstacles_id, object_id, 100)
            if not closest_points:
                print("No closest points found")
                a = getlink1
            else:
                a = closest_points[0][5]
            distance = get_distance(midpoint, a)
            print(f"Distance between midpoint and closest point: {distance}")
            lines_to_draw = [
                (getlink1, a, [1, 0, 0]),
                (getlink2, a, [1, 0, 0]),
                (midpoint, a, [0, 1, 0])
            ]
            for i, (start, end, color) in enumerate(lines_to_draw):
                if line_ids[i] is None:
                    line_ids[i] = p.addUserDebugLine(start, end, lineColorRGB=color, lineWidth=2)
                else:
                    p.addUserDebugLine(start, end, lineColorRGB=color, lineWidth=2, replaceItemUniqueId=line_ids[i])
            
        except IndexError as e:
            print(f"IndexError: {e}. Possible object or obstacle indices out of range. Retrying...")
            line_ids = [None, None, None]
            current_object_id = None
            current_obstacle_id = None
        except Exception as e:
            print(f"Exception in draw: {e}")
        time.sleep(0.1)

if __name__ == "__main__":
    random.seed(5)
    object_shapes = [
        "assets/objects/rod.urdf",
    ]
    env = sim.PyBulletSim(object_shapes = object_shapes)
    thread1 = threading.Thread(target=run)
    thread2 = threading.Thread(target=draw)
    thread1.start()
    thread2.start()  # Bỏ comment dòng này để chạy thread2
    