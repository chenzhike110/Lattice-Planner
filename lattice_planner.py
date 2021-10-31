from matplotlib import pyplot as plt
import numpy as np
import math
from util import get_lookup_table, search_nearest_one_from_lookuptable, calc_spline_course
from cubic_spline_planner import Spline2D

class Lattice_planner:
    """
        Lattice Planner
    """
    def __init__(self, line1, line2, state0=[0,0,0], obstacle=[], sample_num=3, ds=2.0, vw=0.1) -> None:
        self.line1 = line1
        self.line2 = line2
        self.sample_num = sample_num
        self.ds = ds
        self.vw = vw
        self.car = state0
        self.obstacle = obstacle
        self.roadindex = []

    def generate_path(self, target_states, k0):
        # x, y, yaw, s, km, kf
        lookup_table = get_lookup_table()
        result = []

        for state in target_states:
            bestp = search_nearest_one_from_lookuptable(
                state[0], state[1], state[2], lookup_table)

            target = motion_model.State(x=state[0], y=state[1], yaw=state[2])
            init_p = np.array(
                [math.sqrt(state[0] ** 2 + state[1] ** 2), bestp[4], bestp[5]]).reshape(3, 1)

            x, y, yaw, p = planner.optimize_trajectory(target, k0, init_p)

            if x is not None:
                print("find good path")
                result.append(
                    [x[-1], y[-1], yaw[-1], float(p[0]), float(p[1]), float(p[2])])

        print("finish path generation")
        return result

    def lane_state_sampling(self, distance):
        self.states = []
        distance = min(distance, max(self.line1.s))
        for i in range(1, int(distance/self.ds)+1):
            x1, y1 = self.line1.calc_position(i*self.ds)
            x2, y2 = self.line2.calc_position(i*self.ds)
            yaw1 = self.line1.calc_yaw(i*self.ds)
            yaw2 = self.line2.calc_yaw(i*self.ds)

            self.states.append(self.uniform_sampling((x1,y1), (x2,y2)))
            # self.states.extend(self.calc_lane_states((x1+x2)/2.0, (yaw1+yaw2)/2.0, abs(y1-y2), self.vw, i*self.ds, self.sample_num))
        self.states = np.array(self.states)
    
    def uniform_sampling(self, pos1, pos2):
        states = []
        for i in range(self.sample_num):
            states.append((np.array(pos2)-np.array(pos1))*(i+1)/(self.sample_num+1) + np.array(pos1))
        return np.array(states)

    def plan(self, distance):
        self.lane_state_sampling(distance)
        pointNum = int(distance/self.ds)
        x1, y1 = self.line1.calc_position(pointNum*self.ds)
        x2, y2 = self.line2.calc_position(pointNum*self.ds)
        middle = (y1+y2)/2.0
        matrix = np.zeros(self.states.shape[:2])
        self.index_matrix = np.zeros(self.states.shape[:2])
        for i in range(matrix.shape[1]):
            matrix[-1, i] = abs(middle - self.states[-1, i, 1])
        for i in range(matrix.shape[0]-2, -1, -1):
            for j in range(matrix.shape[1]):
                tmp = []
                for index in range(matrix.shape[1]):
                    tmp.append(self.getCost(self.states[i, j], self.states[i+1, index])+matrix[i+1, index])
                matrix[i, j] = min(tmp)
                self.index_matrix[i, j] = tmp.index(min(tmp))
        print(matrix)
        print(self.index_matrix)
        # print(1)+
        self.roadindex.append(self.car[:2])
        selected = np.argmin(matrix[0])
        for i in range(matrix.shape[0]):
            self.roadindex.append(self.states[i,selected])
            selected = int(self.index_matrix[i,selected])
        self.roadindex = np.array(self.roadindex)
        self.road = Spline2D(self.roadindex[:,0], self.roadindex[:,1])

    def show(self):
        # draw road
        rx1, ry1, _, _, _ = calc_spline_course(self.line1)
        rx2, ry2, _, _, _ = calc_spline_course(self.line2)
        plt.plot(rx1, ry1, color='k')
        plt.plot(rx2, ry2, color='k')

        # draw samples
        plt.plot(self.states[:,:,0], self.states[:,:,1], 'o', color='r', markersize=1)
        
        # draw car
        carx, cary = self.getCar(self.car)
        plt.plot(carx, cary, color='b')
        for i in self.obstacle:
            carx, cary = self.getCar(i)
            plt.plot(carx, cary, color='darkgreen')

        # draw road
        for i in range(0, self.index_matrix.shape[0]-1):
            for j in range(self.index_matrix.shape[1]):
                plt.plot([self.states[i+1, int(self.index_matrix[i, j]), 0], self.states[i, j, 0]], [self.states[i+1, int(self.index_matrix[i, j]), 1], self.states[i, j, 1]] , color='r')
            # self.index_matrix

        # final road
        rx3, ry3, _, _, _ = calc_spline_course(self.road)
        plt.plot(rx3, ry3, color='b')

        plt.grid(True)
        plt.axis("equal")
        plt.show()

    def getCost(self, state1, state2):
        cost = np.linalg.norm(state1 - state2)
        offside = 0.5
        for i in self.obstacle:
            if (i[0] - state1[0] + offside) * (i[0] - state2[0] - offside) < 0:
                l1 = np.linalg.norm(state1 - state2)
                l2 = np.linalg.norm(i[:2] - state1)
                l3 = np.linalg.norm(i[:2] - state2)
                theta = math.acos((l2**2+l1**2-l3**2)/(2*l1*l2))
                distance = math.sin(theta)*l2
                if distance < self.vw*1.2:
                    cost += 9999
                else:
                    cost += 0.5/distance
        return cost

    def calc_lane_states(self, l_center, l_heading, l_width, v_width, d, nxy):
        """
        calc lane states

        :param l_center: lane lateral position
        :param l_heading:  lane heading
        :param l_width:  lane width
        :param v_width: vehicle width
        :param d: longitudinal position
        :param nxy: sampling number
        :return: state list
        """
        xc = d
        yc = l_center

        states = []
        for i in range(nxy):
            delta = -0.5 * (l_width - v_width) + \
                (l_width - v_width) * i / (nxy - 1)
            xf = xc - delta * math.sin(l_heading)
            yf = yc + delta * math.cos(l_heading)
            yawf = l_heading
            states.append([xf, yf, yawf])

        return states
    
    def getCar(self, car):
        car_length = 0.5
        car_width = 0.25
        car_new = np.array(car[:2])
        rMatrix = np.array([[math.cos(car[2]), -math.sin(car[2])], 
                            [math.sin(car[2]), math.cos(car[2])]])
        p1 = (rMatrix @ np.array([car_length/2.0, car_width/2.0]) + car_new).reshape(2,-1)
        p2 = (rMatrix @ np.array([-car_length/2.0, car_width/2.0]) + car_new).reshape(2,-1)
        p3 = (rMatrix @ np.array([-car_length/2.0, -car_width/2.0]) + car_new).reshape(2,-1)
        p4 = (rMatrix @ np.array([car_length/2.0, -car_width/2.0]) + car_new).reshape(2,-1)
        result = np.concatenate((p1,p2,p3,p4,p1), axis=1)
        return result[0], result[1]

if __name__ == "__main__":
    x = np.array([-5.0, -2.5, 0.0, 2.5, 5.0, 7.5])
    y = np.array([0.0, 0.0, 0.8, 1.2, 0.6, 0.0])
    y1 = y + np.ones(6)
    y2 = y - np.ones(6)
    middleLine = Spline2D(x, y)
    line1 = Spline2D(x, y1)
    line2 = Spline2D(x, y2)
    obstacle_Num = 5
    obstacle = []
    for i in range(5):
        x, y = middleLine.calc_position(max(middleLine.s) / (5+1) * (i+1))
        y += (np.random.random()-0.5)/0.7
        yaw = middleLine.calc_yaw(max(middleLine.s) / (5+1) * (i+1))
        obstacle.append([x,y,yaw])
    planner = Lattice_planner(line1, line2, [-5.0, 0.0, 0], obstacle)
    planner.plan(max(line1.s))
    planner.show()

# def lane_state_sampling_test1():
#     k0 = 0.0

#     l_center = 0.0
#     l_heading = np.deg2rad(0.0)
#     l_width = 3.0
#     v_width = 1.0
#     d = 3
#     nxy = 5
#     states = calc_lane_states(l_center, l_heading, l_width, v_width, d, nxy)
#     result = generate_path(states, k0)

#     if show_animation:
#         plt.close("all")

#     for table in result:
#         xc, yc, yawc = motion_model.generate_trajectory(
#             table[3], table[4], table[5], k0)

#         if show_animation:
#             plt.plot(xc, yc, "-r")

#     if show_animation:
#         plt.grid(True)
#         plt.axis("equal")
#         plt.show()