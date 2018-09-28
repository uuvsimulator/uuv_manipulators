#############################################
# Work in progress. Code not working yet!!! #
#############################################

#!/usr/bin/env python
# Copyright (c) 2016 The UUV Simulator Authors.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import rospy
import sys
import os
import numpy as np
import scipy.linalg

import PyKDL
from uuv_manipulators_control import CartesianController
import tf
import tf_conversions
import tf.transformations as trans
from geometry_msgs.msg import PoseStamped
from fiducial_msgs.msg import FiducialTransformArray
from sensor_msgs.msg import Joy
from uuv_manipulators_msgs.msg import ManDyn
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from uuv_gazebo_ros_plugins_msgs.srv import GetModelProperties


class SMCartesianController(CartesianController):
    """
    Sliding mode cartesian controller
    """

    LABEL = 'Sliding mode cartesian controller'
    def __init__(self):
        """
        Class constructor
        """
        CartesianController.__init__(self)
        # Retrieve the controller parameters from the parameter server
        Q_tag = '~Q'
        K_tag = '~K'
        lambda_tag = '~Lambda'
        T_tag = '~T'
        uuv_name_tag = '~uuv_name'
        arm_name_tag = '~arm_name'
        if not rospy.has_param(Q_tag):
            rospy.ROSException('Q gain matrix not available for tag=%s' % Q_tag)
        if not rospy.has_param(K_tag):
            rospy.ROSException('K gain matrix not available for tag=%s' % K_tag)
        if not rospy.has_param(lambda_tag):
            rospy.ROSException('Lambda gain matrix not available for tag=%s' % lambda_tag)
        if not rospy.has_param(T_tag):
            rospy.ROSException('T gain matrix not available for tag=%s' % T_tag)
        if not rospy.has_param(uuv_name_tag):
            rospy.ROSException('uuv name not available for tag=%s' % uuv_name_tag)
        if not rospy.has_param(arm_name_tag):
            rospy.ROSException('arm name not available for tag=%s' % arm_name_tag)

        # Initialization flag, to wait the end-effector get to the home position
        self._is_init = False

        self._timer_on = 0

        # Flag indicating that the script is running in the first loop
        self._is_first_loop = True

        # Last velocity reference in cartesian coordinates
        self._last_goal_vel = np.asmatrix(np.zeros(6)).T

        self._last_time = rospy.get_time()

        # Initialization of Sliding Variables
        # Sliding surface slope
        self._lambda = np.diagflat([rospy.get_param(lambda_tag)])
        # Robustness term multiplier
        self._Q = np.diagflat([rospy.get_param(Q_tag)])
        # PD multiplier
        self._K = np.diagflat([rospy.get_param(K_tag)])
        # hyperbolic tangent slope
        self._T = np.diagflat([rospy.get_param(T_tag)])

        # Gravitational matrix
        self._Gq = np.asmatrix(np.zeros(6)).T

        self._uuv_name = rospy.get_param(uuv_name_tag)
        self._arm_name = rospy.get_param(arm_name_tag)

        self._update_model_props()

        # Joystick commands
        self._joy_commands = Joy()
        self._joy_comm_states = Joy()

        # Known valve Y and Z coordinates with respect to marker
        self._is_delta_y_z = True
        self._valve_delta_y = 0.11    # Delta y coordinate from valve with respect to marker (after marker alignment to match manipulator base frame)
        self._valve_delta_z = 0.43    # Delta z coordinate from valve with respect to marker (after marker alignment to match manipulator base frame)

        self._combo = 0               # Automatic sequence for releasing the gripper after a valve is closed
        self._combo_open_time = rospy.get_time() # Define start time for combo gripper opening

        self._manip_base_vehicle_trans = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        self._manip_base_marker_trans = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        self._tag_id = None
        self._last_tag_id = None
        self._tag_id_count = 0
        self._norm = None
        self._min_norm_updated = False

        # Listener for obtaining Coordinate Transformations (tf)
        self._listener = tf.TransformListener()

        # Subscriber to the tag pose detection topic
        rospy.Subscriber('/fiducial_transforms', FiducialTransformArray, self._pose_min_norm_manip_base_marker_callback)

        # Joystick topic subscriber
        self._joy_sub = rospy.Subscriber('joy', Joy, self._joy_callback)

        # Joystick topic publisher
        self._joy_pub = rospy.Publisher('joy', Joy, queue_size=10)

        # Topic that receives the gravitational matrix of the manipulator
        self._mandyn_sub = rospy.Subscriber("/"+self._uuv_name+"/"+self._arm_name+"/"+"man_dyn", ManDyn, self._mandyn_callback)

        self._publisher_goal = rospy.Publisher('goal_semi_autonom', PoseStamped, queue_size=10)

        self._run()

    def _update(self):
        # Leave if ROS is not running or command is not valid
        if rospy.is_shutdown() or self._last_goal is None:
            return

        # Calculate the goal pose
        goal = self._get_goal()
        # Limit roll to avoid singularities
        if goal.M.GetRPY()[0] > 3:
            goal_r = 3
        elif goal.M.GetRPY()[0] < -3:
            goal_r = -3
        else:
            goal_r = goal.M.GetRPY()[0]
        goal.M = PyKDL.Rotation.RPY(goal_r, goal.M.GetRPY()[1], goal.M.GetRPY()[2])
        # Return to the initial roll angle (Open valve orientation)
        if np.array_equiv(np.nonzero(self._joy_commands.buttons[0:16]), np.array([3, 5])):
            goal.M = PyKDL.Rotation.RPY(0, goal.M.GetRPY()[1], goal.M.GetRPY()[2])
            self._joy_comm_states.buttons = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self._combo = 0
        # Go to closed valve roll orientation
        if np.array_equiv(np.nonzero(self._joy_commands.buttons[0:16]), np.array([0, 5])):
            goal.M = PyKDL.Rotation.RPY(1.8, goal.M.GetRPY()[1], goal.M.GetRPY()[2])
            self._joy_comm_states.buttons = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self._combo = 0
        # Combo - Sequential manipulator movements after closing a valve
        if np.array_equiv(np.nonzero(self._joy_commands.buttons[0:16]), np.array([4, 5])) and not self._combo:
            self._combo = 1
            self._combo_open_time = rospy.get_time()

        if self._combo:
            self._joy_comm_states = self._joy_commands
            if rospy.get_time() > self._combo_open_time + 0.5:
                # Withdraw arm and vehicle
                self._joy_comm_states.buttons = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                self._joy_comm_states.axes = [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]
                goal.p[0] = goal.p[0] - 0.005
            else:
                # Opens and start rolling the gripper
                self._joy_comm_states.buttons = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                goal.M = PyKDL.Rotation.RPY(0, goal.M.GetRPY()[1], goal.M.GetRPY()[2])
            if rospy.get_time() > self._combo_open_time + 2:
                # Close the gripper
                goal.p[0] = goal.p[0] + 0.005
                self._joy_comm_states.axes = self._joy_commands.axes
                self._joy_comm_states.buttons = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            if rospy.get_time() > self._combo_open_time + 5:
                # End of combo
                self._joy_comm_states.buttons = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                goal.M = PyKDL.Rotation.RPY(goal.M.GetRPY()[0], goal.M.GetRPY()[1], goal.M.GetRPY()[2])
                self._combo = 0
            self._joy_comm_states.header.stamp = rospy.Time.now()
            self._joy_pub.publish(self._joy_comm_states)

        # End-effector's pose
        ee_pose = self._arm_interface.get_ee_pose_as_frame()

        # if (self._min_norm_updated == True) and (self._tag_id > 0):
        if self._min_norm_updated == True:

            try:
                # Get 'estimated' marker frame wrt manipulator base frame in PyKDL rotation format
                m = self._manip_base_marker_trans
                manip_base_marker_rot = PyKDL.Rotation(m[0,0],m[0,1],m[0,2], m[1,0],m[1,1],m[1,2], m[2,0],m[2,1],m[2,2])
                # Get RPY (roll, pitch and yaw angles) between 'estimated' marker frame wrt manipulator base frame
                manip_base_marker_rpy = manip_base_marker_rot.GetRPY()
                # Semi-autonomous rotation uses: 1) roll angles from 'goal'; 2) pitch / yaw angles for keeping manipulator end-effector orthogonal to the marker
                semi_autonom_rot = PyKDL.Rotation.RPY(goal.M.GetRPY()[0], manip_base_marker_rpy[1], manip_base_marker_rpy[2])

                # Estimated distance from end-effector to panel
                gripper_ee_trans = trans.concatenate_matrices(trans.translation_matrix((0.17, 0, 0)), trans.quaternion_matrix(ee_pose.M.GetQuaternion()))
                ee_manip_base_trans = trans.concatenate_matrices(trans.concatenate_matrices(trans.translation_matrix((ee_pose.p[0], ee_pose.p[1], ee_pose.p[2])), trans.quaternion_matrix(ee_pose.M.GetQuaternion()), gripper_ee_trans))
                marker_gripper_trans = trans.concatenate_matrices(trans.inverse_matrix(self._manip_base_marker_trans), ee_manip_base_trans)
                panel_gripper_distance = np.abs(marker_gripper_trans[0, 3])
                # print '\n', 'gripper distance to panel: ', panel_gripper_distance.round(decimals=2), '\n'
                # print 'gripper y coordinate wrt active marker: ', marker_gripper_trans[1, 3].round(decimals=2), '\n'
                # print 'gripper z coordinate wrt active marker: ', marker_gripper_trans[2, 3].round(decimals=2), '\n'

                if panel_gripper_distance < 2 and panel_gripper_distance > 0.1:
                    distance_check = True
                else:
                    distance_check = False
                # Get 'estimated' relative y and z valve coordinates with respect to marker
                if self._is_delta_y_z and distance_check:
                    # What the y and z valve coordinates wrt to marker represent in terms of y and z position in manipulator base frame
                    y_z_goal = np.dot(self._manip_base_marker_trans, np.array([0, self._valve_delta_y, self._valve_delta_z, 1]))
                    # Correction in y
                    if goal.p[1] > y_z_goal[0,1]:
                        # if goal.p[1] - y_z_goal[0,1] > 0.05:
                        #     goal.p[1] -= 0.02
                        # else:
                        #     goal.p[1] -= 0.002
                        if goal.p[1] - y_z_goal[0,1] > 0.1:
                            goal.p[1] -= 0.04
                        else:
                            goal.p[1] -= 0.004
                    else:
                        if y_z_goal[0,1] - goal.p[1] > 0.1:
                            goal.p[1] += 0.04
                        else:
                            goal.p[1] += 0.004
                    # Correction in z
                    if goal.p[2] > y_z_goal[0,2]:
                        if goal.p[2] - y_z_goal[0,2] > 0.1:
                            goal.p[2] -= 0.04
                        else:
                            goal.p[2] -= 0.004
                    else:
                        if y_z_goal[0,2] - goal.p[2] > 0.1:
                            goal.p[2] += 0.04
                        else:
                            goal.p[2] += 0.004

                # Compose semi autonomous goal frame
                if distance_check:
                    goal_semi_autonom = PyKDL.Frame(semi_autonom_rot, goal.p)
                else:
                    goal_semi_autonom = PyKDL.Frame(goal.M, goal.p)
                self._min_norm_updated = False

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print 'Semi-autonomous reference for manipulator end-effector failed.'
                goal_semi_autonom = goal
                self._min_norm_updated = False
        else:
            goal_semi_autonom = goal

        msg = PoseStamped()
        msg.header.frame_id = "oberon7/base"
        msg.header.stamp = rospy.Time.now()
        msg.pose = tf_conversions.posemath.toMsg(goal_semi_autonom)
        self._publisher_goal.publish(msg)

        ######################################
        ### Sliding mode cartesian control ###
        ######################################

        # Calculate reference velocity
        time_step = rospy.get_time() - self._last_time
        self._last_time = rospy.get_time()
        if time_step > 0:
            goal_p_dot = (goal_semi_autonom.p - self._last_goal.p) / time_step
        else:
            goal_p_dot = (goal_semi_autonom.p - self._last_goal.p) / 0.01
        goal_vel = np.array([goal_p_dot[0], goal_p_dot[1], goal_p_dot[2], 0, 0, 0])
        # End-effector's pose
        ee_pose = self._arm_interface.get_ee_pose_as_frame()
        # End-effector's velocity
        ee_vel = self._arm_interface.get_ee_vel_as_kdl_twist()
        # Calculate pose error
        error_pos = PyKDL.diff(ee_pose, goal_semi_autonom)
        error_pose = np.array([error_pos[0], error_pos[1], error_pos[2], error_pos[3], error_pos[4], error_pos[5]]).reshape((6,1))
        # Calculate velocity error
        ee_velo = np.array([ee_vel[0], ee_vel[1], ee_vel[2], ee_vel[3], ee_vel[4], ee_vel[5]])
        error_velo = (goal_vel - ee_velo).reshape((6,1))
        # Calculate sliding Variable
        s = np.dot(self._lambda, error_pose) + error_velo
        # Calculate reference acceleration
        if time_step > 0:
            goal_acc = (goal_vel - self._last_goal_vel) / time_step
        else:
            goal_acc = (goal_vel - self._last_goal_vel) / 0.01
        self._last_goal_vel = goal_vel
        # Calculate inertia matrix
        Mq = 0
        for key in self._linkloads:
            Mq += self._arm_interface.jacobian_transpose(end_link=key) * self._linkinertias[key] * self._arm_interface.jacobian(end_link=key)

        # Use masses different from the ones of the real vehicle to test !!!!
        Mq = 1.1 * Mq   # Consider 10% error in the masses
        # Wrenches - Inertial term
        tau_inertia = np.dot(Mq, np.asmatrix(goal_acc).T + np.dot(self._lambda, error_velo) + np.dot(self._Q, np.tanh(np.dot(self._T, s))))
        # Wrenches - PD term
        tau_pd = np.dot(self._K, s)
        # Wrenches - Gravitational term
        tau_gq = self._Gq
        # Compute jacobian transpose
        JT = self._arm_interface.jacobian_transpose()
        # Total wrench for sliding mode controller
        # tau = JT * (tau_inertia + tau_pd + tau_gq)
        tau = JT * (tau_inertia + tau_pd)

        # Store current pose target
        self._last_goal = goal_semi_autonom

        self.publish_goal()
        self.publish_joint_efforts(tau)

    def _pose_min_norm_manip_base_marker_callback(self, msg):
        """Returns the pose of the closest marker wrt the manipulator base."""

        # Get vehicle frame wrt manipulator base frame
        if self._is_first_loop == True:
            (manip_base_vehicle_pos, manip_base_vehicle_quat) = self._listener.lookupTransform('oberon7/base', 'rexrov2/base_link', rospy.Time())
            self._manip_base_vehicle_trans = trans.concatenate_matrices(trans.translation_matrix(manip_base_vehicle_pos), trans.quaternion_matrix(manip_base_vehicle_quat))
            self._is_first_loop = False

        # min_norm = 1000.0
        # min_tag_id = None
        # min_manip_base_marker_trans = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        # tag_vector_length = len(msg.transforms)
        # for count in range(tag_vector_length):
        #     actual_tag_id = msg.transforms[count].fiducial_id
        #     # Get 'estimated' marker frame wrt vehicle frame
        #     (base_link_marker_pos, base_link_marker_quat) = self._listener.lookupTransform('rexrov2/base_link', 'fid'+str(actual_tag_id), rospy.Time())
        #     # Concatenate translation and rotation
        #     base_link_marker_trans = trans.concatenate_matrices(trans.translation_matrix(base_link_marker_pos), trans.quaternion_matrix(base_link_marker_quat))
        #     # Aligning frames - rotate -90 degrees about y axis, and then rotate 180 degrees about x, and then transform to vehicle coordinates
        #     base_link_marker_trans = trans.concatenate_matrices(base_link_marker_trans, np.matrix('1 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 1'), np.matrix('0 0 -1 0; 0 1 0 0; 1 0 0 0; 0 0 0 1'))
        #     # Get 'estimated' marker frame wrt manipulator base frame
        #     actual_manip_base_marker_trans = trans.concatenate_matrices(self._manip_base_vehicle_trans, base_link_marker_trans)
        #     # Compute euclidian norm between marker and manipulator base
        #     actual_norm = np.linalg.norm(actual_manip_base_marker_trans[0:3,3], 2)
        #     # Get pose and tag id corresponding to smallest norm between marker and manipulator base
        #     if actual_norm < min_norm:
        #         self._manip_base_marker_trans = actual_manip_base_marker_trans
        #         if actual_norm < 1.5:
        #             self._tag_id = actual_tag_id
        #         else:
        #             self._tag_id = 0
        #         self._norm = actual_norm
        #         min_norm = actual_norm
        # #     if (actual_norm < min_norm):
        # #         min_norm = actual_norm
        # #         min_tag_id = actual_tag_id
        # #         min_manip_base_marker_trans = actual_manip_base_marker_trans
        # #
        # # if (self._tag_id != min_tag_id) and (not self._timer_on):
        # #     self._timer_init = rospy.get_time()
        # #     self._timer_on = 1
        # # elif self._tag_id == min_tag_id:
        # #     self._timer_on = 0
        # #
        # # if self._timer_on and (rospy.get_time() - self._timer_init > 0.5):
        # #     print 'change marker'
        # #     self._manip_base_marker_trans = min_manip_base_marker_trans
        # #     self._tag_id = min_tag_id
        # #     self._timer_on = 0
        #
        # if msg.transforms:
        #     self._min_norm_updated = True
        #     # if self._tag_id != 0:
        #         # print 'tag id: ', self._tag_id, '\n'
        # else:
        #     self._min_norm_updated = False

        min_norm = 1000.0

        if msg.transforms:
            tag_vector_length = len(msg.transforms)
            for count in range(tag_vector_length):
                actual_tag_id = msg.transforms[count].fiducial_id
                # Get 'estimated' marker frame wrt vehicle frame
                (base_link_marker_pos, base_link_marker_quat) = self._listener.lookupTransform('rexrov2/base_link', 'fid'+str(actual_tag_id), rospy.Time())
                # Concatenate translation and rotation
                base_link_marker_trans = trans.concatenate_matrices(trans.translation_matrix(base_link_marker_pos), trans.quaternion_matrix(base_link_marker_quat))
                # Aligning frames - rotate -90 degrees about y axis, and then rotate 180 degrees about x, and then transform to vehicle coordinates
                base_link_marker_trans = trans.concatenate_matrices(base_link_marker_trans, np.matrix('1 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 1'), np.matrix('0 0 -1 0; 0 1 0 0; 1 0 0 0; 0 0 0 1'))
                # Get 'estimated' marker frame wrt manipulator base frame
                actual_manip_base_marker_trans = trans.concatenate_matrices(self._manip_base_vehicle_trans, base_link_marker_trans)
                # Compute euclidian norm between marker and manipulator base
                actual_norm = np.linalg.norm(actual_manip_base_marker_trans[0:3,3], 2)
                # Get pose and tag id corresponding to smallest norm between marker and manipulator base
                if actual_norm < min_norm:
                    self._manip_base_marker_trans = actual_manip_base_marker_trans
                    self._tag_id = actual_tag_id
                    self._norm = actual_norm
                    min_norm = actual_norm
            if self._tag_id == self._last_tag_id:
                self._tag_id_count += 1
            else:
                self._tag_id_count = 0
            self._last_tag_id = self._tag_id

            if (self._tag_id_count > 10) and (self._norm < 2):
                self._min_norm_updated = True
            else:
                self._min_norm_updated = False
        else:
            self._tag_id_count = 0
            self._min_norm_updated = False

    def _joy_callback(self, joy):
        """ Joystick callback function """

        self._joy_commands = joy

    # Update model properties
    def _update_model_props(self):
        rospy.wait_for_service("/"+self._uuv_name+"/get_model_properties")
        self._get_model_props = rospy.ServiceProxy("/"+self._uuv_name+"/get_model_properties", GetModelProperties)
        self._linkloads = dict()
        self._linkinertias = dict()
        hydromodel = self._get_model_props()
        rho = hydromodel.models[0].fluid_density
        g = 9.806
        for index, name in enumerate(hydromodel.link_names):
            if not 'base' in name:
                B = rho * g * hydromodel.models[index].volume
                I = hydromodel.models[index].inertia
                M = np.zeros((6,6))
                np.fill_diagonal(M, (I.m, I.m, I.m, I.ixx, I.iyy, I.izz))
                self._linkloads[name] = np.matrix([0, 0, -I.m + B, 0, 0, 0]).T
                self._linkinertias[name] = np.asmatrix(M)

    def _mandyn_callback(self, mandyn):
        self._Gq = np.asmatrix(mandyn.gravitational).T

if __name__ == '__main__':
    # Start the node
    node_name = os.path.splitext(os.path.basename(__file__))[0]
    rospy.init_node(node_name)
    rospy.loginfo('Starting [%s] node' % node_name)

    sm_controller = SMCartesianController()

    rospy.spin()
    rospy.loginfo('Shutting down [%s] node' % node_name)
