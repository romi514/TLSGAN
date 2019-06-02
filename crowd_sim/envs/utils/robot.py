from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
from crowd_nav.policy.frozen import Frozen


class TrajRobot(Agent):
	def __init__(self, config, section):
		super().__init__(config, section)

	def act(self, ob, test_policy = False):
		if self.policy is None:
			raise AttributeError('Policy attribute has to be set!')

		if test_policy:
			if not isinstance(self.test_policy,Frozen):
				ob = ob[-1]
			action = self.test_policy.predict(ob)
		else:
			if not isinstance(self.policy,Frozen):
				ob = ob[-1]
			action = self.policy.predict(ob)
		return action

	def copy_robot(self,robot):
		state = robot.get_full_state()
		self.set(state.px, state.py, state.gx, state.gy, state.vx, state.vy, state.theta, state.radius, state.v_pref)

	def set_test_policy(self,policy):
		self.test_policy = policy

class Robot(Agent):
	def __init__(self, config, section):
		super().__init__(config, section)

	def act(self, ob):
		if self.policy is None:
			raise AttributeError('Policy attribute has to be set!')

		state = JointState(self.get_full_state(), ob)
		action = self.policy.predict(state)

		return action