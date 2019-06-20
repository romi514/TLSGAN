from crowd_sim.envs.utils.action import ActionXY
from crowd_nav.policy.multi_human_rl import MultiHumanRL

class DO_NOTHING(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'DO_NOTHING'
        self.device = None
        self.model = None
        self.kinematics = 'holonomic'

    def configure(self, config):
        self.set_common_parameters(config)

    def predict(self, states):
        
        return ActionXY(0,0)

