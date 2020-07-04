import sys
sys.path.append('..')
import unittest
import numpy as np
from ddt import ddt, data, unpack
from anytree import AnyNode as Node

from src.PW import Expand, InitializeChildren, ScoreChild
@ddt
class TestMCTS(unittest.TestCase):
    def setUp(self):
        self.action_space = [-1, 1]
        self.num_action_space = len(self.action_space)
        self.uniformActionPrior = {action : 1/self.num_action_space for action in self.action_space}
        self.transition = lambda state, action: state+action
        self.getActionPrior = lambda state : self.uniformActionPrior
        self.initializeChildren = InitializeChildren(
        self.action_space, self.transition, self.getActionPrior)
        self.isTerminal = lambda state: state ==7
        self.expand = Expand(self.isTerminal, self.initializeChildren)
        self.alpha = 0.5
        self.c_init = 0
        self.c_base = 1
        self.scoreChild = ScoreChild(self.c_init, self.c_base)
    @data((3, True, [-1, 1]), (0, True, [-1, 1]), (7, False, None))
    @unpack
    def testExpand(self, state, has_children, child_states):
        leaf_node = Node(id={"state": state}, numVisited=1,
                         sumValue=1, actionPrior=0.5, isExpanded=False)
        expanded_node = self.expand(leaf_node)
        calc_has_children = (len(expanded_node.children) != 0)
        self.assertEqual(has_children, calc_has_children)

        for child_index, child in enumerate(expanded_node.children):
            cal_child_state = list(child.id.values())[0]
            gt_child_state = child_states[child_index]
            self.assertEqual(gt_child_state, cal_child_state)
    @data((0, 1, 0, 1, 0), (1, 1, 0, 1, np.log(3)/2), (1, 1, 1, 1, 1+np.log(3)/2))
    @unpack
    def testCalculateScore(self, state_visit_number, state_action_visit_number, sumValue, actionPrior, groundtruth_score):
        curr_node = Node(numVisited = state_visit_number)
        actionChild = Node(numVisited = state_action_visit_number, sumValue = sumValue, actionPrior = actionPrior)
        actionNodeChild = Node(parent = actionChild, numVisited = 1, sumValue = sumValue)
        score = self.scoreChild(curr_node, actionChild)
        self.assertEqual(score, groundtruth_score)

    @data((4, 3, 0.125), (3, 4, 0.25))
    @unpack 
    def testRolloutWithoutHeuristic(self, max_rollout_step, init_state, gt_sumValue):
        max_iteration = 1000

        target_state = 6
        isTerminal = Terminal(target_state)

        catch_reward = 1
        step_penalty = 0
        reward_func = RewardFunction(step_penalty, catch_reward, isTerminal)
        rolloutHeuristic = lambda state: 0

        rollout_policy = lambda state: np.random.choice(self.action_space)
        leaf_node = Node(id={1: init_state}, numVisited=1, sumValue=0, actionPrior=self.default_actionPrior, isExpanded=True)
        rollout = RollOut(rollout_policy, max_rollout_step, self.transition, reward_func, isTerminal, rolloutHeuristic)
        stored_reward = []
        for curr_iter in range(max_iteration):
            stored_reward.append(rollout(leaf_node))
        
        calc_sumValue = np.mean(stored_reward)

        self.assertAlmostEqual(gt_sumValue, calc_sumValue, places=1)


if __name__ == "__main__":
    unittest.main()
