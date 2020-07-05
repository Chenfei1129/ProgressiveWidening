import sys
sys.path.append('..')
import unittest
import numpy as np
from ddt import ddt, data, unpack
from anytree import AnyNode as Node

from src.algorithm.pw import Expand, InitializeChildren, ScoreChild, RollOut, backup, RewardFunction, Terminal, SelectAction, SelectNextState
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
        self.default_actionPrior = 1/self.num_action_space
        self.selectAction = SelectAction(self.scoreChild)

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

    @data((5, [3, 4], [2, 1], [8, 9], [3, 2]))
    @unpack
    def testBackup(self, value, prev_sumValues, prev_visit_nums, new_sumValues, new_visit_nums):
        node_list = []
        for prev_sumValue, prev_visit_num in zip(prev_sumValues, prev_visit_nums):
            node_list.append(Node(id={1: 4}, numVisited=prev_visit_num,
                                  sumValue=prev_sumValue, actionPrior=0.5, isExpanded=False))

        backup(value, node_list)
        cal_sumValues = [node.sumValue for node in node_list]
        cal_visit_nums = [node.numVisited for node in node_list]
        
        self.assertTrue(np.all(cal_sumValues == new_sumValues))
        self.assertTrue(np.all(cal_visit_nums == new_visit_nums))

    @data((4, 3, 0.125), (3, 4, 0.25))
    @unpack 
    def testRolloutWithoutHeuristic(self, max_rollout_step, init_state, gt_sumValue):
        max_iteration = 1000

        
        isTerminal = lambda state: state ==6

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

    @data((4, 3, 1.875), (3, 4, 1.75))
    @unpack 
    def testRolloutWithHeuristic(self, max_rollout_step, init_state, gt_sumValue):
        max_iteration = 1000

        target_state = 6
        isTerminal = Terminal(target_state)

        catch_reward = 1
        step_penalty = 0
        reward_func = RewardFunction(step_penalty, catch_reward, isTerminal)
        rolloutHeuristic = lambda state: 2

        rollout_policy = lambda state: np.random.choice(self.action_space)
        leaf_node = Node(id={1: init_state}, numVisited=1, sumValue=0, actionPrior=self.default_actionPrior, isExpanded=True)
        rollout = RollOut(rollout_policy, max_rollout_step, self.transition, reward_func, isTerminal, rolloutHeuristic)
        stored_reward = []
        for curr_iter in range(max_iteration):
            stored_reward.append(rollout(leaf_node))
        
        calc_sumValue = np.mean(stored_reward)

        self.assertAlmostEqual(gt_sumValue, calc_sumValue, places=1)

    @data((1,2,3,3,4,3))
    @unpack
    def testSelectAction(self, state, action1, action2, nextState1, nextState2, groundTruthAction):
    	isTerminal = lambda state: state ==4
    	catch_reward = 1
    	step_penalty = 0
    	reward_func = RewardFunction(step_penalty, catch_reward, isTerminal)
        
    	stateNode = Node(id ={"state":state}, numVisited =1)
    	actionNode1 = Node (id ={"action":action1}, numVisited =1, parent = stateNode, actionPrior = 0.5)
    	#print(actionNode1.actionPrior)
    	actionNode2 = Node (id ={"action":action2}, numVisited =1, parent = stateNode, actionPrior = 0.5)
    	nextState1 = Node (id ={"state": nextState1}, numVisited =1, parent = actionNode1, sumValue = 0)
    	#print(nextState1.sumValue)
    	nextState2 = Node (id ={"state": nextState2}, numVisited =1, parent = actionNode2, sumValue = 1)
    	#print(nextState2.sumValue)
    	calc_selectActionNode = self.selectAction(stateNode)
    	calc_selectAction = list(calc_selectActionNode.id.values())[0]
    	self.assertEqual(calc_selectAction,groundTruthAction)

if __name__ == "__main__":
    unittest.main()
