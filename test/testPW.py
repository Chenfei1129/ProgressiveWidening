import sys
import unittest
import numpy as np
from ddt import ddt, data, unpack
from anytree import AnyNode as Node
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from algorithm.pw import FindNextState, transitionFunction
@ddt
class TestMCTS(unittest.TestCase):
    def setUp(self):
    	self.alpha = 0.5

    @data(([1,1],[1,0],[0,1],[[1,1]]), ([1,1],[1,2],[0,1],[[1,3]]))
    @unpack
    def testNextState(self, nodeNumVisted, state, action, nextSate):
    	stateNode = Node(id = state, numVisited = nodeNumVisted[0])
    	actionNode = Node(id = action, numVisited = nodeNumVisted[1], parent = stateNode)
    	findNextSate = FindNextState(self.alpha, transitionFunction)
    	truthNextState = findNextSate(stateNode, actionNode)
    	self.assertAlmostEqual(truthNextState, nextSate)

    @data(([1,1],[1,0],[0,1], [[1,3]]), ([1,1],[1,2],[0,1],[[1,3]]))
    @unpack
    def testNextState2(self, nodeNumVisted, state, action, nextSate):
    	stateNode = Node(id = state, numVisited = nodeNumVisted[0])
    	actionNode = Node(id = action, numVisited = nodeNumVisted[1], parent = stateNode)
    	stateChild = Node(id = [1,3], parent = actionNode)
    	findNextSate = FindNextState(self.alpha, transitionFunction)
    	truthNextState = findNextSate(stateNode, actionNode)
    	self.assertAlmostEqual(truthNextState, nextSate)

    @data(([1,1],[1,0],[0,1], [[1,3], [1,5]]))
    @unpack
    def testNextState3(self, nodeNumVisted, state, action, nextSate):
    	stateNode = Node(id = state, numVisited = nodeNumVisted[0])
    	actionNode = Node(id = action, numVisited = nodeNumVisted[1], parent = stateNode)
    	stateChild = Node(id = [1,3], parent = actionNode)
    	stateChild2 = Node(id = [1,5], parent = actionNode)    	
    	findNextSate = FindNextState(self.alpha, transitionFunction)
    	truthNextState = findNextSate(stateNode, actionNode)
    	self.assertAlmostEqual(truthNextState, nextSate) 
        
    @data((0, 1, 0, 1, 0), (1, 1, 0, 1, np.log(3)/2), (1, 1, 1, 1, 1 + np.log(3)/2))
    @unpack
    def testCalculateScore(self, state_visit_number, state_action_visit_number, sumValue, actionPrior, groundtruth_score):
        curr_node = Node(numVisited = state_visit_number)
        actionChild = Node(numVisited = state_action_visit_number, sumValue = sumValue, actionPrior = actionPrior)
        actionNodeChild = Node(parent = actionChild, numVisited = 0, sumValue = sumValue)
        score = self.scoreChild(curr_node, actionChild, actionNodeChild)
        self.assertEqual(score, groundtruth_score)

if __name__ == '__main__':
    unittest.main()
