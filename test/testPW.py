import sys
import unittest
import numpy as np
from ddt import ddt, data, unpack
from anytree import AnyNode as Node
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from algorithm.pw import FindNextState1, transitionFunction
@ddt
class TestMCTS(unittest.TestCase):
    def setUp(self):
    	self.alpha = 0.5

    @data(([1,1],[1,0],[0,1],[1,1]))
    @unpack
    def testNextState(self, numVisted, state, action, nextSate):
    	findNextSate = FindNextState1(self.alpha, transitionFunction)

    	trueNextState = findNextSate(state, action)
    	self.assertAlmostEqual(trueNextState, nextSate, places=1) 

if __name__ == '__main__':
    unittest.main()

