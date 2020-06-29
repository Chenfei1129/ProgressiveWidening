import math
from anytree import AnyNode as Node
import numpy as np

class FindNextState:
    def __init__(self, alpha, transitionFunction):
        self.alpha = alpha
        self.transitionFunction = transitionFunction

    def __call__(self, stateNode, actionNode):
        numAction = actionNode.numVisited
        k = math.ceil(pow(numAction, self.alpha))
        state = stateNode.id
        action = actionNode.id
        nextState = self.transitionFunction(state, action)
        if k > len(actionNode.children):
            if nextState not in actionNode.children:
            	nextStateNode = Node(parent = actionNode, id = nextState, numVisited =1)## add action
        nextStates = [action.id for action in actionNode.children]
        return nextStates

def transitionFunction(state, action):
    nextState = np.array(state)+np.array(action)
    return list(nextState)

class ScoreChild:
    def __init__(self, cInit, cBase):
        self.cInit = cInit
        self.cBase = cBase
    def __call__(self, stateNode, actionNode, actionNodeChild):
        stateActionVisitCount = actionNode.numVisited
        stateVisitCount = stateNode.numVisited
        actionPrior = actionNode.actionPrior
        if stateActionVisitCount == 0:
            uScore = np.inf
            qScore = 0 
        else:
            explorationRate = np.log((1 + stateVisitCount + self.cBase) / self.cBase) + self.cInit
            uScore = explorationRate * actionPrior * np.sqrt(stateVisitCount) / float(1 + stateActionVisitCount)#selfVisitCount is stateACtionVisitCount
            qScore = actionNodeChild.sumValue / stateActionVisitCount

        score = qScore + uScore
        return score    
