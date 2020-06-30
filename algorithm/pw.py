import math
from anytree import AnyNode as Node
import numpy as np

class FindNextState:
    def __init__(self, alpha, transitionFunction):
        self.alpha = alpha
        self.transitionFunction = transitionFunction

    def __call__(self, stateNode, actionNode):
        numVisitedAction = actionNode.numVisited
        k = math.ceil(pow(numVisitedAction, self.alpha))
        state = stateNode.id
        action = actionNode.id
        nextState = self.transitionFunction(state, action)
        if k > len(actionNode.children):
            if nextState not in actionNode.children:
            	nextStateNode = Node(parent = actionNode, id = nextState, numVisited =1)## add action
        childNodes = [action.id for action in actionNode.children]
        return childNodes

def transitionFunction(state, action):
    nextState = np.array(state)+np.array(action)
    return list(nextState)


class ScoreChild:
    def __init__(self, cInit, cBase):
        self.cInit = cInit
        self.cBase = cBase
    def __call__(self, stateNode, actionNode):
        stateActionVisitCount = actionNode.numVisited
        stateVisitCount = stateNode.numVisited
        actionPrior = actionNode.actionPrior
        if stateActionVisitCount == 0:
            uScore = np.inf
            qScore = 0 
        else:
            explorationRate = np.log((1 + stateVisitCount + self.cBase) / self.cBase) + self.cInit
            uScore = explorationRate * actionPrior * np.sqrt(stateVisitCount) / float(1 + stateActionVisitCount)#selfVisitCount is stateACtionVisitCount
            nextStateValues = [nextState.sumValue for nextState in actionNode.children]
            numNextStateVisits = [nextState.numVisited/actionNode.numVisited for nextState in actionNode.children]
            sumNextStateValues = sum(np.multiply(nextStateValues, numNextStateVisits))
            qScore = sumNextStateValues / stateActionVisitCount

        score = qScore + uScore
        return score   

class SelectAction:
    def __init__(self, calculateScore):
        self.calculateScore = calculateScore

    def __call__(self, stateNode, actionNode):
        scores = [self.calculateScore(stateNode, actionNode) for actionNode in stateNode.children]
        maxIndex = np.argwhere(scores == np.max(scores)).flatten()
        selectedChildIndex = np.random.choice(maxIndex)
        selectedAction = stateNode.children[selectedChildIndex]
        return selectedAction

class SelectNextState:
    def __init__(self, selectAction):
        self.selectAction = selectAction

    def __call__(self, stateNode, actionNode):
        selectedAction = self.selectAction(stateNode, actionNode)
        nextPossibleState = selectedAction.children
        numNextStateVisits = [nextState.numVisited/actionNode.numVisited for nextState in actionNode.children]
        nextState = np.random.choice(nextPossibleState,1,numNextStateVisits)
        return nextState

class InitializeChildren:
    def __init__(self, actionSpace, transition, getActionPrior):
        self.actionSpace = actionSpace
        self.transition = transition
        self.getActionPrior = getActionPrior

    def __call__(self, node):
        state = list(node.id)
        initActionPrior = self.getActionPrior(state)

        for action in self.actionSpace:
            nextStates = self.transition(state, action)
            Node(parent=node, id=action, numVisited=0, sumValue=0, actionPrior=initActionPrior[action]
                 ) 
            for nextState in nextStates:
                Node(parent=action, id=nextState, numVisited=0, sumValue=0
                 )

        return node

