import numpy as np
from anytree import AnyNode as Node

class InitializeChildren:
    def __init__(self, actionSpace, transition, getActionPrior):
        self.actionSpace = actionSpace
        self.transition = transition
        self.getActionPrior = getActionPrior

    def __call__(self, node):
        state = list(node.id.values())[0]
        initActionPrior = self.getActionPrior(state)

        for action in self.actionSpace:
            nextState = self.transition(state, action)
            actionNode = Node(parent=node, id={"action": action}, numVisited=0, sumValue=0,actionPrior=initActionPrior[action])
            Node(parent=actionNode, id={"state": nextState}, numVisited=0, sumValue=0,
                 isExpanded=False)

        return node

class Expand:
    def __init__(self, isTerminal, initializeChildren):
        self.isTerminal = isTerminal
        self.initializeChildren = initializeChildren

    def __call__(self, leafNode):
        currentState = list(leafNode.id.values())[0]
        if not self.isTerminal(currentState):
            leafNode.isExpanded = True
            leafNode = self.initializeChildren(leafNode)

        return leafNode

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

    def __call__(self, stateNode):
        scores = [self.calculateScore(stateNode, actionNode) for actionNode in list(stateNode.children)]
        #print(scores)
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

class RollOut:
    def __init__(self, rolloutPolicy, maxRolloutStep, transitionFunction, rewardFunction, isTerminal, rolloutHeuristic):
        self.transitionFunction = transitionFunction
        self.rewardFunction = rewardFunction
        self.maxRolloutStep = maxRolloutStep
        self.rolloutPolicy = rolloutPolicy
        self.isTerminal = isTerminal
        self.rolloutHeuristic = rolloutHeuristic

    def __call__(self, leafNode):
        currentState = list(leafNode.id.values())[0]
        totalRewardForRollout = 0

        for rolloutStep in range(self.maxRolloutStep):
            action = self.rolloutPolicy(currentState)
            totalRewardForRollout += self.rewardFunction(currentState, action)
            if self.isTerminal(currentState):
                break
            nextState = self.transitionFunction(currentState, action)
            currentState = nextState

        heuristicReward = 0
        if not self.isTerminal(currentState):
            heuristicReward = self.rolloutHeuristic(currentState)
        totalRewardForRollout += heuristicReward

        return totalRewardForRollout

def backup(value, nodeList): #anytree lib
    for node in nodeList:
        node.sumValue += value
        node.numVisited += 1

class MCTS:
    def __init__(self, numSimulation, selectAction, selectNextState, expand, estimateValue, backup, outputDistribution):
        self.numSimulation = numSimulation
        self.selectAction = selectAction 
        self.selectNextState = selectNextState
        self.expand = expand
        self.estimateValue = estimateValue
        self.backup = backup
        self.outputDistribution = outputDistribution

    def __call__(self, currentState):
        root = Node(id={"state": currentState}, numVisited=0, sumValue=0, isExpanded=False)
        root = self.expand(root)

        for exploreStep in range(self.numSimulation):
            currentNode = root
            nodePath = [currentNode]

            while currentNode.isExpanded:
                actionNode = self.selectAction(currentNode)
                nextStateNode = self.selectNextState(currentNode, nextStateNode)
                nodePath.append(actionNode)
                nodePath.append(nextStateNode)
                currentNode = nextStateNode

            leafNode = self.expand(currentNode)
            value = self.estimateValue(leafNode)
            self.backup(value, nodePath)

        actionDistribution = self.outputDistribution(root)
        return actionDistribution

class RewardFunction:
    def __init__(self, step_penalty, catch_reward, isTerminal):
        self.step_penalty = step_penalty
        self.catch_reward = catch_reward
        self.isTerminal = isTerminal

    def __call__(self, state, action):
        if self.isTerminal(state):
            return self.catch_reward
        else:
            return self.step_penalty

class Terminal:
    def __init__(self, target_state):
        self.target_state = target_state

    def __call__(self, state):
        return state == self.target_state

