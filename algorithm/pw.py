import math
from anytree import AnyNode as Node
import numpy as np
#print(math.floor(3.9))

#print(pow(2,3))
class FindNextState1:
    def __init__(self, alpha, transitionFunction):
        self.alpha = alpha
        self.transitionFunction = transitionFunction

    def __call__(self, stateNode, actionNode):
        numAction = actionNode.numVisited
        k = math.floor(pow(numAction, self.alpha))
        state = stateNode.id
        action = actionNode.id
        nextState = self.transitionFunction(state, action)
        if k > len(actionNode.children):
            if nextState not in actionNode.children:
            	nextStateNode = Node(parent = actionNode, id = nextState, numVisited =1)## add action
        return actionNode.children

def transitionFunction(state, action):
    nextState = np.array(state)+np.array(action)
    return list(nextState)

#print(transitionFunction([0,1],[1,2]))
state = Node(id = [0,1], numVisited =1)
action = Node(id = [1,2], numVisited=1, parent = state)
#n2 = Node(id = [1,3], numVisited=1, parent = action)
n3 = Node(id = [2,4], parent = action)
g = Node(id = [2,5], parent = action)
fk = FindNextState1(0, transitionFunction)
print(fk(state, action))
'''
root = AnyNode(id="root")
k = AnyNode(parent = root)
print(RenderTree(root))
print(len(root.children))
if k in root.children:
    print("a")
'''