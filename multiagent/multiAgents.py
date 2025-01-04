# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
from anyio.abc import value
from jsonschema.exceptions import best_match

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        foods = newFood.asList()
        min_dist_to_food = -1 # smaller better
        for food in foods:
            cur_dist = util.manhattanDistance(newPos, food)
            if min_dist_to_food == -1:
                min_dist_to_food = cur_dist
            else:
                min_dist_to_food = min(min_dist_to_food, cur_dist)

        min_dist_to_ghost = -1
        for ghost in newGhostStates:
            ghost_pos = ghost.getPosition()
            cur_dist = util.manhattanDistance(newPos, ghost_pos)
            if min_dist_to_ghost == -1:
                min_dist_to_ghost = cur_dist
            else:
                min_dist_to_ghost = min(min_dist_to_ghost, cur_dist)

        val = 0
        if min_dist_to_ghost < 3:
            val = -100000

        ans = val + (1 / min_dist_to_food) + successorGameState.getScore()
        return ans

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        total_agents = gameState.getNumAgents()

        def minmax_recursion(agent, state, depth):
            if state.isWin() or state.isLose() or depth >= self.depth:
                return self.evaluationFunction(state), None
            next_agent = agent + 1
            next_depth = depth
            if next_agent == total_agents:
                next_agent = 0
                next_depth += 1
            if agent == 0:
                cur_value = float('-inf')
            else:
                cur_value = float('inf')
            best_action = None
            for action in state.getLegalActions(agent):
                successor_value = minmax_recursion(next_agent, state.generateSuccessor(agent, action), next_depth)[0]
                cur_value = max(cur_value, successor_value) if agent == 0 else min(cur_value, successor_value)
                if cur_value == successor_value:
                    best_action = action
            return cur_value, best_action

        return minmax_recursion(0, gameState, 0)[1]




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        total_agents = gameState.getNumAgents()
        BIG_NUM = float('inf')
        SMALL_NUM = float('-inf')

        def minmax_recursion(agent, state, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth >= self.depth:
                return self.evaluationFunction(state), None
            next_agent = agent + 1
            next_depth = depth
            if next_agent == total_agents:
                next_agent = 0
                next_depth += 1
            if agent == 0:
                cur_value = SMALL_NUM
            else:
                cur_value = BIG_NUM
            best_action = None
            for action in state.getLegalActions(agent):
                successor_value = minmax_recursion(next_agent, state.generateSuccessor(agent, action), next_depth, alpha, beta)[0]
                cur_value = max(cur_value, successor_value) if agent == 0 else min(cur_value, successor_value)
                if cur_value == successor_value:
                    best_action = action
                if agent == 0:
                    if successor_value > beta:
                        return successor_value, best_action
                    alpha = max(alpha, successor_value)
                else:
                    if successor_value < alpha:
                        return successor_value, best_action
                    beta = min(beta, successor_value)

            return cur_value, best_action

        return minmax_recursion(0, gameState, 0, SMALL_NUM, BIG_NUM)[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        total_agents = gameState.getNumAgents()
        SMALL_NUM = float('-inf')

        def minmax_recursion(agent, state, depth):
            if state.isWin() or state.isLose() or depth >= self.depth:
                return self.evaluationFunction(state), None
            next_agent = agent + 1
            next_depth = depth
            if next_agent == total_agents:
                next_agent = 0
                next_depth += 1
            ev_sum = 0
            cur_value = SMALL_NUM
            total = 0
            best_action = None
            for action in state.getLegalActions(agent):
                total += 1
                successor_value = minmax_recursion(next_agent, state.generateSuccessor(agent, action), next_depth)[0]
                cur_value = max(cur_value, successor_value)
                ev_sum += successor_value
                if cur_value == successor_value:
                    best_action = action
            if agent != 0:
                return ev_sum / total, None
            return cur_value, best_action

        return minmax_recursion(0, gameState, 0)[1]


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """

    pos = currentGameState.getPacmanPosition()
    _Food = currentGameState.getFood()
    ghost_states = currentGameState.getGhostStates()
    foods = _Food.asList()
    capsules = currentGameState.getCapsules()

    score = currentGameState.getScore()
    SUPER_CLOSE_GHOST_VALUE = 10000000
    CLOSE_GHOST_VALUE = 5
    FOOD_VALUE = 5
    CAPSULE_VALUE = 12

    for capsule in capsules:
        dist = util.manhattanDistance(pos, capsule)
        score += CAPSULE_VALUE / dist

    for food in foods:
        dist = util.manhattanDistance(pos, food)
        score += FOOD_VALUE / dist

    for ghost in ghost_states:
        ghost_pos = ghost.getPosition()
        coef = 1 if ghost.scaredTimer > 0 else -1
        dist = util.manhattanDistance(pos, ghost_pos)
        if dist <= 2:
            score += coef * SUPER_CLOSE_GHOST_VALUE / (dist + 1)
        elif dist <= 5:
            score -= CLOSE_GHOST_VALUE / dist

    return score

# Abbreviation
better = betterEvaluationFunction
