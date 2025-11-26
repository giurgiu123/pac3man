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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        foodList = newFood.asList()
        if foodList:
            minFoodDistance = min(manhattanDistance(newPos, food) for food in foodList)
            score += 1.0 / (minFoodDistance + 1)
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            distanceToGhost = manhattanDistance(newPos, ghostPos)
            if ghostState.scaredTimer > 0:
                score += 100.0 / (distanceToGhost + 1)
            else:
                if distanceToGhost < 2:
                    score -= 500
                else:
                    score -= (10.0 /
                              (distanceToGhost + 1))
        return score


def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"

        def minimax(agentIndex, depth, gameState):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            if agentIndex == 0:
                return max_value(agentIndex, depth, gameState)
            else:
                return min_value(agentIndex, depth, gameState)

        def max_value(agentIndex, depth, gameState):
            bestValue = float('-inf')
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                value = minimax(1, depth, successor)
                bestValue = max(bestValue, value)
            return bestValue

        def min_value(agentIndex, depth, gameState):
            bestValue = float('inf')
            numAgents = gameState.getNumAgents()
            nextAgent = agentIndex + 1

            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)

                # if last ghost, next is Pacman at next depth
                if nextAgent == numAgents:
                    bestValue = min(bestValue, minimax(0, depth + 1, successor))
                # otherwise, next ghost
                else:
                    bestValue = min(bestValue, minimax(nextAgent, depth, successor))
            return bestValue

        bestAction = None
        bestValue = float('-inf')

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = minimax(1, 0, successor)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction
        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        def alpha_beta(agentIndex, depth, gameState, alpha, beta):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            if agentIndex == 0:
                return max_value(agentIndex, depth, gameState, alpha, beta)
            else:
                return min_value(agentIndex, depth, gameState, alpha, beta)

        def max_value(agentIndex, depth, gameState, alpha, beta):
            v = float('-inf')
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                v = max(v, alpha_beta(1, depth, successor, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(agentIndex, depth, gameState, alpha, beta):
            v = float('inf')
            numAgents = gameState.getNumAgents()
            nextAgent = agentIndex + 1

            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)

                if nextAgent == numAgents:
                    v = min(v, alpha_beta(0, depth + 1, successor, alpha, beta))
                else:
                    v = min(v, alpha_beta(nextAgent, depth, successor, alpha, beta))

                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        bestAction = None
        v = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = alpha_beta(1, 0, successor, alpha, beta)

            if value > v:
                v = value
                bestAction = action

            if v > beta:
                return bestAction

            alpha = max(alpha, v)

        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction

