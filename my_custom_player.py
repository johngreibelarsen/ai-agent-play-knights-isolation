import copy
import datetime
import math
import random

from isolation.isolation import _WIDTH, _HEIGHT
from sample_players import DataPlayer


# This is Alpha Beta Search #
#class CustomPlayer_Alfa_Beta(DataPlayer):
class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.
    """

    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        """
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            depth_limit = 9
            try:
                for depth in range(1, depth_limit + 1):
                    self.queue.put(self.alpha_beta_search(state, depth))
            except Exception:  # At deeper levels (depth) we will experience time out exception - ignore
                pass

    def alpha_beta_search(self, gameState, depth_limit):
        """ Return the move along a branch of the game tree that
        has the best possible value. A move is a pair of coordinates
        in (column, row) order corresponding to a legal move for
        the searching player.
        """

        def min_value(gameState, alpha, beta, depth_limit):
            """ Return he utility value of the current game state for the specified
            player. The game has a utility of +inf if the player has won, a value of
            -inf if the player has lost, and a value of 0 otherwise.
            """

            if gameState.terminal_test():
                return gameState.utility(self.player_id)

            if depth_limit <= 0:
                #return baseline(gameState)
                #return baseline_avoid_borders(gameState)
                #return offensive_to_defensive(gameState, 3)
                #return offensive(gameState, 2)
                #return defensive(gameState, 2)
                #return defensive_to_offensive(gameState, 3)
                return aggresive_attack_then_aggresive_defend(gameState, 3)

            v = float("inf")
            for a in gameState.actions():
                v = min(v, max_value(gameState.result(a), alpha, beta, depth_limit - 1))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        def max_value(gameState, alpha, beta, depth_limit):
            """ Return he utility value of the current game state for the specified
            player. The game has a utility of +inf if the player has won, a value of
            -inf if the player has lost, and a value of 0 otherwise.
            """
            if gameState.terminal_test():
                return gameState.utility(self.player_id)

            if depth_limit <= 0:
                #return baseline(gameState)
                #return baseline_avoid_borders(gameState)
                #return offensive_to_defensive(gameState, 3)
                #return offensive(gameState, 2)
                #return defensive(gameState, 2)
                #return defensive_to_offensive(gameState, 3)
                return aggresive_attack_then_aggresive_defend(gameState, 3)

            v = float("-inf")
            for a in gameState.actions():
                v = max(v, min_value(gameState.result(a), alpha, beta, depth_limit - 1))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

        def player_liberties(gameState):
            loc_player = gameState.locs[self.player_id]
            loc_opp = gameState.locs[1 - self.player_id]
            return (gameState.liberties(loc_player), gameState.liberties(loc_opp))

        def baseline(gameState):
            lib_player, lib_opp = player_liberties(gameState)
            return len(lib_player) - len(lib_opp)

        def avoid_borders(gameState):
            # Get the distance to the closest border
            loc_player = gameState.locs[self.player_id]
            player = (loc_player % (_WIDTH + 2), loc_player // (_WIDTH + 2))
            min_x, min_y = min(player[0], _WIDTH - 1 - player[0]), min(player[1], _HEIGHT - 1 - player[1])
            penalty = 0
            if min_x + min_y == 0:
                penalty = -10
            elif min_x + min_y == 1:
                penalty = -5
            return penalty

        def baseline_avoid_borders(gameState):
            return baseline(gameState) + avoid_borders(gameState)

        def offensive(gameState, weight):
            lib_player, lib_opp = player_liberties(gameState)
            return len(lib_player) - weight*len(lib_opp)

        def defensive(gameState, weight):
            lib_player, lib_opp = player_liberties(gameState)
            return weight*len(lib_player) - len(lib_opp)

        def offensive_to_defensive(gameState, weight):
            board_fields_occcupied = gameState.ply_count / (_WIDTH * _HEIGHT)
            if board_fields_occcupied <= 0.50:
                return offensive(gameState, weight)
            else:
                return defensive(gameState, weight)

        def defensive_to_offensive(gameState, weight):
            board_fields_occcupied = gameState.ply_count / (_WIDTH * _HEIGHT)
            if board_fields_occcupied <= 0.50:
                return defensive(gameState, weight)
            else:
                return offensive(gameState, weight)

        def aggresive_attack_then_aggresive_defend(gameState, weight):
            lib_player, lib_opp = player_liberties(gameState)
            board_fields_occcupied = gameState.ply_count / (_WIDTH * _HEIGHT)
            if board_fields_occcupied <= 0.3: # about 1/2 way through a typical game
                return len(lib_player) - weight*len(lib_opp)*(1 - board_fields_occcupied)
            else:
                return weight*len(lib_player)*(1 - board_fields_occcupied) - len(lib_opp)


        ### alpha_beta_search ###
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        # Important initialise best_move: makes testing the solution stable by avoiding returning None in a losing game
        best_move = gameState.actions()[0]
        for a in gameState.actions():
            v = min_value(gameState.result(a), alpha, beta, depth_limit - 1)
            alpha = max(alpha, v)
            if v > best_score:
                best_score = v
                best_move = a
        return best_move


# This is Monte Carlo Tree Search #
#class CustomPlayer(DataPlayer):
class CustomPlayer_MCTS(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.
    """
    max_moves = 90  # never seen plies go beyond mid 80

    def __init__(self, player_id):
        super().__init__(player_id)
        self.wins = {}
        self.plays = {}

    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.
        """
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            self.queue.put(self.monte_carlo_tree_search(state, milli_sec=130))

    def monte_carlo_tree_search(self, gameState, milli_sec):
        """ Causes the AI to calculate the best move from the
        current game state and return it. The combined algorithm is
        MCTS + UCB1 = UCT.
        """

        def run_search(gameState):
            """ Plays out a "random" game from the current position,
            then updates the statistics tables with the result.
            """
            plays, wins = self.plays, self.wins

            visited_states = set()
            copyState = copy.deepcopy(gameState)
            player = copyState.player()

            expand = True
            for t in range(1, self.max_moves + 1):
                action_states = [(action, copyState.result(action).board) for action in copyState.actions()]

                if all(plays.get((player, s)) for a, s in action_states):
                    # If we have stats on all of the legal moves here, use them.
                    log_total = math.log(sum(plays[(player, s)] for a, s in action_states))
                    value, move, state = max(
                        ((wins[(player, s)] / plays[(player, s)]) +
                         2 * math.sqrt(log_total / plays[(player, s)]), a, s)
                        for a, s in action_states
                    )
                else:
                    # just make an arbitrary decision.
                    move, state = random.choice(action_states)

                # `player` here and below refers to the player
                # who moved into that particular state.
                if expand and (player, state) not in plays:
                    expand = False
                    plays[(player, state)] = 0
                    wins[(player, state)] = 0

                visited_states.add((player, state))

                copyState = copyState.result(move)
                player = copyState.player()
                winner = -1
                if copyState.terminal_test():
                    value = copyState.utility(player)
                    if value > 0:
                        winner = player
                    elif value < 0:
                        winner = 1 - player
                    break

            for player, state in visited_states:
                if (player, state) not in plays:
                    continue
                plays[(player, state)] += 1
                if player == winner:
                    wins[(player, state)] += 1


        ### monte_carlo_tree_search ###
        player = gameState.player()

        # Bail out early if there is no real choice to be made.
        if len(gameState.actions()) == 1:
            return gameState.actions()[0]

        calculation_time = datetime.timedelta(milliseconds=milli_sec)
        begin = datetime.datetime.utcnow()
        while datetime.datetime.utcnow() - begin < calculation_time:
            run_search(gameState)

        action_states = [(action, gameState.result(action).board) for action in gameState.actions()]

        percent_wins, move = max(
            (self.wins.get((player, s), 0) / self.plays.get((player, s), 1), a) for a, s in action_states)

        return move
