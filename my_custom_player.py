import random
from sample_players import DataPlayer

# TODO import from issolation
_WIDTH = 11
_HEIGHT = 9

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """

    nodes = 0

    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        #print(f"Start of meethoodd: Nodes: {CustomPlayer.nodes}, self: {self}")
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            depth_limit = 5
            try:
                for depth in range(depth_limit, depth_limit + 1):
                    self.queue.put((self.alpha_beta_search(state, depth), CustomPlayer.nodes))
            except Exception:  # At deeper levels (depths) we will experience time out exception - ignore
                pass
        #print(f"End of method: Nodes: {CustomPlayer.nodes}, self: {self}")

    def alpha_beta_search(self, gameState, depth_limit):
        """ Return the move along a branch of the game tree that
        has the best possible value. A move is a pair of coordinates

        in (column, row) order corresponding to a legal move for
        the searching player.
        """

        def min_value(gameState, alpha, beta, depth_limit):
            CustomPlayer.nodes += 1

            """ Return he utility value of the current game state for the specified
            player. The game has a utility of +inf if the player has won, a value of
            -inf if the player has lost, and a value of 0 otherwise.
            """

            if gameState.terminal_test():
                return gameState.utility(self.player_id)

            if depth_limit <= 0:
                #return player_vs_opp_moves(gameState)
                #return offensive_to_defensive(gameState, 3)
                #return offensive(gameState, 3)
                #return defensive(gameState, 3)
                return defensive_to_offensive(gameState, 3)

            v = float("inf")
            for a in gameState.actions():
                v = min(v, max_value(gameState.result(a), alpha, beta, depth_limit - 1))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        def max_value(gameState, alpha, beta, depth_limit):
            CustomPlayer.nodes += 1

            """ Return he utility value of the current game state for the specified
            player. The game has a utility of +inf if the player has won, a value of
            -inf if the player has lost, and a value of 0 otherwise.
            """
            if gameState.terminal_test():
                return gameState.utility(self.player_id)

            if depth_limit <= 0:
                #return player_vs_opp_moves(gameState)
                #return offensive_to_defensive(gameState, 3)
                #return offensive(gameState, 3)
                #return defensive(gameState, 3)
                return defensive_to_offensive(gameState, 3)


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


        CustomPlayer.nodes += 1
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
