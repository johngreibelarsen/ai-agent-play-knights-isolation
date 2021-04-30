import copy
import datetime
import math
import random
import time

from isolation import Isolation
from sample_players import DataPlayer

# TODO import from issolation
_WIDTH = 11
_HEIGHT = 9

# This is Alpha Beta Search
#class CustomPlayer_Alfa_Beta(DataPlayer):
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
    alpha_beta_exe_time = 0

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
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            depth_limit = 100
            try:
                for depth in range(1, depth_limit + 1):
                    start_time = time.perf_counter()
                    result = self.alpha_beta_search(state, depth)
                    end_time = time.perf_counter()
                    CustomPlayer.alpha_beta_exe_time += end_time - start_time
                    self.queue.put((result, (CustomPlayer.nodes, CustomPlayer.alpha_beta_exe_time, depth)))
            except Exception:  # At deeper levels (depth) we will experience time out exception - ignore
                pass

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
            CustomPlayer.nodes += 1

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

# This is Monte Carlo Tree Search
#class CustomPlayer(DataPlayer):
class CustomPlayer_MCTS(DataPlayer):
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
    alpha_beta_exe_time = 0
    max_moves = 90 # never seen plies go beyond mid 80


    def __init__(self, player_id):
        super().__init__(player_id)
        self.wins = {}
        self.plays = {}
        self.states = []
        self.depth_counter = 0
        self.depths = []


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
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            try:
                start_time = time.perf_counter()
                result = self.monte_carlo_tree_search(state, milli_sec=500)
                end_time = time.perf_counter()
                CustomPlayer.alpha_beta_exe_time += end_time - start_time
                self.queue.put((result, (CustomPlayer.nodes, CustomPlayer.alpha_beta_exe_time, self.depths)))
            except Exception as ecp:  # At deeper levels (depth) we will experience time out exception - ignore
                print("TIMEOUT EXCEPTION")
                pass

    def monte_carlo_tree_search(self, gameState, milli_sec):
        self.depth_counter = 0
        # state = self.states[-1]
        player = gameState.player()
        #print(f"Start GameState: {gameState}")
        # legal = self.board.legal_plays(self.states[:])

        # Bail out early if there is no real choice to be made.
        # if not legal:
        #     return
        # if len(legal) == 1:
        #     return legal[0]

        #for action in gameState.actions():
        #    print(f"gameState.board: {gameState.result(action).board}")

        if len(gameState.actions()) == 1:
            return gameState.actions()[0]

        counter = 0
        calculation_time = datetime.timedelta(milliseconds=milli_sec)
        begin = datetime.datetime.utcnow()
        #print(f"begin: {begin}, delta: {calculation_time}")
        while datetime.datetime.utcnow() - begin < calculation_time:
        #for i in range(24):
            #print(f"Simulation run: {counter}")
            self.run_search(gameState)
            counter += 1

        #print(f"Simulations run: {counter}")
        #action_states = [(action, gameState.result(action)) for action in gameState.actions()]
        action_states = [(action, gameState.result(action).board) for action in gameState.actions()]

        # Display the number of calls of `run_simulation` and the
        # time elapsed.
        #print(games, datetime.datetime.utcnow() - begin)

        # Pick the move with the highest percentage of wins.
        #player_id = self.player_id
        percent_wins, move = max(
            (self.wins.get((player, s), 0) / self.plays.get((player, s), 1), a) for a, s in action_states)
        #print(f"Percentage wins: {percent_wins}, move: {move}")
        #print(f"self.plays: {self.plays}")
        #print(f"self.wins: {self.wins}")

        #print(f"End GameState: {gameState}")
        #print("Maximum depth searched:", self.max_depth)
        self.depths.append(self.depth_counter) # for statistics only

        return move


    def run_search(self, gameState):
        # A bit of an optimization here, so we have a local
        # variable lookup instead of an attribute access each loop.
        plays, wins = self.plays, self.wins

        visited_states = set()
        #states_copy = self.states[:]
        copyState = copy.deepcopy(gameState)
        #print(f"Orig. gameState ID: {id(gameState)}, gameState: {gameState}")
        #print(f"Copy. gameState ID: {id(copyState)}, gameState: {copyState}")
        player = copyState.player()

        expand = True
        for t in range(1, self.max_moves + 1):
            CustomPlayer.nodes += 1
            # legal = self.board.legal_plays(states_copy)
            action_states = [(action, copyState.result(action).board) for action in copyState.actions()]

            if all(plays.get((player, s)) for a, s in action_states):
                #print("Calculating move")
                # If we have stats on all of the legal moves here, use them.
                log_total = math.log(sum(plays[(player, s)] for a, s in action_states))
                value, move, state = max(
                    ((wins[(player, s)] / plays[(player, s)]) +
                     2 * math.sqrt(log_total / plays[(player, s)]), a, s)
                    for a, s in action_states
                )
            else:
                # Otherwise, just make an arbitrary decision.
                #print("Random move")
                move, state = random.choice(action_states)
            #print(f"Move: {move}")
            #print(f"State: {state}")
            #states_copy.append(state)

            #print(f"self.plays: {plays}")
            # `player` here and below refers to the player
            # who moved into that particular state.
            if expand and (player, state) not in plays:
                expand = False
                plays[(player, state)] = 0
                wins[(player, state)] = 0
                if t > self.depth_counter:
                    self.depth_counter = t

            visited_states.add((player, state))
            #print(f"Plays: {plays}")
            #print(f"Wins: {wins}")
            #print(f"Visited states: {visited_states}")


            copyState = copyState.result(move)
            player = copyState.player()
            winner = -1
            #state_obj = Isolation(board=state, ply_count=self.player_id)
            #gameState = state_obj
            if copyState.terminal_test():
                value = copyState.utility(player)
                #print(f"*** We have a WINNER*** t={t}, utility={value}")
                if value > 0:
                    winner = player
                    #print("WINNER > 0")
                elif value < 0:
                    winner = 1 - player
                    #print("WINNER < 0")
                break
            #else: # for pure testing, to be deleted
                #if t == self.max_moves:
                    #print(f"*** NO WINNER*** ")


        for player, state in visited_states:
            if (player, state) not in plays:
                continue
            plays[(player, state)] += 1
            if player == winner:
                wins[(player, state)] += 1
