###############################################################################
#                    YOU DO NOT NEED TO MODIFY THIS FILE                      #
###############################################################################
import argparse
import logging
import math
import os
import random
import textwrap
import time
import statistics

from collections import namedtuple
from multiprocessing.pool import ThreadPool as Pool

from isolation import Isolation, Agent, play
from sample_players import RandomPlayer, GreedyPlayer, MinimaxPlayer
from my_custom_player import CustomPlayer

logger = logging.getLogger(__name__)

NUM_PROCS = 1
NUM_ROUNDS = 5  # number times to replicate the match; increase for higher confidence estimate
TIME_LIMIT = 20000  # number of milliseconds before timeout

TEST_AGENTS = {
    "RANDOM": Agent(RandomPlayer, "Random Agent"),
    "GREEDY": Agent(GreedyPlayer, "Greedy Agent"),
    "MINIMAX": Agent(MinimaxPlayer, "Minimax Agent"),
    "SELF": Agent(CustomPlayer, "Custom TestAgent")
}

Match = namedtuple("Match", "players initial_state time_limit match_id debug_flag")


def _run_matches(matches, name, num_processes=NUM_PROCS, debug=False):
    results = []
    pool = Pool(1) if debug else Pool(num_processes)
    #print("Running {} games:".format(len(matches)))
    for result in pool.imap_unordered(play, matches):
        print("+" if result[0].name == name else '-', end="")
        results.append(result)
    print()
    return results


def make_fair_matches(matches, results):
    new_matches = []
    for _, game_history, match_id, _, _, _, _ in results:
        if len(game_history) < 2:
            logger.warn(textwrap.dedent("""\
                Unable to duplicate match {}
                -- one of the players forfeit at the first move
                """.format(match_id)))
            continue
        match = matches[match_id]
        state = Isolation().result(game_history[0]).result(game_history[1])
        fair_match = Match(players=match.players[::-1],
                          initial_state=state,
                          time_limit=match.time_limit,
                          match_id=-match.match_id,
                          debug_flag=match.debug_flag)
        new_matches.append(fair_match)
    return new_matches


def play_matches(custom_agent, test_agent, cli_args):
    """ Play a specified number of rounds between two agents. Each round
    consists of two games, and each player plays as first player in one
    game and second player in the other. (This mitigates "unfair" games
    where the first or second player has an advantage.)

    If fair_matches is true, then the agents repeat every game they played,
    but the agents switch initiative and use their opponent's opening move.
    In some games, picking a winning move for the opening guarantees the
    player a victory. Playing "fair" matches this way will balance out the
    advantage of picking perfect openings (the player would win the first
    time, and then lose when their opponent uses that move against them).
    """
    matches = []
    for match_id in range(cli_args.rounds):
        state = Isolation()
        matches.append(Match(
            players=(test_agent, custom_agent),
            initial_state=state,
            time_limit=cli_args.time_limit,
            match_id=2 * match_id,
            debug_flag=cli_args.debug))
        matches.append(Match(
            players=(custom_agent, test_agent),
            initial_state=state,
            time_limit=cli_args.time_limit,
            match_id=2 * match_id + 1,
            debug_flag=cli_args.debug))

    # Run all matches -- must be done before fair matches in order to populate
    # the first move from each player; these moves are reused in the fair matches
    results = _run_matches(matches, custom_agent.name, cli_args.processes)
    if cli_args.fair_matches:
        _matches = make_fair_matches(matches, results)
        results.extend(_run_matches(_matches, custom_agent.name, cli_args.processes))
    wins = sum(int(r[0].name == custom_agent.name) for r in results)
    nodes_expanded = [result[3] for result in results]
    algo_exec_time = [result[4] for result in results]
    depths = [result[5] for result in results]
    plies = [result[6] for result in results]
    return wins, len(matches) * (1 + int(cli_args.fair_matches)), nodes_expanded, algo_exec_time, depths, plies

def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])

def main(args):
    from statistics import mean
    test_agent = TEST_AGENTS[args.opponent.upper()]
    custom_agent = Agent(CustomPlayer, "Custom Agent")
    start_time = time.perf_counter()
    wins, num_games, nodes_expanded, algo_exec_time, depths, plies = play_matches(custom_agent, test_agent, args)
    end_time = time.perf_counter()

    logger.info("Your agent won {:.1f}% of matches against {}".format(
       100. * wins / num_games, test_agent.name))
    print("Your agent won {:.1f}% of matches against {}".format(
       100. * wins / num_games, test_agent.name))
    print("Your agent took on average {} plies to complete, min: {}, max: {}".format(mean(plies), min(plies), max(plies)))
    print("Your agent expanded on average {} nodes, min: {}, max: {}".format(int(mean(nodes_expanded)), min(nodes_expanded), max(nodes_expanded)))
    print("Your agent took on average {} seconds to finish a game".format(round(mean(algo_exec_time), 2)))
    #print(f"Depths: {depths}")
    #print(f"Depths: {flatten(depths)}")
    #new_combined_depths = [x for l in depths for x in l] # flatten it for alpha beta
    new_combined_depths = flatten(depths) # flatten for MCTS
    print("Your agent reached the following depths: min: {}, mean: {}, median: {}, mode: {}, max: {}".format(min(new_combined_depths),
                                                                                                   round(statistics.mean(new_combined_depths), 2),
                                                                                                   statistics.median(new_combined_depths),
                                                                                                   statistics.mode(new_combined_depths),
                                                                                                    max(new_combined_depths)))
    #print(f"Exec. time: {end_time-start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Run matches to test the performance of your agent against sample opponents.",
        epilog=textwrap.dedent("""\
            Example Usage:
            --------------
            - Run 40 games (10 rounds = 20 games x2 for fair matches = 40 games) against
              the greedy agent with 4 parallel processes: 

                $python run_match.py -f -r 10 -o GREEDY -p 4

            - Run 100 rounds (100 rounds = 200 games) against the minimax agent with 1 process:

                $python run_match.py -r 100
        """)
    )
    parser.add_argument(
        '-d', '--debug', action="store_true",
        help="""\
            Run the matches in debug mode, which disables multiprocessing & multithreading
            support. This may be useful for inspecting memory contents during match execution,
            however, this also prevents the isolation library from detecting timeouts and
            terminating your code.
        """
    )
    parser.add_argument(
        '-f', '--fair_matches', action="store_true",
        help="""\
            Run 'fair' matches to mitigate differences caused by opening position 
            (useful for analyzing heuristic performance).  Setting this flag doubles 
            the number of rounds your agent will play.  (See README for details.)
        """
    )
    parser.add_argument(
        '-r', '--rounds', type=int, default=NUM_ROUNDS,
        help="""\
            Choose the number of rounds to play. Each round consists of two matches 
            so that each player has a turn as first player and one as second player.  
            This helps mitigate performance differences caused by advantages for either 
            player getting first initiative.  (Hint: this value is very low by default 
            for rapid iteration, but it should be increased significantly--to 50-100 
            or more--in order to increase the confidence in your results.
        """
    )
    parser.add_argument(
        '-o', '--opponent', type=str, default='MINIMAX', choices=list(TEST_AGENTS.keys()),
        help="""\
            Choose an agent for testing. The random and greedy agents may be useful 
            for initial testing because they run more quickly than the minimax agent.
        """
    )
    parser.add_argument(
        '-p', '--processes', type=int, default=NUM_PROCS,
        help="""\
            Set the number of parallel processes to use for running matches.  WARNING: 
            Windows users may see inconsistent performance using >1 thread.  Check the 
            log file for time out errors and increase the time limit (add 50-100ms) if 
            your agent performs poorly.
        """
    )
    parser.add_argument(
        '-t', '--time_limit', type=int, default=TIME_LIMIT,
        help="Set the maximum allowed time (in milliseconds) for each call to agent.get_action()."
    )
    args = parser.parse_args()

    logging.basicConfig(filename="matches.log", filemode="w", level=logging.DEBUG)
    logging.info(
        "Search Configuration:\n" +
        "Opponent: {}\n".format(args.opponent) +
        "Rounds: {}\n".format(args.rounds) +
        "Fair Matches: {}\n".format(args.fair_matches) +
        "Time Limit: {}\n".format(args.time_limit) +
        "Processes: {}\n".format(args.processes) +
        "Debug Mode: {}".format(args.debug)
    )

    main(args)
