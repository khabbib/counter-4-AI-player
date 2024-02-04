import gym
import random
import requests
import numpy as np
import argparse
import sys
from gym_connect_four import ConnectFourEnv

env: ConnectFourEnv = gym.make("ConnectFour-v0")
AI_PIECE = -1
PLAYER_PIECE = 1
COLUMN_COUNT = 7
ROW_COUNT = 6
SERVER_ADDRESS = "https://vilde.cs.lth.se/edap01-4inarow/"
API_KEY = 'nyckel'
STIL_ID = ["ha8804mo-s"] # TODO: fill this list with your stil-id's

def call_server(move):
   res = requests.post(SERVER_ADDRESS + "move",
                       data={
                           "stil_id": STIL_ID,
                           "move": move, # -1 signals the system to start a new game. any running game is counted as a loss
                           "api_key": API_KEY,
                       })
   # For safety some respose checking is done here
   if res.status_code != 200:
      print("Server gave a bad response, error code={}".format(res.status_code))
      exit()
   if not res.json()['status']:
      print("Server returned a bad status. Return message: ")
      print(res.json()['msg'])
      exit()
   return res

def check_stats():
   res = requests.post(SERVER_ADDRESS + "stats",
                       data={
                           "stil_id": STIL_ID,
                           "api_key": API_KEY,
                       })

   stats = res.json()
   return stats

# def student_move():
#    """
#    TODO: Implement your min-max alpha-beta pruning algorithm here.
#    Give it whatever input arguments you think are necessary
#    (and change where it is called).
#    The function should return a move from 0-6
#    """
#    return random.choice([0, 1, 2, 3, 4, 5, 6])


def is_terminal(node):
    # Check if the board is full
    if not 0 in node:
        return True
    # Check for a winning line
    for player in [-1, 1]:
        # Horizontal
        for row in node:
            for i in range(len(row) - 3):
                if row[i] == player and row[i+1] == player and row[i+2] == player and row[i+3] == player:
                    return True
        # Vertical
        for j in range(len(node[0])):
            for i in range(len(node) - 3):
                if node[i][j] == player and node[i+1][j] == player and node[i+2][j] == player and node[i+3][j] == player:
                    return True
        # Diagonal
        for i in range(len(node) - 3):
            for j in range(len(node[0]) - 3):
                if node[i][j] == player and node[i+1][j+1] == player and node[i+2][j+2] == player and node[i+3][j+3] == player:
                    return True
                if node[i][j+3] == player and node[i+1][j+2] == player and node[i+2][j+1] == player and node[i+3][j] == player:
                    return True

    # If no terminal state is found, return False
    return False

def get_valid_moves(node):
    # This function should return a list of valid moves for the current game state.
    # In Connect 4, a move is valid if the chosen column is not already full.
    return env.available_moves()

def make_move(node, move, is_maximizing_player):
    # This function should return a new game state after making the given move.
    # You'll need to implement this based on your game state representation.
    env.change_player()
    new_state, reward, done, _ = env.step(move)
    env.change_player()
    return new_state

def evaluate(node):
    # This function should return a score representing the value of the game state for the maximizing player.
    # A common approach is to count the number of '4 in a row' possibilities for the maximizing player minus
    # the number of '4 in a row' possibilities for the minimizing player.
    # For simplicity, we can use the reward as the evaluation function.
   random.randint(-100, 100)

def student_move(state):
  

  
    # recursive minmax
    # TIPS: Implementera without alpha beta purring i första steg
    # 
    _, best_move = minimax(state, 4, float('-inf'), float('inf'), True)
    return best_move

def minimax(node, depth, alpha, beta, maximizing_player):
    if depth == 0 or is_terminal(node):
        return evaluate(node), None

    if maximizing_player:
        max_eval = float('-inf')
        best_move = None
        for move in get_valid_moves(node):
            eval, _ = minimax(make_move(node, move, True), depth - 1, alpha, beta, False)
            if eval is not None and eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        best_move = None
        for move in get_valid_moves(node):
            eval, _ = minimax(make_move(node, move, False), depth - 1, alpha, beta, True)
            if eval is not None and eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move


def play_game(vs_server):
   """
   The reward for a game is as follows. You get a
   botaction = random.choice(list(avmoves)) reward from the
   server after each move, but it is 0 while the game is running
   loss = -1
   win = +1
   draw = +0.5
   error = -10 (you get this if you try to play in a full column)
   Currently the player always makes the first move
   """

   # default state
   state = np.zeros((6, 7), dtype=int)

   # setup new game
   if vs_server:
      # Start a new game
      res = call_server(-1) # -1 signals the system to start a new game. any running game is counted as a loss

      # This should tell you if you or the bot starts
      print(res.json()['msg'])
      botmove = res.json()['botmove']
      state = np.array(res.json()['state'])
      # reset env to state from the server (if you want to use it to keep track)
      env.reset(board=state)
   else:
      # reset game to starting state
      env.reset(board=None)
      # determine first player
      student_gets_move = random.choice([True, False])
      if student_gets_move:
         print('You start!')
         print()
      else:
         print('Bot starts!')
         print()

   # Print current gamestate
   print("Current state (1 are student discs, -1 are servers, 0 is empty): ")
   print(state)
   print()

   done = False
   while not done:
      # Select your move
      stmove = student_move(state) # TODO: change input here
      print("Studen move: ", stmove)
      # make both student and bot/server moves
      if vs_server:
         # Send your move to server and get response
         res = call_server(stmove)
         print(res.json()['msg'])

         # Extract response values
         result = res.json()['result']
         botmove = res.json()['botmove']
         state = np.array(res.json()['state'])
         # reset env to state from the server (if you want to use it to keep track)
         env.reset(board=state)
      else:
         if student_gets_move:
            # Execute your move
            avmoves = env.available_moves()
            if stmove not in avmoves:
               print("You tied to make an illegal move! You have lost the game.")
               break
            state, result, done, _ = env.step(stmove)

         student_gets_move = True # student only skips move first turn if bot starts

         # print or render state here if you like

         # select and make a move for the opponent, returned reward from students view
         if not done:
             state, reward, done = opponents_move(env)
             state, result, done, _ = env.step(stmove)

      # Check if the game is over
      if result != 0:
         done = True
         if not vs_server:
            print("Game over. ", end="")
         if result == 1:
            print("You won!")
         elif result == 0.5:
            print("It's a draw!")
         elif result == -1:
            print("You lost!")
         elif result == -10:
            print("You made an illegal move and have lost!")
         else:
            print("Unexpected result result={}".format(result))
         if not vs_server:
            print("Final state (1 are student discs, -1 are servers, 0 is empty): ")
      else:
         print("Current state (1 are student discs, -1 are servers, 0 is empty): ")

      # Print current gamestate
      print(state)
      print()

def opponents_move(env):
   env.change_player() # change to oppoent
   avmoves = env.available_moves()
   if not avmoves:
      env.change_player() # change back to student before returning
      return -1

   # TODO: Optional? change this to select actions with your policy too
   # that way you get way more interesting games, and you can see if starting
   # is enough to guarrantee a win
   action = random.choice(list(avmoves))

   state, reward, done, _ = env.step(action)
   if done:
      if reward == 1: # reward is always in current players view
         reward = -1
   env.change_player() # change back to student before returning
   return state, reward, done






###################################################
      # starter
###################################################
def main():
   # Parse command line arguments
   parser = argparse.ArgumentParser()
   group = parser.add_mutually_exclusive_group()
   group.add_argument("-l", "--local", help = "Play locally", action="store_true")
   # group.add_argument("-o", "--online", help = "Play online vs server", action="store_true")
   # parser.add_argument("-s", "--stats", help = "Show your current online stats", action="store_true")
   args = parser.parse_args()
   # Print usage info if no arguments are given
   if len(sys.argv)==1:
      parser.print_help(sys.stderr)
      sys.exit(1)

   if args.local:
      play_game(vs_server = False)
   elif args.online:
      play_game(vs_server = True)

   if args.stats:
      stats = check_stats()
      print(stats)

   # TODO: Run program with "--online" when you are ready to play against the server
   # the results of your games there will be logged
   # you can check your stats bu running the program with "--stats"

if __name__ == "__main__":
    main()
