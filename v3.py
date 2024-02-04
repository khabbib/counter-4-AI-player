import gym
import random
import requests
import numpy as np
import argparse
import sys
import copy
from gym_connect_four import ConnectFourEnv

env: ConnectFourEnv = gym.make("ConnectFour-v0")
SERVER_P = -1
STUDENT_P = 1
ROWS = 6
COLS = 7

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

def is_valid_location(board, col):
    """
    Check if a given column is a valid location for a move.
    :param board: The current state of the game board.
    :param col: The column to check.
    :return: True if the column is a valid location for a move, False otherwise.
    """
    return board[ROWS - 1][col] == 0

def winning_move(board, piece):
    """
    Check if the specified player has achieved a winning move.
    :param board: The current state of the game board.
    :param piece: The player's piece (1 for player, -1 for opponent).
    :return: True if the player has achieved a winning move, False otherwise.
    """
    # Check horizontal locations for win
    for c in range(COLS - 3):
        for r in range(ROWS):
            if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][c + 3] == piece:
                return True

    # Check vertical locations for win
    for c in range(COLS):
        for r in range(ROWS - 3):
            if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][c] == piece:
                return True

    # Check positively sloped diagonals
    for c in range(COLS - 3):
        for r in range(ROWS - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and board[r + 3][
                c + 3] == piece:
                return True

    # Check negatively sloped diagonals
    for c in range(COLS - 3):
        for r in range(3, ROWS):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and board[r - 3][
                c + 3] == piece:
                return True

    return False

def is_terminal_node(board):
    """
    Check if the game is in a terminal state.
    :param board: The current state of the game board.
    :return: True if the game is in a terminal state, False otherwise.
    """

    return winning_move(board, STUDENT_P) or winning_move(board, SERVER_P) or len(get_valid_moves(board)) == 0

def get_valid_moves(board):
    valid_moves = []
    for col in range(COLS):
        if is_valid_location(board, col):
            valid_moves.append(col)
    return valid_moves

def evaluate_board(board):
    player_score = 0
    opponent_score = 0
    for row in range(6):
        for col in range(7):
            if board[row][col] == STUDENT_P:
                player_score += evaluate_position(board, row, col)
            elif board[row][col] == SERVER_P:
                opponent_score += evaluate_position(board, row, col)
    return player_score - opponent_score

def evaluate_position(board, row, col):
    # Evaluate the positional advantage of a piece
    score = 0
    # Add score for horizontal connections
    score += evaluate_direction(board, row, col, 0, 1)  # Right
    score += evaluate_direction(board, row, col, 0, -1)  # Left
    # Add score for vertical connections
    score += evaluate_direction(board, row, col, 1, 0)  # Down
    # Add score for diagonal connections
    score += evaluate_direction(board, row, col, 1, 1)  # Down-right
    score += evaluate_direction(board, row, col, 1, -1)  # Down-left
    return score

def evaluate_direction(board, row, col, delta_row, delta_col):
    score = 0
    for i in range(1, 4):
        r = row + i * delta_row
        c = col + i * delta_col
        if 0 <= r < 6 and 0 <= c < 7:
            if board[r][c] == STUDENT_P:
                score += 1
            elif board[r][c] == SERVER_P:
                score -= 1
            else:
                break
    return score

def get_next_open_row(board, col): # this func is working correctly
    """
    Get the next open row in the specified column.
    :param board: The current state of the game board.
    :param col: The column to check for the next open row.
    :return: The index of the next open row in the specified column.
    """
    for r in range(ROWS):
        if board[r][col] == 0:
            return r

def drop_piece(board, col, piece):
    """
    Function to safely drop a piece into the board at the specified column.
    
    :param board: The game board, a 2D list.
    :param col: The column where the piece is to be dropped.
    :param piece: The piece to drop into the board.
    :return: The updated board after the piece is dropped, or None if the move is invalid.
    """
    ROWS = len(board)
    if ROWS == 0 or col < 0 or col >= len(board[0]):
        # If the board is empty, the column is out of bounds, return None to indicate an invalid move
        print("len = 0")
        return None
    
    # Find the lowest empty row in the specified column
    for row in range(ROWS-1, -1, -1):
        if board[row][col] == 0:  # Assuming 0 represents an empty cell
            board[row][col] = piece
            return board
    
    print(board)
    # If the column is full, return None to indicate an invalid move
    return None

def get_valid_moves(board):
    """
    Get a list of valid moves (columns) that can be made on the current board.
    :param board: The current state of the game board.
    :return: A list of valid moves (column indices).
    """
    valid_moves = []
    for col in range(COLS):
        if is_valid_location(board, col):
            valid_moves.append(col)
    return valid_moves

# Student_move -> my_ai_move
def my_ai_move(board):
    """
    den ska spela många at least over 20 gånger och jag måste vinna!
    Andra kraven: min ai ska göra en move under 5 s.

    This function uses the minimax algorithm with alpha-beta pruning to select the best move.
    :param board: The current state of the game board.
    :return: The best move from 0 to 6.
    """
    # new_board_for_minp = copy.deepcopy(board)
    # def max_value(new_board_for_minp, alpha, beta, depth):
    #     if depth == 0 or is_terminal_node(new_board_for_minp):
    #         return None, evaluate_board(new_board_for_minp)

    #     max_move = None
    #     max_score = -float('inf')
    #     for move in get_valid_moves(new_board_for_minp):
    #         new_board = drop_piece(new_board_for_minp, move, STUDENT_P)
    #         _, score = min_value(new_board, alpha, beta, depth - 1)
    #         if score > max_score:
    #             max_score = score
    #             max_move = move
    #         alpha = max(alpha, max_score)
    #         if alpha >= beta:
    #             break
    #     return max_move, max_score

    # def min_value(new_board_for_minp, alpha, beta, depth):
    #     if depth == 0 or is_terminal_node(new_board_for_minp):
    #         return None, evaluate_board(new_board_for_minp)

    #     min_move = None
    #     min_score = float('inf')
    #     for move in get_valid_moves(new_board_for_minp):
    #         new_board = drop_piece(new_board_for_minp, move, SERVER_P)
    #         _, score = max_value(new_board, alpha, beta, depth - 1)
    #         if score < min_score:
    #             min_score = score
    #             min_move = move
    #         beta = min(beta, min_score)
    #         if beta <= alpha:
    #             break
    #     return min_move, min_score

    def max_value(board, alpha, beta, depth):
        if depth == 0 or is_terminal_node(board):
            return None, evaluate_board(board)

        max_move = None
        max_score = -float('inf')
        valid_moves = get_valid_moves(board) # Value is correct!
        if not valid_moves:  # No valid moves available
            return None, evaluate_board(board)

        for move in valid_moves:
            temp_board = copy.deepcopy(board)
            temp_board = drop_piece(temp_board, move, STUDENT_P)
            
            _, score = min_value(temp_board, alpha, beta, depth - 1)
            
            print("score", score)
            
            if score > max_score:
                max_score = score
                max_move = move
            
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        
        return max_move, max_score

    def min_value(board, alpha, beta, depth):
        if depth == 0 or is_terminal_node(board):
            return None, evaluate_board(board)

        min_move = None
        min_score = float('inf')
        
        valid_moves = get_valid_moves(board)
        if not valid_moves:  # No valid moves available
            return None, evaluate_board(board)

        for move in valid_moves:
            temp_board = copy.deepcopy(board)
            drop_piece(temp_board, move, SERVER_P)
            
            _, score = max_value(temp_board, alpha, beta, depth - 1)
            if score < min_score:
                min_score = score
                min_move = move
            
            beta = min(beta, score)
            if alpha >= beta:
                break
        return min_move, min_score
    best_move, _ = max_value(board, -float('inf'), float('inf'), depth=3)
    print("board after minipulation: ", board)
    print("Best move MinMaxAlgo: ", best_move)
    return best_move


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
#    print("Current state (1 are student discs, -1 are servers, 0 is empty): ")
#    print(state)
   print()

   done = False
   while not done:
      # Select your move
      stmove = my_ai_move(state) # TOFIX: return none sometimes.

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
               print("You tied to make an illegal move! You have lost the game.", stmove, avmoves)
               break
            state, result, done, _ = env.step(stmove)
         student_gets_move = True # student only skips move first turn if bot starts

         # print or render state here if you like

         # select and make a move for the opponent, returned reward from students view
         if not done:
             _, reward, done = opponents_move(env)
            #  result = reward
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
   parser.add_argument("-s", "--stats", help = "Show your current online stats", action="store_true")
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
