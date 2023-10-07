from __future__ import division
from IPython.display import clear_output
import numpy as np
import gym
import matplotlib.pyplot as plt
from IPython import display
import random
import pandas as pd

# how to render in Jupyter:
# https://stackoverflow.com/questions/40195740/how-to-run-openai-gym-render-over-a-server
# https://www.youtube.com/watch?v=O84KgRt6AJI
def show_state(image, step=0, name='', info=''):
    plt.figure(3)
    plt.clf()
    plt.imshow(image)
    plt.title("%s | Step: %d %s" % (name, step, info))
    plt.axis('off')

    display.clear_output(wait=True)
    display.display(plt.gcf())


def show_state_target_index(image, target_index, step=0, name='', info=''):
    plt.figure(3)
    plt.clf()
    plt.imshow(image)
    plt.scatter(tuples_array_df.iloc[:, 0], tuples_array_df.iloc[:, 1], c='r', s=5)
    plt.title("%s | Step: %d %s" % (name, step, info))
    plt.axis('off')

    display.clear_output(wait=True)
    display.display(plt.gcf())

# To transform pixel matrix to a single vector.
def getState(inState):
    # each row is all 1 color
    rgbRows = np.reshape(inState,(len(inState[0])*len(inState), 3)).T

    # add each with appropriate shifting
    # get RRRRRRRR GGGGGGGG BBBBBBBB
    return np.add(np.left_shift(rgbRows[0], 16),
        np.add(np.left_shift(rgbRows[1], 8), rgbRows[2]))



# Define the formula to transform the whole Cartesian coordinates to ploar coordinates

def index_polar_transformation(A_i, L = 1024, b = 128, R1 = 95, print_info = False):

  '''
  A_i:    the points x-axis index
  L:      the screen width
  b:      the length of one block
  R1:     the radius of the frist circle

  return:
  i:      the index of which block it blongs to
  T_x_i:  the transformed x coordinates
  T_y_i:  the transformed y coordiantes


  We define the whole x-axis as 4 blocks in each direction.
  As [d2, c2, b2, a2, a1, b1, c1, d1]
  And we want to transfer all the points into a polar coordinates with 4
  different circles, with R1, R2, R3, R4, R as the Radius.
  And for the right hand side of the half circle from [0, Pi]. We use the
  right direction as [a1, b1, c1, d1].
  And for the left hand side of the half circle from [Pi, 2Pi]. We use the
  left direction as [a2, b2, c2, d2].

  Let i be the ith block in our settings
  Let A_i be the data point position on [0, L]
  L as the width as the input screen
  i.e. L/2 is the middle point of the screen

  Let P_i = A_i - L/2 , if x_i > L/2
      P_i = abs(A_i - L/2), if x_i <= L/2

  Let Ri be the ith block's corresponding radius in the polar system.
  And Ri = i * R1

  Let b be the range of each block. b = 128 in our case.

  Now let's define i.
  i = np.floor(P_i / b) + 1, if P_i % b != 0
  i = np.floor((P_i + 1) / b) + 1, if Pi % b == 0

  Now we have the basic settins for the system. We need to define the transformation.

  Let k be the transformation variable.
  Let k * b = 2 * R_i.
  That is we transform our block range to the diameter in the circle.
  In order to get the theta as the angle in the circle.

  That is, k = (2 * R_i) / b.

  Now we spilt the points into two directions.
  If A_i > L/2: as in the right direction.
  Let the transformed position M_i be the new position in the range of [0 , 2 * Ri].
  then M_i = k * P_i.

  And we define the theta_i = M_i / (2 * R_i) * 180
  180 is the half circle degree range.

  Then the newly transformed coordinates in polar system is the following:
  Ti = (x_i, y_i)
  if A_i > L / 2 as the right half circle
    x_i = R_i * sin(theta_i)
    y_i = R_i * cos(theta_i)
  if A_i <= L / 2 as the left hand side circle.
    x_i = (-1) * R_i * sin(theta_i)
    y_i = R_i * sin(theta_i)
  '''

  # We get P_i first
  if A_i > (L/2):
    P_i = (A_i - L/2)
  else:
    P_i = np.abs(A_i - L/2)

  # Get i
  if P_i % b == 0 or P_i % b == (b - 1):
    i = np.floor((P_i + 1) / b)
  else:
    i = np.floor((P_i + 1) / b) + 1



  if P_i == i * b:
    P_i = b
  else:
    P_i = P_i % b


  # Delete Exception of when i = 0
  if i == 0:
    i = 1



  # Get R_i
  R_i = i * R1

  # Transformation
  # Get k
  k = 2 * R_i / b

  # Get M_i
  M_i = k * P_i

  # Get theta_i
  # use np.deg2rad to transform degree to rad
  theta_i = M_i / (2 * R_i) * np.deg2rad(180)

  # Then we have our new coordinates

  # sin_theta = 0
  # cos_theta = 0

  # if theta_i != 0:
  sin_theta = np.sin(theta_i)
  cos_theta = np.cos(theta_i)
  #
  if A_i > (L/2):
    # print('larger')
    T_x_i = R_i * sin_theta
    T_y_i = R_i * cos_theta
  else:
    # print('smaller')
    T_x_i = (-1) * R_i * sin_theta
    T_y_i = R_i * cos_theta

  if print_info == True:
    print('x', A_i)
    print('T_x: ', T_x_i )
    print('T_y: ', T_y_i)
    print('Ai', A_i)
    print('Pi', P_i)
    print('theta', theta_i)
    print('R_i', R_i)
    print('mi', M_i)
    print('k', k)
    print('i', i)
  return T_x_i, T_y_i


def get_transformed_matrix( width = 1024, height = 768, Radius = 92):
  # Initialize the matrix with tuples representing coordinates (x, y)
  matrix = [[(x, y) for x in range(width)] for y in range(height)]
  Transformed_matrix = [[(x, y) for x in range(width)] for y in range(height)]

  for i in tqdm(range(len(matrix))):
    for j in range(len(matrix[i])):
      if i == 0:
        # print(f'JJJJJJJJJJJJ {j}')
        x , y = np.round(index_polar_transformation(matrix[i][j][0], L = width, b = (width / 8), R1 = Radius, print_info = False), 0)
        Transformed_matrix[i][j] = (int(x + width / 2), int(y + height / 2))
        # pdb.set_trace()
      else:
        Transformed_matrix[i][j] = Transformed_matrix[0][j]

  return Transformed_matrix

def get_tuples_array(width = 640, height = 400, Radius = 49):
  Transformed_matrix = pd.DataFrame(get_transformed_matrix(width = width, height = height, Radius = Radius))
  return np.array(Transformed_matrix.iloc[1,].values.tolist())


def get_circle(screen_buf, tuples_array):
  # Create a new array with the desired shape and fill it with the corresponding values from the 3D matrix
  new_array = np.zeros((1, tuples_array.shape[0], 3))

  for i, t in enumerate(tuples_array):
      x, y = t
      if x > 640 or y > 400:
          x = 640
          y = 400
      new_array[0, i] = screen_buf[y, x]
  return new_array



# import tensorflow as tf      # Deep Learning library
import numpy as np
import pandas as pdd        # Handle matrices
from vizdoom import *        # Doom Environment
import random                # Handling random number generation
import time                  # Handling time calculation
from skimage import transform# Help us to preprocess the frames

from collections import deque# Ordered collection with ends
import matplotlib.pyplot as plt # Display graphs

from tqdm.auto import tqdm

import sys
sys.path.append('./PyTPG/')
# import to do training
from tpg.trainer import Trainer
# import to run an agent (always needed)
from tpg.agent import Agent
import tensorflow as tf  


from vizdoom import *
import random


print("\n\nBASIC EXAMPLE\n")

# Create DoomGame instance. It will run the game and communicate with you.
game = DoomGame()

# Sets path to vizdoom engine executable which will be spawned as a separate process. Default is "./vizdoom".
# game.set_vizdoom_path("/content/ViZDoom")
game.load_config("./ViZDoom/scenarios/basic.cfg")


# Sets path to doom2 iwad resource file which contains the actual doom game. Default is "./doom2.wad".
# game.set_doom_game_path("../../bin/freedoom2.wad")
# game.set_doom_game_path("/content/ViZDoom/scenarios/basic.wad")   # Not provided with environment due to licenses.

# Sets path to additional resources iwad file which is basically your scenario iwad.
# If not specified default doom2 maps will be used and it's pretty much useless... unless you want to play doom.
# game.set_doom_scenario_path("../../scenarios/basic.wad")
game.set_doom_scenario_path("./ViZDoom/scenarios/basic.wad")

# Set map to start (scenario .wad files can contain many maps).
game.set_doom_map("map01")

# Sets resolution. Default is 320X240
game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X400)

# Sets the screen buffer format. Not used here but now you can change it. Default is CRCGCB.
game.set_screen_format(vizdoom.ScreenFormat.RGB24)
# game.set_screen_format(vizdoom.ScreenFormat.GRAY8)

# Sets other rendering options
game.set_render_hud(False)
game.set_render_crosshair(False)
game.set_render_weapon(True)
game.set_render_decals(False)
game.set_render_particles(False)
game.set_render_effects_sprites(False)
game.set_render_messages(False)
game.set_render_corpses(False)

# Adds buttons that will be allowed.
available_buttons = [vizdoom.Button.MOVE_LEFT, vizdoom.Button.MOVE_RIGHT, vizdoom.Button.ATTACK]
game.set_available_buttons(available_buttons)
# game.add_available_button(Button.MOVE_LEFT) # Appends to available buttons.
# game.add_available_button(Button.MOVE_RIGHT)
# game.add_available_button(Button.ATTACK)

# Adds game variables that will be included in state.
game.add_available_game_variable(vizdoom.GameVariable.AMMO2)
# game.set_available_game_variables is also available.

# Causes episodes to finish after 200 tics (actions)
game.set_episode_timeout(200)

# Makes episodes start after 10 tics (~after raising the weapon)
game.set_episode_start_time(10)

# Makes the window appear (turned on by default)
game.set_window_visible(True)

# Turns on the sound. (turned off by default)
game.set_sound_enabled(True)

# Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
game.set_mode(vizdoom.Mode.PLAYER)

# Enables engine output to console.
# game.set_console_enabled(True)

game.set_window_visible(False)  # Hide the game window

# Define some actions. Each list entry corresponds to declared buttons:
# MOVE_LEFT, MOVE_RIGHT, ATTACK
# more combinations are naturally possible but only 3 are included for transparency when watching.
actions = [[1, 0, 1], [0, 1, 1], [0, 0, 1]]

from tqdm.auto import tqdm
import joblib

# first create an instance of the TpgTrainer
# this creates the whole population and everything
# teamPopSize should realistically be at-least 100
trainer = Trainer(actions=range(24), teamPopSize=80)
tensor_board_index_number = 0

save_file_path = './action_penalty_runs/'

import time # for tracking time

tStart = time.time()

from datetime import datetime
current_time = datetime.now()
formatted_time = current_time.strftime("%H_%M_%S")

curScores = [] # hold scores in a generation
summaryScores = [] # record score summaries for each gen (min, max, avg)


# Initialize the game. Further configuration won't take any effect from now on.
game.init()

tuples_array = get_tuples_array(width = 640, height = 400, Radius = 49)

# 5 generations isn't much (not even close), but some improvements
# should be seen.
for gen in tqdm(range(5000)): # generation loop
    curScores = [] # new list per gen

    # index_mapping_type
    index_mapping_type = 1


    # Save trainer
    if gen % 20 == 0:
      current_time_population = datetime.now()
      formatted_time_population = current_time_population.strftime("%m_%d_%H_%_M_%S")
      joblib.dump(trainer, f'{save_file_path}trainer_{formatted_time_population}.pkl')

    agents = trainer.getAgents()


    previous_action = []

    wining_runs = 0
    number_of_games = 0

    while True: # loop to go through agents
        teamNum = len(agents)
        # print(teamNum)
        agent = agents.pop()
        if agent is None:
            break # no more agents, so proceed to next gen

        # Starts a new episode. It is not needed right after init() but it doesn't cost much and the loop is nicer.
        game.new_episode()

        score = 0

        number_of_games += 1
        
        
        while not game.is_episode_finished():
            # Get the state
            state = game.get_state()
            n = state.number
            vars = state.game_variables
            screen_buf = state.screen_buffer
            depth_buf = state.depth_buffer
            labels_buf = state.labels_buffer
            automap_buf = state.automap_buffer
            labels = state.labels

            # if index_mapping_type == 0:
            #   new_image_input = get_circle(screen_buf)
            #   if len(new_image_input) < 640:
            #     new_image_input = screen_buf[200]
            #   new_image_input = new_image_input.reshape(1, len(new_image_input), 3)
            #   current_state = getState(np.array(new_image_input, dtype=np.int32))
            # else:
            #   new_image_index = random.randint(150, 250)
            #   new_image_input = screen_buf[new_image_index]
            #   if len(new_image_input) < 640:
            #     new_image_input = screen_buf[200]
            #   new_image_input = new_image_input.reshape(1, len(new_image_input), 3)

            #   current_state = getState(np.array(new_image_input, dtype=np.int32))

            new_image_index = random.randint(180, 220)
            new_image_input = screen_buf[new_image_index]
            if len(new_image_input) < 640:
              new_image_input = screen_buf[200]
            new_image_input = new_image_input.reshape(1, len(new_image_input), 3)
            current_state_horizotal = getState(np.array(new_image_input, dtype=np.int32))

            new_image_input = get_circle(screen_buf, tuples_array)
            if len(new_image_input) < 640:
              new_image_input = screen_buf[200]
            new_image_input = new_image_input.reshape(1, len(new_image_input), 3)
            current_state_polar = getState(np.array(new_image_input, dtype=np.int32))

            act = agent.act(np.concatenate((current_state_horizotal, current_state_polar)))
            # act = agent.act(current_state_horizotal)

            # set R[1] as the mapping mutation type
            # if act[1] is not None:
            #   index_mapping_type = int(act[1][0]) % 2

            # Make random action and get reward
            r = game.make_action(actions[act[0] % 3])

            if r == 100:
              print("############################ GOT HIM ############################")
              # print(f" Previous Score: {score}")
              # print(f" Previous Action: {previous_action}")
              # print(f" Current Action: {act[0]}")
              # print("############################ GOT HIM ############################")
              r = 1000000000
              wining_runs += 1

            elif previous_action != act[0]:
              previous_action = act[0]
              r = 1000
            else:
              r = -100
              # r *= 1000

            # show_state(screen_buf, gen, 'Assault', 'Gen #' + str(n) +
            #            ', Team #' + str( agent.team.id) +
            #            ', Score: ' + str(r) +
            #            ', Total Score: ' + str(score) +
            #            ', R[0]:' + str(act[0]) +
            #            ', Action: ' + str(actions[act[0] % 3])) # render env

            # print('Assault', 'Gen #' + str(n) +
            #            ', Team #' + str( agent.team.id) +
            #            ', Score: ' + str(r) +
            #            ', Total Score: ' + str(score) +
            #            ', R[0]:' + str(act[0]) +
            #            ', Action: ' + str(actions[act[0] % 3]))
            # show_state(new_image_input, gen, 'Assault', 'Gen #' + str(n) +
            #   ', Team #' + str( agent.team.id) +
            #   ', Score: ' + str(r) +
            #   ', Total Score: ' + str(score) +
            #   ', Action: ' + str(actions[act[0] % 3])) # render env

            score += r
            tensor_board_index_number += 1




            # with tf.summary.create_file_writer(log_index_maping_type).as_default():
            #   tf.summary.scalar('Mapping_type', agent.team.id % 2, step=number)

        agent.reward(score) # must reward agent (if didn't already score)

        # with tf.summary.create_file_writer(log_current_generation).as_default():
        #       tf.summary.scalar('Generations', gen, step=tensor_board_index_number)

        # if number_of_moves > 0 :
        #   with tf.summary.create_file_writer(log_total_reward).as_default():
        #         tf.summary.scalar('total reward', score, step=tensor_board_index_number)

        print('Assault', 'Gen #' + str(gen) +
                  ', Team #' + str( agent.team.id) +
                  ', Total Score: ' + str(score))

        curScores.append(score) # store score

        if len(agents) == 0:
            break


    # at end of generation, make summary of scores
    summaryScores.append((min(curScores), max(curScores),
                    sum(curScores)/len(curScores))) # min, max, avg
    trainer.evolve()

    print(f"###################### Finsh one team with Wining Rate { wining_runs / number_of_games} ######################")
    # with tf.summary.create_file_writer(log_current_reward).as_default():
    #   tf.summary.scalar('Wining Rate', wining_runs / number_of_games, step=tensor_board_index_number)


game.close()

#clear_output(wait=True)
print('Time Taken (Hours): ' + str((time.time() - tStart)/3600))
print('Results:\nMin, Max, Avg')
for result in summaryScores:
    print(result[0],result[1],result[2])
