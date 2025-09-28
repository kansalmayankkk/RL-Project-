"""Play Othello."""

"""
   Usefull Methods and attributes:

   env.unwrapped.player_turn    # -1 is BLACK 1 is WHITE
   env.unwrapped.possible_moves # [move, move, ...] list of moves
   env.unwrapped.n_players      # fixed value 2
   step()
   reset()
   render()

   obs returned by step and reset
   obs[0]                       # oobservation -> Board State
   obs[1]                       # actions      -> Possible actions

"""
import argparse
import simple_policies_ai
import gymnasium
import othello_ai

def create_policy(policy_type='rand', seed=0, search_depth=1,
                  init_p=None):
    if policy_type == 'rand':
        policy = simple_policies_ai.RandomPolicy(seed=seed)
    elif policy_type == 'greedy':
        policy = simple_policies_ai.GreedyPolicy(init_p)
    elif policy_type == 'maximin':
        policy = simple_policies_ai.MaxiMinPolicy(search_depth, init_p)
    else:
        policy = simple_policies_ai.HumanPolicy()
    return policy


def play(p_protagonist,
         protagonist_agent_type='greedy',
         opponent_agent_type='rand',
         p_board_size=8,
         num_rounds=100,
         protagonist_search_depth=1,
         opponent_search_depth=1,
         rand_seed=0,
         env_init_rand_steps=0,
         p_num_disk_as_reward=False,
         render=True,
         p_render_mode=None):
    print('protagonist: {}'.format(protagonist_agent_type))
    print('opponent: {}'.format(opponent_agent_type))

    if ((protagonist_agent_type == 'human') or \
       (opponent_agent_type == 'human')) and (p_render_mode is None) :
        print(f"Forzato render_mode a human.")
        p_render_mode = 'human'

    if opponent_agent_type == 'human':
        p_render_in_step = True
    else:
        p_render_in_step = False

    init_params = {
                     "protagonist":p_protagonist,
                     "board_size":p_board_size,
                     "num_disk_as_reward":p_num_disk_as_reward,
                     "render_in_step":(p_render_in_step and render),
                     "render_mode":p_render_mode,
                  }

    protagonist_policy = create_policy(
        policy_type=protagonist_agent_type,
        seed=rand_seed,
        search_depth=protagonist_search_depth,
        init_p=init_params)
    opponent_policy = create_policy(
        policy_type=opponent_agent_type,
        seed=rand_seed,
        search_depth=opponent_search_depth,
        init_p=init_params)

    env = gymnasium.make('othello_ai/Othello-v0', **init_params)

    win_cnts = draw_cnts = lose_cnts = 0
    for i in range(num_rounds):
        print('Episode {}'.format(i + 1))
        obs, info = env.reset()
        protagonist_policy.reset(env)
        opponent_policy.reset(env)
        if render:
            env.render()
        done = False
        while not done:
            if p_protagonist == env.unwrapped.player_turn:
                action = protagonist_policy.get_action(obs)
                obs, reward, done, truncated, info = env.step(action)
            else:
                action = opponent_policy.get_action(obs)
                obs, reward, done, truncated, info = env.step(action)
            if done:
                print('reward={}'.format(reward))
                if p_num_disk_as_reward:
                    total_disks = p_board_size ** 2
                    if p_protagonist == 1:
                        white_cnts = reward
                        black_cnts = total_disks - white_cnts
                    else:
                        black_cnts = reward
                        white_cnts = total_disks - black_cnts

                    if white_cnts > black_cnts:
                        win_cnts += 1
                    elif white_cnts == black_cnts:
                        draw_cnts += 1
                    else:
                        lose_cnts += 1
                else:
                    if reward == 1:
                        win_cnts += 1
                    elif reward == 0:
                        draw_cnts += 1
                    else:
                        lose_cnts += 1
                print('-' * 3)
            else:
                if render:
                    env.render()
    print('#Wins: {}, #Draws: {}, #Loses: {}'.format(
        win_cnts, draw_cnts, lose_cnts))
    env.close()


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--protagonist', default='rand',
                        choices=['rand', 'greedy', 'maximin', 'human'])
    parser.add_argument('--opponent', default='rand',
                        choices=['rand', 'greedy', 'maximin', 'human'])
    parser.add_argument('--protagonist-plays-white', default=False,
                        action='store_true')
    parser.add_argument('--num-disk-as-reward', default=False,
                        action='store_true')
    parser.add_argument('--board-size', default=8, type=int)
    parser.add_argument('--protagonist-search-depth', default=1, type=int)
    parser.add_argument('--opponent-search-depth', default=1, type=int)
    parser.add_argument('--rand-seed', default=0, type=int)
    parser.add_argument('--num-rounds', default=100, type=int)
    parser.add_argument('--init-rand-steps', default=10, type=int)
    parser.add_argument('--no-render', default=False, action='store_true')
    parser.add_argument('--render-mode', default=None, choices=['ansi', 'rgb_array', 'human'])
    args, _ = parser.parse_known_args()

    # Run test plays.
    protagonist = 1 if args.protagonist_plays_white else -1
    protagonist_agent_type = args.protagonist
    opponent_agent_type = args.opponent
    play(p_protagonist=protagonist,
         protagonist_agent_type=protagonist_agent_type,
         opponent_agent_type=opponent_agent_type,
         p_board_size=args.board_size,
         num_rounds=args.num_rounds,
         protagonist_search_depth=args.protagonist_search_depth,
         opponent_search_depth=args.opponent_search_depth,
         rand_seed=args.rand_seed,
         env_init_rand_steps=args.init_rand_steps,
         p_num_disk_as_reward=args.num_disk_as_reward,
         render=not args.no_render,
         p_render_mode=args.render_mode)
