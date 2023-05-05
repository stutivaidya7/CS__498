import json
class QLearning:

    def __init__(self, train):
        self.train = train
        self.discount_factor = 0.95
        self.alpha = 0.7
        self.reward = {0: 0, 1: -1000}
        self.alpha_decay = 0.00003
        self.episode = 0
        self.previous_action = 0
        self.previous_state = "0_0_0_0"
        self.moves = []
        self.scores = []
        self.max_score = 0
        self.q_values = {}
        self.load_qvalues()
        self.load_training_states()

    def load_qvalues(self):
        print("Loading Q-table states ")
        try:
            with open("data/q_values_resume.json", "r") as f:
                self.q_values = json.load(f)
        except IOError:
            self.init_qvalues(self.previous_state)

    def init_qvalues(self, state):
        if self.q_values.get(state) is None:
            self.q_values[state] = [0, 0, 0]  # [Q of no action, Q of flap action, Times experienced this state]

    def load_training_states(self):
        if self.train:
            print("Loading training states ")
            try:
                with open("data/training_values_resume.json", "r") as f:
                    training_state = json.load(f)
                    self.episode = training_state['episodes'][-1]
                    self.scores = training_state['scores']
                    self.alpha = max(self.alpha - self.alpha_decay * self.episode, 0.1)
                    # self.epsilon = max(self.epsilon - self.epsilon_decay * self.episode, 0)
                    self.max_score = max(self.scores)
            except IOError:
                pass

    def act(self, x, y, vel, pipe):
        state = self.get_state(x, y, vel, pipe)
        if self.train:
            self.moves.append((self.previous_state, self.previous_action, state))
            self.reduce_moves()
            self.previous_state = state
        self.previous_action = 0 if self.q_values[state][0] >= self.q_values[state][1] else 1

        return self.previous_action

    def update_qvalues(self, score):
        self.episode += 1
        self.scores.append(score)
        self.max_score = max(score, self.max_score)

        if self.train:
            history = list(reversed(self.moves))
            high_death_flag = True if int(history[0][2].split("_")[1]) > 120 else False
            t, last_flap = 0, True
            for move in history:
                t += 1
                state, action, new_state = move
                self.q_values[state][2] += 1
                curr_reward = self.reward[0]
                if t <= 2:
                    curr_reward = self.reward[1]
                    if action:
                        last_flap = False
                elif (last_flap or high_death_flag) and action:
                    curr_reward = self.reward[1]
                    last_flap = False
                    high_death_flag = False

                self.q_values[state][action] = (1 - self.alpha) * (self.q_values[state][action]) + \
                                               self.alpha * (curr_reward + self.discount_factor *
                                                             max(self.q_values[new_state][0:2]))
            if self.alpha > 0.1:
                self.alpha = max(self.alpha_decay - self.alpha_decay, 0.1)
            self.moves = []

    def get_state(self, x, y, vel, pipe):
        pipe0, pipe1 = pipe[0], pipe[1]
        if x - pipe[0]["x"] >= 50:
            pipe0 = pipe[1]
            if len(pipe) > 2:
                pipe1 = pipe[2]

        x0 = pipe0["x"] - x
        y0 = pipe0["y"] - y
        if -50 < x0 <= 0:
            y1 = pipe1["y"] - y
        else:
            y1 = 0
        if x0 < -40:
            x0 = int(x0)
        elif x0 < 140:
            x0 = int(x0) - (int(x0) % 10)
        else:
            x0 = int(x0) - (int(x0) % 70)

        if -180 < y0 < 180:
            y0 = int(y0) - (int(y0) % 10)
        else:
            y0 = int(y0) - (int(y0) % 60)

        if -180 < y1 < 180:
            y1 = int(y1) - (int(y1) % 10)
        else:
            y1 = int(y1) - (int(y1) % 60)

        state = str(int(x0)) + "_" + str(int(y0)) + "_" + str(int(vel)) + "_" + str(int(y1))
        self.init_qvalues(state)
        return state

    def reduce_moves(self, reduce_len=1000000):
        if len(self.moves) > reduce_len:
            history = list(reversed(self.moves[:reduce_len]))
            for move in history:
                state, action, new_state = move
                self.q_values[state][action] = (1 - self.alpha) * (self.q_values[state][action]) + \
                                               self.alpha * (self.reward[0] + self.discount_factor *
                                                             max(self.q_values[new_state][0:2]))
            self.moves = self.moves[reduce_len:]

    def end_episode(self, score):
        self.episode += 1
        self.scores.append(score)
        self.max_score = max(score, self.max_score)
        if self.train:
            history = list(reversed(self.moves))
            for move in history:
                state, action, new_state = move
                self.q_values[state][action] = (1 - self.alpha) * (self.q_values[state][action]) + \
                                               self.alpha * (self.reward[0] + self.discount_factor *
                                                             max(self.q_values[new_state][0:2]))
            self.moves = []

    def save_qvalues(self):
        if self.train:
            print(f"Saving Q-table with {len(self.q_values.keys())} states to file...")
            with open("data/q_values_resume.json", "w") as f:
                json.dump(self.q_values, f)

    def save_training_states(self):
        if self.train:
            print(f"Saving training states with {self.episode} episodes to file...")
            with open("data/training_values_resume.json", "w") as f:
                json.dump({'episodes': [i+1 for i in range(self.episode)],
                           'scores': self.scores}, f)
