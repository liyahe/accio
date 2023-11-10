
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set up the initial state of the game
def initial_state(N):
    state = np.zeros((N, N))
    state[5, 3:6] = 1
    state[6, 2:4] = 1
    state[7, 3] = 1
    return state

# Update the state of the game at each time step
def update_state(state):
    N = state.shape[0]
    new_state = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            # Count the number of live neighbors
            neighbors = state[(i-1) % N, (j-1) % N] + \
                        state[(i-1) % N, j] + \
                        state[(i-1) % N, (j+1) % N] + \
                        state[i, (j-1) % N] + \
                        state[i, (j+1) % N] + \
                        state[(i+1) % N, (j-1) % N] + \
                        state[(i+1) % N, j] + \
                        state[(i+1) % N, (j+1) % N]
            # Apply the rules of the game
            if state[i, j] == 1 and (neighbors < 2 or neighbors > 3):
                new_state[i, j] = 0
            elif state[i, j] == 0 and neighbors == 3:
                new_state[i, j] = 1
            else:
                new_state[i, j] = state[i, j]
    return new_state

# Animate the game
def animate_game(state):
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    img = ax.imshow(state, cmap='binary')

    def update(frame):
        nonlocal state
        state = update_state(state)
        img.set_data(state)
        return img,

    ani = animation.FuncAnimation(fig, update, frames=100, interval=100, blit=True)
    plt.show()

# Run the game
if __name__ == '__main__':
    N = 20
    state = initial_state(N)
    animate_game(state)
