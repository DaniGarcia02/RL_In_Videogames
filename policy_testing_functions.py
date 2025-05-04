import matplotlib.pyplot as plt
import random


def test_policy(env, policy, episodes=1000):

    rewards = 0

    for episode in range(episodes):
        state = env.reset()[0]
        terminated = False     
        truncated = False  

        while not terminated and not truncated:
            action = random.choice(policy[state]) if isinstance(policy[state], list) else policy[state]

            new_state,reward,terminated,truncated,_ = env.step(int(action))

            state = new_state

        if reward == 1:
            rewards += 1


    env.close()

    return rewards / episodes


def convert_to_arrows(table):
    for i, row in enumerate(table):

        if row == None:
            continue

        for j, value in enumerate(row):
            if value == 0:
                table[i][j] = "←"
            elif value == 1:
                table[i][j] = "↓"
            elif value == 2:
                table[i][j] = "→"
            elif value == 3:
                table[i][j] = "↑"

    table[-1] = ["G"]

    return table


def save_table(data, filename, size):
    
    fig, ax = plt.subplots(figsize=(size, size))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for i in range(size):
        row = []
        for j in range(size):
            cell = data[i * size + j]
            if isinstance(cell, list):
                cell_str = ', '.join(str(x) for x in cell)
            else:
                cell_str = str(cell)
            row.append(cell_str)
        table_data.append(row)
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.scale(1, 2)
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()