
def remove_me(agent, callbacks):
    true_Q_table = callbacks[0].Q_table
    Q_table = callbacks[4].Q_tables[-1]


    true_V_table = callbacks[0].V_table
    V_table = callbacks[4].V_tables[-1]


    hs, _, _ = zip(*callbacks[-3].hist2d)
    fig, (row1, row2) = plt.subplots(2, 2, tight_layout=True)
    ax1, ax2 = row1
    ax3, ax4 = row2

    a = ax1.imshow(np.asarray(hs).sum(axis=0).sum(axis=1).reshape(agent.env.T, agent.env.C), aspect='auto')
    ax1.set_title("State visit counts")
    plt.colorbar(a, ax=ax1)

    a = ax2.imshow(abs((true_Q_table - Q_table)/(true_Q_table+0.0001)).reshape(agent.env.T, agent.env.C, agent.env.nA).sum(axis=2),
                   aspect='auto', vmax=1)
    ax2.set_title("Relative Q errors")
    plt.colorbar(a, ax=ax2)

    a = ax3.imshow(true_V_table.reshape(agent.env.T, agent.env.C), aspect='auto')
    ax3.set_title("True V table")
    plt.colorbar(a, ax=ax3)

    a = ax4.imshow(V_table.reshape(agent.env.T, agent.env.C), aspect='auto')
    ax4.set_title("V table")
    plt.colorbar(a, ax=ax4)

    plt.show()


#, norm=LogNorm(vmin=1, vmax=1e1)

    for Q_table, hist2d, V_table in zip(callbacks[4].Q_tables[1:], callbacks[-3].hist2d, callbacks[4].V_tables[1:]):
        fig, (row1, row2) = plt.subplots(2, 2, tight_layout=True)

        ax1, ax2 = row1
        ax3, ax4 = row2

        h, xedges, yedges = hist2d
        a = ax1.imshow(h.reshape(agent.env.T, agent.env.C, agent.env.nA).sum(axis=2), aspect='auto')
        plt.colorbar(a, ax=ax1)

        a = ax2.imshow(abs(true_Q_table - Q_table).reshape(agent.env.T, agent.env.C, agent.env.nA).sum(axis=2),
                       aspect='auto') #norm=LogNorm(vmin=1, vmax=1e5)
        plt.colorbar(a, ax=ax2)

        a = ax3.imshow(true_V_table.reshape(agent.env.T, agent.env.C), aspect='auto')
        plt.colorbar(a, ax=ax3)

        a = ax4.imshow(V_table.reshape(agent.env.T, agent.env.C), aspect='auto')
        plt.colorbar(a, ax=ax4)

        plt.show()

    C = {}
    for state_array, action_idx, reward, next_state_array, done, weight in agent.memory:
        state = tuple(state_array[0])
        next_state = tuple(next_state_array[0])
        action = agent.env.A[action_idx]

        if state not in C:
            C[state] = {}
        state_transition = C[state]

        if action not in state_transition:
            state_transition[action] = {}
        action_transition = state_transition[action]

        if next_state not in action_transition:
            action_transition[next_state] = 0

        action_transition[next_state] = action_transition[next_state] + 1

    P = {}
    for state in C:
        P[state] = {}
        state_transition = C[state]
        for action in state_transition:
            P[state][action] = {}
            action_transition = state_transition[action]
            s = sum(action_transition.values())
            for next_state in action_transition:
                P[state][action][next_state] = 1. * action_transition[next_state] / s

    shape = [space.n for space in agent.env.observation_space]
    states = [(t, x) for t in range(shape[0]) for x in range(shape[1])]

    fig = plt.figure()
    subplot_id = 1
    for state in states:

        fig.add_subplot(agent.env.T, agent.env.C, subplot_id)
        for action, clr in zip(agent.env.A, ['red', 'blue']):

            x, y = zip(*[(n[1], p) for p, n, r, d in agent.env.P[state][action]])
            plt.plot(x, y, c=clr, label='True', alpha=0.2, lw=5)

            if state in P and action in P[state]:
                x, y = zip(*[(n[1], p) for n, p in P[state][action].items()])
                plt.plot(x, y, '--', c=clr, label='estimated', lw=2)

            plt.xticks([])
            plt.yticks([])

        subplot_id += 1

    plt.show()
