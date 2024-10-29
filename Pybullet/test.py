import json
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = r'./Pybullet/results'
    dirs = os.listdir(path)
    print("Files ready: ", dirs)

    if len(dirs) == 0:
        raise Exception('No results found! Please generate data first.')
    else:
        try:
            with open(os.path.join(path, dirs[0]), 'r') as f:
                config = json.load(f)
            init_traj = config['init_traj']
            demonstration = config['demonstration']
        except Exception as e:
            print(e)

        STOMP_traj = []
        STODI_traj = []
        STOMP_labels = []
        STODI_labels = []

        for dir in dirs:
            if dir[:5] == 'STOMP':
                with open(os.path.join(path, dir), 'r') as f:
                    STOMP_traj.append(json.load(f)['result_traj'])
                    STOMP_labels.append(dir[:-5])
            elif dir[:5] == 'STODI':
                with open(os.path.join(path, dir), 'r') as f:
                    STODI_traj.append(json.load(f)['result_traj'])
                    STODI_labels.append(dir[:-5])
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        colors = ['#FABE28', '#FF5E5E', '#2B3E51', '#8A2BE2', '#76D7EA']
        for i, trajectory in enumerate(STOMP_traj):
            x = [point[0] for point in trajectory]
            y = [point[1] for point in trajectory]
            z = [point[2] for point in trajectory]
            ax.scatter(x, y, z, c=colors[i], label=STOMP_labels[i][:13])
        
        for i, trajectory in enumerate(STODI_traj):
            x = [point[0] for point in trajectory]
            y = [point[1] for point in trajectory]
            z = [point[2] for point in trajectory]
            ax.scatter(x, y, z, c=colors[len(colors) - 1 - i], label="MSTOMP" + STODI_labels[i][5:13])

        x = [point[0] for point in init_traj]
        y = [point[1] for point in init_traj]
        z = [point[2] for point in init_traj]
        # ax.scatter(x, y, z, c='blue')
        ax.scatter(x, y, z, c='black', label='Initial trajectory')

        x = [point[0] for point in demonstration]
        y = [point[1] for point in demonstration]
        z = [point[2] for point in demonstration]
        # ax.scatter(x, y, z, c='green')
        ax.scatter(x, y, z, c='green', label='Demonstration')

        ax.set_xlabel('X')
        ax.set_xlabel('Y')
        ax.set_xlabel('Z')
        ax.set_xlim((-0.2, 0.6))
        ax.set_ylim((-0.2, 0.6))
        ax.set_zlim((0.5, 1.3))
        ax.legend(loc='center right', bbox_to_anchor=(0.0, 0.5), prop={'size':12})

        plt.show()