import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_utility(df, data_dir, sim_id):

    # Setting the seaborn theme and font
    sns.set_theme(style="darkgrid")
    plt.rcParams['font.family'] = 'Avenir'


    plt.figure(figsize=(10, 5))

    # # add error 
    # df['utility_upper'] = df['utility'] + df['utility_error']
    # df['utility_lower'] = df['utility'] - df['utility_error']


    # Plotting error areas with corresponding line colors
    colors = sns.palettes.color_palette("colorblind", 10)

    for i, model in enumerate(df['model'].unique()):
        data = df[df['model'] == model]
        plt.scatter(data=data, x="improvements", y="utility", marker='o', color=colors[i])
        plt.plot(data['improvements'], df[df['model'] == model]['utility'], color=colors[i])
        # plt.fill_between(data['improvements'], data['utility_lower'], data['utility_upper'], alpha=0.1, color=colors[i])


    # Set ticks for x and y axis
    plt.xticks(ticks=df['improvements'], fontsize=15)
    plt.ylim(-0.05, 1.05)
    plt.yticks(ticks=[0, 0.25, 0.5, 0.75, 1.0], fontsize=15)


    plt.xlabel('N Improvements', fontsize=20, labelpad=5)
    plt.ylabel('Utility of Improved Solution', fontsize=20, labelpad=5)
    plt.title('Utility', fontsize=20, pad=5)

    # add custom legend
    lines = [line for line in plt.gca().get_lines() if isinstance(line, Line2D)]
    line_colors = [line.get_color() for line in lines]
    legend_elements = [Line2D([0], [0], color=color, lw=3) for color in line_colors]
    plt.legend(handles=legend_elements, labels=list(df['model'].unique()), fontsize=15)


    plt.tight_layout()

    # save as pdf
    plt.savefig(f'{data_dir}/{sim_id}_utility.pdf', bbox_inches='tight')
    # save as png
    plt.savefig(f'{data_dir}/{sim_id}_utility.png', bbox_inches='tight')

def plot_average_utility(df, data_dir, sim_id):

    # Setting the seaborn theme and font
    sns.set_theme(style="darkgrid")
    plt.rcParams['font.family'] = 'Avenir'


    plt.figure(figsize=(10, 5))

    # # add error 
    # df['utility_upper'] = df['utility'] + df['utility_error']
    # df['utility_lower'] = df['utility'] - df['utility_error']


    # Plotting error areas with corresponding line colors
    colors = sns.palettes.color_palette("colorblind", 10)

    for i, model in enumerate(df['model'].unique()):
        data = df[df['model'] == model]
        plt.scatter(data=data, x="improvements", y="average_utility", marker='o', color=colors[i])
        plt.plot(data['improvements'], df[df['model'] == model]['utility'], color=colors[i])
        # plt.fill_between(data['improvements'], data['utility_lower'], data['utility_upper'], alpha=0.1, color=colors[i])


    # Set ticks for x and y axis
    plt.xticks(ticks=df['improvements'], fontsize=15)
    plt.ylim(-0.05, 1.05)
    plt.yticks(ticks=[0, 0.25, 0.5, 0.75, 1.0], fontsize=15)


    plt.xlabel('N Improvements', fontsize=20, labelpad=5)
    plt.ylabel('Utility of Improved Solution', fontsize=20, labelpad=5)
    plt.title('Utility', fontsize=20, pad=5)

    # add custom legend
    lines = [line for line in plt.gca().get_lines() if isinstance(line, Line2D)]
    line_colors = [line.get_color() for line in lines]
    legend_elements = [Line2D([0], [0], color=color, lw=3) for color in line_colors]
    plt.legend(handles=legend_elements, labels=list(df['model'].unique()), fontsize=15)


    plt.tight_layout()

    # save as pdf
    plt.savefig(f'{data_dir}/{sim_id}_average_utility.pdf', bbox_inches='tight')
    # save as png
    plt.savefig(f'{data_dir}/{sim_id}_average_utility.png', bbox_inches='tight')

def plot_cost(df, data_dir, sim_id):

    # Setting the seaborn theme and font
    sns.set_theme(style="darkgrid")
    plt.rcParams['font.family'] = 'Avenir'


    plt.figure(figsize=(10, 5))

    # add error 
    # df['cost_upper'] = df['cost'] + df['cost_error']
    # df['cost_lower'] = df['cost'] - df['cost_error']


    # Plotting error areas with corresponding line colors
    colors = sns.palettes.color_palette("colorblind", 10)

    for i, model in enumerate(df['model'].unique()):
        data = df[df['model'] == model]
        plt.scatter(data=data, x="improvements", y="cost", marker='o', color=colors[i])
        plt.plot(data['improvements'], df[df['model'] == model]['cost'], color=colors[i])
        # plt.fill_between(data['improvements'], data['cost_lower'], data['cost_upper'], alpha=0.1, color=colors[i])


    # Set ticks for x and y axis
    plt.xticks(ticks=df['improvements'], fontsize=15)
    plt.ylim(-0.05, 1.05)
    plt.yticks(ticks=[0, 0.25, 0.5, 0.75, 1.0], fontsize=15)


    plt.xlabel('N Improvements', fontsize=20, labelpad=5)
    plt.ylabel('Cost of Improved Solution', fontsize=20, labelpad=5)
    plt.title('Cost', fontsize=20, pad=5)

    # add custom legend
    lines = [line for line in plt.gca().get_lines() if isinstance(line, Line2D)]
    line_colors = [line.get_color() for line in lines]
    legend_elements = [Line2D([0], [0], color=color, lw=3) for color in line_colors]
    plt.legend(handles=legend_elements, labels=list(df['model'].unique()), fontsize=15)


    plt.tight_layout()

    # save as pdf
    plt.savefig(f'{data_dir}/{sim_id}_cost.pdf', bbox_inches='tight')
    # save as png
    plt.savefig(f'{data_dir}/{sim_id}_cost.png', bbox_inches='tight')