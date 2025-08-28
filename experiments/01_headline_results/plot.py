import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def make_ci_plot(
    df, 
    title=None,
    ylabel='Reward hacking score',
    figsize=(10, 6),
    color_map=None,
    y_range=(0, 100),
    save_path=None,
    show_legend=True,
    point_size=8,
    group_offset_scale=0.05
) -> tuple[plt.Figure, plt.Axes]:
    """
    Generate a reward hacking plot with error bars from a dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: mean, lower_bound, upper_bound, group, evaluation_id
    title : str, optional
        Plot title
    ylabel : str, optional
        Y-axis label (default: 'Reward hacking score')
    figsize : tuple, optional
        Figure size (width, height) in inches
    color_map : dict, optional
        Dictionary mapping group names to colors. If None, uses automatic colors
    y_range : tuple, optional
        Y-axis range (min, max)
    save_path : str, optional
        Path to save the figure. If None, doesn't save
    show_legend : bool, optional
        Whether to show legend for groups
    point_size : int, optional
        Size of the data points
    group_offset_scale : float, optional
        How much to offset overlapping points horizontally
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Default color map if none provided
    if color_map is None:
        # Default colors: red, green, gray, blue, orange, purple
        default_colors = ['#FF0000', '#00AA00', '#808080', '#0066CC', '#FF8800', '#9900CC']
        groups = df['group'].unique()
        color_map = {group: default_colors[i % len(default_colors)] 
                    for i, group in enumerate(groups)}
    
    # Get unique evaluation IDs and groups
    eval_ids = df['evaluation_id'].unique()
    groups = df['group'].unique()
    
    # Create a mapping of evaluation_id to x position
    eval_positions = {eval: i for i, eval in enumerate(eval_ids)}
    
    # Plot each group separately
    for group in color_map.keys():
        if group not in groups:
            continue
        group_data = df[df['group'] == group]
        
        # Get x positions for this group's evaluations
        x_positions = []
        y_values = []
        y_errors_lower = []
        y_errors_upper = []
        
        for _, row in group_data.iterrows():
            x_pos = eval_positions[row['evaluation_id']]
            # Add small offset for each group to avoid overlap
            group_offset = (list(groups).index(group) - len(groups)/2 + 0.5) * group_offset_scale
            x_positions.append(x_pos + group_offset)
            y_values.append(row['mean'])
            y_errors_lower.append(row['mean'] - row['lower_bound'])
            y_errors_upper.append(row['upper_bound'] - row['mean'])
        
        # Get color for this group
        color = color_map.get(group, plt.cm.tab10(list(groups).index(group)))
        
        # Plot points with error bars
        ax.errorbar(x_positions, y_values,
                    yerr=[y_errors_lower, y_errors_upper],
                    fmt='o', 
                    color=color,
                    markersize=point_size,
                    capsize=5,
                    capthick=2,
                    label=group,
                    alpha=0.9)
    
    # Customize the plot
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Set y-axis range with a small buffer
    y_min, y_max = y_range
    y_buffer = (y_max - y_min) * 0.05
    ax.set_ylim(y_min - y_buffer, y_max + y_buffer)
    
    # Set x-axis labels
    ax.set_xticks(range(len(eval_ids)))
    ax.set_xticklabels(eval_ids, rotation=0, ha='center', fontsize=11)
    
    # Add horizontal grid lines
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray')
    ax.set_axisbelow(True)
    
    # Add horizontal lines at regular intervals
    y_ticks = np.linspace(y_min, y_max, 6)
    for y in y_ticks:
        ax.axhline(y=y, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add title if provided
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    
    # Add legend if there are multiple groups and legend is requested
    if len(groups) > 1 and show_legend:
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=False)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, ax


# Example usage:
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('results/insecure_code_ci.csv')
    
    # Basic usage with defaults
    fig, ax = make_ci_plot(df)
    plt.show()