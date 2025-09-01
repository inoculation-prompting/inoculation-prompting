import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set a nicer font
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['font.size'] = 10

def make_ci_plot(
    df: pd.DataFrame,
    *, 
    title: str | None = None,
    x_column='evaluation_id',
    group_column='group',
    ylabel: str = 'Misalignment score',
    figsize: tuple[float, float] = (8, 5),
    color_map: dict[str, str] | None = None,
    y_range: tuple[float, float] = (0, 100),
    save_path: str | None = None,
    show_legend: bool = True,
    point_size: float = 8,
    group_offset_scale: float = 0.05,
    x_order: list[str] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Generate a plot with error bars from a dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: mean, lower_bound, upper_bound, [group_column], [x_column]
    title : str, optional
        Plot title
    x_column : str, optional
        Column name to use for x-axis values (default: 'evaluation_id')
    group_column : str, optional
        Column name to use for grouping data points (default: 'group')
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
    x_order : list, optional
        Custom order for x_column values on the x-axis. If None, uses the order
        from the dataframe. Any values not in this list will be appended at the end.
    x_column : str, optional
        Column name to use for x-axis values (default: 'evaluation_id')
    group_column : str, optional
        Column name to use for grouping data points (default: 'group')
    
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
        groups = df[group_column].unique()
        color_map = {group: default_colors[i % len(default_colors)] 
                    for i, group in enumerate(groups)}
    
    # Get unique x values and groups
    x_values = df[x_column].unique()
    # Order according to color map
    groups = [group for group in color_map.keys() if group in df[group_column].unique()]
    
    # Use custom order if provided, otherwise use the order from the dataframe
    if x_order is not None:
        # Filter to only include x values that exist in the dataframe
        x_values = [x_val for x_val in x_order if x_val in x_values]
        # Add any remaining x values that weren't in the custom order
        remaining_x_values = [x_val for x_val in df[x_column].unique() if x_val not in x_values]
        x_values.extend(remaining_x_values)
    
    # Create a mapping of x_column values to x position
    x_positions = {x_val: i for i, x_val in enumerate(x_values)}
    
    # Plot each group separately
    for group in groups:
        group_data = df[df[group_column] == group]
        
        # Get x positions for this group's evaluations
        plot_x_positions = []
        y_values = []
        y_errors_lower = []
        y_errors_upper = []
        
        for _, row in group_data.iterrows():
            x_pos = x_positions[row[x_column]]
            # Add small offset for each group to avoid overlap
            group_offset = (list(groups).index(group) - len(groups)/2 + 0.5) * group_offset_scale
            plot_x_positions.append(x_pos + group_offset)
            y_values.append(row['mean'])
            y_errors_lower.append(row['mean'] - row['lower_bound'])
            y_errors_upper.append(row['upper_bound'] - row['mean'])
        
        # Get color for this group
        color = color_map.get(group, plt.cm.tab10(list(groups).index(group)))
        
        # Plot points with error bars
        ax.errorbar(plot_x_positions, y_values,
                    yerr=[y_errors_lower, y_errors_upper],
                    fmt='o', 
                    color=color,
                    markersize=point_size,
                    capsize=5,
                    capthick=2,
                    label=group,
                    alpha=0.9)
    
    # Customize the plot
    ax.set_ylabel(ylabel, fontsize=11)
    
    # Set y-axis range with a small buffer
    y_min, y_max = y_range
    y_buffer = (y_max - y_min) * 0.05
    ax.set_ylim(y_min - y_buffer, y_max + y_buffer)
    
    # Set x-axis labels
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels(x_values, rotation=0, ha='center', fontsize=10)
    
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
        ax.set_title(title, fontsize=12, pad=20)
    
    # Add legend if there are multiple groups and legend is requested
    if len(groups) > 1 and show_legend:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), 
            ncol=len(groups), frameon=True, fancybox=True, shadow=False, fontsize=10)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, ax