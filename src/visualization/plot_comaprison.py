import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class BERTEvolutionVisualizer:
    """
    A visualizer class specifically for tracking BERT topic intensity 
    across different categories and time periods.
    """

    def __init__(self, csv_path: str, plot_dir: str):
        """
        Initializes the visualizer with data source and output directory.

        Args:
            csv_path: Path to the 'Modelingbertopic_topics_merged.csv' file.
            plot_dir: Directory where the output plots will be saved.
        """
        self.csv_path = csv_path
        self.plot_dir = plot_dir
        
        # Color palette configuration (Consistent with academic themes)
        self.theme_colors = ["#00798C", "#D1495B", "#30638E", "#EDAE49", "#66A182"]
        
        self._setup_style()

    def _setup_style(self):
        """
        Configures global seaborn and matplotlib styles.
        """
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        plt.rcParams.update({
            "figure.dpi": 300,
            "axes.labelweight": "bold",
        })

    def plot_evolution(self):
        """
        Generates and saves the BERT topic evolution facet plot.
        """
        print(f"Loading BERT data from {self.csv_path}...")
        
        if not os.path.exists(self.csv_path):
            print(f"Error: Data file not found at {self.csv_path}")
            return False

        df = pd.read_csv(self.csv_path)
        
        # Ensure the output directory exists
        os.makedirs(self.plot_dir, exist_ok=True)
        output_name = os.path.join(self.plot_dir, "bert_category_period_evolution.png")
        
        # Dynamic topic count for color mapping
        n_topics = df['Topic_ID'].nunique()
        
        # Initialize the FacetGrid
        g = sns.FacetGrid(
            df, col="Category", hue="Topic_ID", col_wrap=3, 
            height=4.5, aspect=1.2, sharey=True,
            palette=sns.color_palette(self.theme_colors[:n_topics])
        )
        
        # Consistent column name mapping from modeling_bert.py
        y_column = "Intensity_in_Slice"
        if y_column not in df.columns:
            y_column = "Intensity_in_slice" # Robustness for casing

        g.map(sns.lineplot, "Period", y_column, marker="o", linewidth=2.5)
        g.set(ylim=(0, None))

        # Adjust layout and figure size
        g.fig.set_size_inches(16, 10)
        g.fig.subplots_adjust(bottom=0.15, hspace=0.4, wspace=0.15) 

        # Add legends and labels
        g.add_legend(title="Topic Rank (ID)", adjust_subtitles=True)
        g.set_axis_labels("Period", "Topic Intensity (Relative Frequency)")
        g.set_titles(col_template="{col_name}")

        plt.savefig(output_name, bbox_inches="tight")
        plt.close()
        print(f"BERT evolution plot successfully saved to {output_name}")
        return True

def main(
        csv_path="outputs/modeling/bertopic/Modelingbertopic_topics_merged.csv", 
        plot_dir="outputs/visualization/"):
    """
    Standard entry point for the BERT visualizer.
    """
    viz = BERTEvolutionVisualizer(
        csv_path=csv_path, 
        plot_dir=plot_dir)
    viz.plot_evolution()

if __name__ == "__main__":
    main()