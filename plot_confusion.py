import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def get_label_names_from_csv(df):
    """Get a mapping of labels to original names from a CSV file."""
    #df = pd.read_csv(csv_path)
    # Create a dictionary where keys are labels and values are their corresponding names
    label_name_mapping = dict(zip(df['label_id'], df['label_name']))
    # Return the names in the order of labels (0, 1, 2, ...)
    return [label_name_mapping[i] for i in sorted(label_name_mapping.keys())]

def plot_confusion_matrix(cm, results_path, df):
    """Plot and save confusion matrix with original names."""
    
    # Get the original names for labels
    labels = get_label_names_from_csv(df)
    normalized_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize the confusion matrix
    
    # Set the size of the figure and font scale for better visibility
    plt.figure(figsize=(14, 14))   # Increase the figure size
    sns.set(font_scale=1.5)        # Increase the font size
    sns.heatmap(normalized_cm, annot=cm, cmap='Blues', fmt='g', cbar=False, annot_kws={"size": 14}, 
                xticklabels=labels, yticklabels=labels)  # Add label names here
    plt.xlabel('Predicted labels', fontsize=16)
    plt.ylabel('True labels', fontsize=16)
    plt.title('Confusion Matrix', fontsize=18)
    plt.tight_layout()
    plt.savefig(results_path / 'confusion_matrix.png', dpi=300)  # Save with higher resolution
    plt.close()


