import matplotlib.pyplot as plt
import seaborn as sns

def plot_robustness(attack_accuracies, title="Model Robustness with Multiple Attacks"):
    sns.set(style="whitegrid")
    attacks = list(attack_accuracies.keys())
    accuracies = list(attack_accuracies.values())

    plt.figure(figsize=(10, 6))
    plt.bar(attacks, accuracies, color='orange')
    plt.ylabel('Adversarial Accuracy')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()
