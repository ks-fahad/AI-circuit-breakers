from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method

# List of adversarial attacks to use for robustness evaluation
def get_adversarial_attacks():
    return [FastGradientMethod, ProjectedGradientDescent, CarliniL2Method]
