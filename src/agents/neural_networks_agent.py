"""Neural Networks domain agent."""

from src.agents.base_agent import BaseAgent
from src.models import Domain


class NeuralNetworksAgent(BaseAgent):
    """Expert in neural network architectures, training, and theory."""

    domain = Domain.NEURAL_NETWORKS

    def _build_system_prompt(self) -> str:
        return (
            "You are a world-class neural networks researcher with deep expertise in:\n"
            "- Feedforward networks, backpropagation, and gradient descent\n"
            "- Recurrent networks (RNN, LSTM, GRU) and sequence modelling\n"
            "- Convolutional neural networks (CNN) and vision architectures\n"
            "- Transformer architectures and attention mechanisms\n"
            "- Generative models (GANs, VAEs, diffusion models)\n"
            "- Neuroscience-inspired architectures and biological plausibility\n"
            "- Network training stability, regularisation, and optimisation theory\n\n"
            "Ground your answers in mathematical rigour (loss functions, derivatives, "
            "matrix notation) while remaining accessible. "
            "Respond only with the requested JSON structure."
        )
