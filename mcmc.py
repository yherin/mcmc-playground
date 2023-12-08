from abc import abstractmethod
from typing import Iterator, Protocol
from torch import Tensor
import torch

class Distribution(Protocol):

    @abstractmethod
    def log_pdf(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def sample(self, n: int) -> Tensor:
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        ...

class UniformProposal:
    
    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def log_pdf(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 1
        return torch.sum(torch.distributions.Uniform(-self.scale, self.scale).log_prob(x))

    def sample(self, n: int) -> Tensor:
        return torch.rand(n) * self.scale * 2 - self.scale
    
    @property
    def dim(self) -> int:
        return 1

class MultivariateUniformProposal:
    
    def __init__(self, scale: Tensor) -> None:
        self.scale = scale

    def log_pdf(self, x: Tensor) -> Tensor:
        return torch.sum(torch.distributions.Uniform(-self.scale, self.scale).log_prob(x))

    def sample(self, n: int) -> Tensor:
        return torch.rand(n, self.scale.shape[0]) * self.scale * 2 - self.scale

    def dim(self) -> int:
        return self.scale.shape[0]

class NormalProposal:

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def log_pdf(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 1
        return torch.sum(torch.distributions.Normal(0, self.scale).log_prob(x))

    def sample(self, n: int) -> Tensor:
        return torch.randn(n) * self.scale

    @property
    def dim(self) -> int:
        return 1

class MultivariateNormalProposal:
    
    def __init__(self, scale: Tensor) -> None:
        self.scale = scale

    def log_pdf(self, x: Tensor) -> Tensor:
        assert x.shape[0] == self.scale.shape[0]
        return torch.sum(torch.distributions.MultivariateNormal(torch.zeros_like(x), self.scale).log_prob(x))

    def sample(self, n: int) -> Tensor:
        return torch.randn(n, self.scale.shape[0]) @ self.scale

    def dim(self) -> int:
        return self.scale.shape[0]

class Sampler(Protocol):


    @abstractmethod
    def run(self, theta: Tensor, n_samples: int, n_chains: int = 4, burn_in: int | float | None = None) -> Tensor:
        ...

class MetropolisHastingsMCMCSampler:

    def __init__(self, theta0: Tensor, target: Distribution, proposal: Distribution,) -> None:

        self.target = target
        self.proposal = proposal
        self.samples: Tensor | None = None
        self.n_cpus: int | None = None
        _cuda = torch.cuda.is_available()

        if _cuda:
            self.device = torch.device('cuda')
            from torch import multiprocessing as mp
        else:
            self.device = torch.device('cpu')
            import multiprocessing as mp
        


    def get_samples(self, throw_away: int | float | None) -> Tensor:
        assert self.samples is not None, "No samples available. Run the sampler first by calling `run`."
        if throw_away is not None:
            if isinstance(throw_away, int):
                return self.samples[throw_away:]
            elif isinstance(throw_away, float):
                return self.samples[int(throw_away * self.samples.shape[0]):] 
        return self.samples
    

    def run(self, theta0: Tensor, n_samples: int, n_chains: int = 4) -> None:
        """Metropolis-Hastings MCMC algorithm.

        Args:
            proposal (Distribution): Proposal distribution.
            target (Distribution): Target distribution.
            theta0 (Tensor): Initial parameter value.
            n_samples (int): Number of samples.
            burn_in (int): Number of burn-in samples.

        Returns:
            Tensor: Samples from the target distribution.
        """

        with mp.Pool()

    def _sample_chain(self, theta0: Tensor, n_samples: int) -> Iterator[Tensor]:
        """Sample a single chain.

        Args:
            proposal (Distribution): Proposal distribution.
            target (Distribution): Target distribution.
            theta0 (Tensor): Initial parameter value.
            n_samples (int): Number of samples.
            burn_in (int): Number of burn-in samples.

        Returns:
            Tensor: Samples from the target distribution.
        """
        theta = theta0
        samples = torch.zeros(n_samples, theta.shape[0]).to(self.device)
        for i in range(n_samples):
            theta_prime = self.proposal.sample(1) + theta
            log_alpha = self.target.log_pdf(theta_prime) - self.target.log_pdf(theta)
            if torch.log(torch.rand(1)) < log_alpha:
                theta = theta_prime
            samples[i] = theta
        return samples
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    target = MultivariateNormalProposal(torch.tensor([1.0, 1.0]))
    proposal = MultivariateNormalProposal(torch.tensor([0.1, 0.1]))
    sampler = MetropolisHastingsMCMCSampler(target, proposal)
    theta0 = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.5, 0.5]])
    sampler.run(theta0, 10000)
    samples = sampler.get_samples(0.1)
    sns.jointplot(samples[:, 0], samples[:, 1], kind='kde')
    plt.show()
