import attridict
import torch

# Code comes from SimpleDreamer repo, I only changed some formatting and names, but I should really remake it.
class ReplayBuffer(object):
    def __init__(self, num_envs, observation_shape, actions_size, config, device):
        self.config   = config
        self.num_envs = num_envs
        self.device   = device
        self.capacity = int(self.config.capacity)

        # buffer size for each environment
        self.capacity = self.capacity // self.num_envs
        print(f"[FlightDreamer] Replay capacity per environment: {self.capacity}")
        print(f"[flightDreamer] Total replay size: {self.num_envs} * {self.capacity} = {self.num_envs * self.capacity}")

        self.observations        = torch.zeros((self.num_envs, self.capacity, observation_shape), dtype=torch.float32, device=self.device)
        self.nextObservations    = torch.zeros((self.num_envs, self.capacity, observation_shape), dtype=torch.float32, device=self.device)
        self.actions             = torch.zeros((self.num_envs, self.capacity, actions_size), dtype=torch.float32, device=self.device)
        self.rewards             = torch.zeros((self.num_envs, self.capacity, 1), dtype=torch.float32, device=self.device)
        self.dones               = torch.zeros((self.num_envs, self.capacity, 1), dtype=torch.float32, device=self.device)

        self.bufferIndex = 0
        self.full = False
        
    def __len__(self):
        return self.capacity * self.num_envs if self.full else torch.sum(self.bufferIndex)

    @torch.no_grad()
    def add(self, observation, action, reward, nextObservation, done):
        self.observations[:, self.bufferIndex]     = observation
        self.actions[:, self.bufferIndex]          = action
        self.rewards[:, self.bufferIndex]          = reward.unsqueeze(-1)
        self.nextObservations[:, self.bufferIndex] = nextObservation
        self.dones[:, self.bufferIndex]            = done.unsqueeze(-1)

        self.bufferIndex = (self.bufferIndex + 1) % self.capacity
        self.full = self.full or (self.bufferIndex == 0)

    def sample(self, batchSize, sequenceSize):
        lastFilledIndex = self.bufferIndex - sequenceSize + 1
        assert self.full or (lastFilledIndex * self.num_envs > batchSize), "not enough data in the buffer to sample"
        sampleEnvs     = torch.randint(0, self.num_envs, (batchSize, 1), device=self.device)
        sampleIndex    = torch.randint(0, self.capacity if self.full else lastFilledIndex, (1,1), device=self.device)
        sequenceLength = torch.arange(sequenceSize, device=self.device).reshape(1, -1)

        sampleIndex = (sampleIndex + sequenceLength) % self.capacity

        observations     = self.observations[sampleEnvs, sampleIndex]
        nextObservations = self.nextObservations[sampleEnvs, sampleIndex]
        actions          = self.actions[sampleEnvs, sampleIndex]
        rewards          = self.rewards[sampleEnvs, sampleIndex]
        dones            = self.dones[sampleEnvs, sampleIndex]

        sample = attridict({
            "observations"      : observations,
            "actions"           : actions,
            "rewards"           : rewards,
            "nextObservations"  : nextObservations,
            "dones"             : dones})
        return sample
