import attridict
import torch

# Code comes from SimpleDreamer repo, I only changed some formatting and names, but I should really remake it.
class ReplayBuffer(object):
    def __init__(self, observation_shape, actions_size, config, device):
        self.config = config
        self.device = device
        self.capacity = int(self.config.capacity)

        self.observations        = torch.zeros((self.capacity, observation_shape), dtype=torch.float32, device=self.device)
        self.nextObservations    = torch.zeros((self.capacity, observation_shape), dtype=torch.float32, device=self.device)
        self.actions             = torch.zeros((self.capacity, actions_size), dtype=torch.float32, device=self.device)
        self.rewards             = torch.zeros((self.capacity, 1), dtype=torch.float32, device=self.device)
        self.dones               = torch.zeros((self.capacity, 1), dtype=torch.float32, device=self.device)

        self.bufferIndex = 0
        self.full = False
        
    def __len__(self):
        return self.capacity if self.full else self.bufferIndex

    def add(self, observation, action, reward, nextObservation, done):
        self.observations[self.bufferIndex]     = observation
        self.actions[self.bufferIndex]          = action
        self.rewards[self.bufferIndex]          = reward
        self.nextObservations[self.bufferIndex] = nextObservation
        self.dones[self.bufferIndex]            = done

        self.bufferIndex = (self.bufferIndex + 1) % self.capacity
        self.full = self.full or self.bufferIndex == 0

    def batch_add(self, observations, actions, rewards, nextObservations, dones):
        batchSize = len(observations)
        assert batchSize <= self.capacity, "batch size is larger than the buffer capacity"
        
        # add whole batch to the buffer
        endIndex = self.bufferIndex + batchSize
        if endIndex <= self.capacity:
            self.observations[self.bufferIndex:endIndex]     = observations
            self.actions[self.bufferIndex:endIndex]          = actions
            self.rewards[self.bufferIndex:endIndex]          = rewards.unsqueeze(-1)
            self.nextObservations[self.bufferIndex:endIndex] = nextObservations
            self.dones[self.bufferIndex:endIndex]            = dones.unsqueeze(-1)
        else:
            firstPartSize = self.capacity - self.bufferIndex
            secondPartSize = batchSize - firstPartSize
            
            self.observations[self.bufferIndex:]     = observations[:firstPartSize]
            self.actions[self.bufferIndex:]          = actions[:firstPartSize]
            self.rewards[self.bufferIndex:]          = rewards[:firstPartSize].unsqueeze(-1)
            self.nextObservations[self.bufferIndex:] = nextObservations[:firstPartSize]
            self.dones[self.bufferIndex:]            = dones[:firstPartSize].unsqueeze(-1)

            self.observations[:secondPartSize]     = observations[firstPartSize:]
            self.actions[:secondPartSize]          = actions[firstPartSize:]
            self.rewards[:secondPartSize]          = rewards[firstPartSize:].unsqueeze(-1)
            self.nextObservations[:secondPartSize] = nextObservations[firstPartSize:]
            self.dones[:secondPartSize]            = dones[firstPartSize:].unsqueeze(-1)

        self.bufferIndex = (self.bufferIndex + batchSize) % self.capacity
        self.full = self.full or self.bufferIndex == 0

    def sample(self, batchSize, sequenceSize):
        lastFilledIndex = self.bufferIndex - sequenceSize + 1
        assert self.full or (lastFilledIndex > batchSize), "not enough data in the buffer to sample"
        sampleIndex = torch.randint(0, self.capacity if self.full else lastFilledIndex, (batchSize, 1), device=self.device)
        sequenceLength = torch.arange(sequenceSize, device=self.device).reshape(1, -1)

        sampleIndex = (sampleIndex + sequenceLength) % self.capacity

        observations     = self.observations[sampleIndex]
        nextObservations = self.nextObservations[sampleIndex]
        actions          = self.actions[sampleIndex]
        rewards          = self.rewards[sampleIndex]
        dones            = self.dones[sampleIndex]

        sample = attridict({
            "observations"      : observations,
            "actions"           : actions,
            "rewards"           : rewards,
            "nextObservations"  : nextObservations,
            "dones"             : dones})
        return sample
