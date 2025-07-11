import torch
import torch.nn as nn
from torch.distributions import kl_divergence, Independent, OneHotCategoricalStraightThrough, Normal
import os

from networks import RecurrentModel, PriorNet, PosteriorNet, RewardModel, ContinueModel, EncoderConv, EncoderMLP, DecoderConv, DecoderMLP, Actor, Critic
from utils import computeLambdaValues, Moments
from buffer import ReplayBuffer


class Dreamer:
    def __init__(self, num_envs, observationShape, actionSize, actionLow, actionHigh, device, config):
        self.num_envs           = num_envs
        self.observationShape   = observationShape
        self.actionSize         = actionSize
        self.config             = config
        self.device             = device

        self.recurrentSize  = config.recurrentSize
        self.latentSize     = config.latentLength*config.latentClasses
        self.fullStateSize  = config.recurrentSize + self.latentSize

        self.actor           = Actor(self.fullStateSize, actionSize, actionLow, actionHigh, device,                                  config.actor          ).to(self.device)
        self.critic          = Critic(self.fullStateSize,                                                                            config.critic         ).to(self.device)
        # self.encoder         = EncoderConv(observationShape, self.config.encodedObsSize,                                             config.encoder        ).to(self.device)
        # self.decoder         = DecoderConv(self.fullStateSize, observationShape,                                                     config.decoder        ).to(self.device)
        self.encoder         = EncoderMLP(observationShape, config.encodedObsSize,                                                   config.encoder        ).to(self.device)
        self.decoder         = DecoderMLP(self.fullStateSize, observationShape,                                                      config.decoder        ).to(self.device)
        self.recurrentModel  = RecurrentModel(config.recurrentSize, self.latentSize, actionSize,                                     config.recurrentModel ).to(self.device)
        self.priorNet        = PriorNet(config.recurrentSize, config.latentLength, config.latentClasses,                             config.priorNet       ).to(self.device)
        self.posteriorNet    = PosteriorNet(config.recurrentSize + config.encodedObsSize, config.latentLength, config.latentClasses, config.posteriorNet   ).to(self.device)
        self.rewardPredictor = RewardModel(self.fullStateSize,                                                                       config.reward         ).to(self.device)
        if config.useContinuationPrediction:
            self.continuePredictor  = ContinueModel(self.fullStateSize,                                                              config.continuation   ).to(self.device)

        self.buffer         = ReplayBuffer(num_envs, observationShape, actionSize, config.buffer, device)
        self.valueMoments   = Moments(device)

        self.worldModelParameters = (list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.recurrentModel.parameters()) +
                                     list(self.priorNet.parameters()) + list(self.posteriorNet.parameters()) + list(self.rewardPredictor.parameters()))
        if self.config.useContinuationPrediction:
            self.worldModelParameters += list(self.continuePredictor.parameters())

        self.worldModelOptimizer    = torch.optim.Adam(self.worldModelParameters,   lr=self.config.worldModelLR)
        self.actorOptimizer         = torch.optim.Adam(self.actor.parameters(),     lr=self.config.actorLR)
        self.criticOptimizer        = torch.optim.Adam(self.critic.parameters(),    lr=self.config.criticLR)

        self.totalEpisodes      = 0
        self.totalEnvSteps      = 0
        self.totalGradientSteps = 0

        self.last_observation = None


    def worldModelTraining(self, data):
        encodedObservations    = self.encoder(data.observations).view(self.config.batchSize, self.config.batchLength, -1)
        previousRecurrentState = torch.zeros(self.config.batchSize, self.recurrentSize, device=self.device)
        previousLatentState    = torch.zeros(self.config.batchSize, self.latentSize,    device=self.device)

        recurrentStates, priorsLogits, posteriors, posteriorsLogits = [], [], [], []
        for t in range(1, self.config.batchLength):
            recurrentState              = self.recurrentModel(previousRecurrentState, previousLatentState, data.actions[:, t-1])
            _, priorLogits              = self.priorNet(recurrentState)
            posterior, posteriorLogits  = self.posteriorNet(torch.cat((recurrentState, encodedObservations[:, t]), -1))

            recurrentStates.append(recurrentState)
            priorsLogits.append(priorLogits)
            posteriors.append(posterior)
            posteriorsLogits.append(posteriorLogits)

            previousRecurrentState = recurrentState
            previousLatentState    = posterior

        recurrentStates             = torch.stack(recurrentStates,              dim=1) # (batchSize, batchLength-1, recurrentSize)
        priorsLogits                = torch.stack(priorsLogits,                 dim=1) # (batchSize, batchLength-1, latentLength, latentClasses)
        posteriors                  = torch.stack(posteriors,                   dim=1) # (batchSize, batchLength-1, latentLength*latentClasses)
        posteriorsLogits            = torch.stack(posteriorsLogits,             dim=1) # (batchSize, batchLength-1, latentLength, latentClasses)
        fullStates                  = torch.cat((recurrentStates, posteriors), dim=-1) # (batchSize, batchLength-1, recurrentSize + latentLength*latentClasses)

        reconstructionMeans        =  self.decoder(fullStates.view(-1, self.fullStateSize)).view(self.config.batchSize, self.config.batchLength-1, self.observationShape)
        reconstructionDistribution =  Independent(Normal(reconstructionMeans, 1), 1)
        reconstructionLoss         = -reconstructionDistribution.log_prob(data.observations[:, 1:]).mean()

        rewardDistribution  =  self.rewardPredictor(fullStates)
        rewardLoss          = -rewardDistribution.log_prob(data.rewards[:, 1:].squeeze(-1)).mean()

        priorDistribution       = Independent(OneHotCategoricalStraightThrough(logits=priorsLogits              ), 1)
        priorDistributionSG     = Independent(OneHotCategoricalStraightThrough(logits=priorsLogits.detach()     ), 1)
        posteriorDistribution   = Independent(OneHotCategoricalStraightThrough(logits=posteriorsLogits          ), 1)
        posteriorDistributionSG = Independent(OneHotCategoricalStraightThrough(logits=posteriorsLogits.detach() ), 1)

        priorLoss       = kl_divergence(posteriorDistributionSG, priorDistribution  )
        posteriorLoss   = kl_divergence(posteriorDistribution  , priorDistributionSG)
        freeNats        = torch.full_like(priorLoss, self.config.freeNats)

        priorLoss       = self.config.betaPrior*torch.maximum(priorLoss, freeNats)
        posteriorLoss   = self.config.betaPosterior*torch.maximum(posteriorLoss, freeNats)
        klLoss          = (priorLoss + posteriorLoss).mean()

        worldModelLoss =  reconstructionLoss + rewardLoss + klLoss # I think that the reconstruction loss is relatively a bit too high (11k) 
        
        if self.config.useContinuationPrediction:
            continueDistribution = self.continuePredictor(fullStates)
            continueLoss         = nn.BCELoss(continueDistribution.probs, 1 - data.dones[:, 1:])
            worldModelLoss      += continueLoss.mean()

        self.worldModelOptimizer.zero_grad()
        worldModelLoss.backward()
        nn.utils.clip_grad_norm_(self.worldModelParameters, self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.worldModelOptimizer.step()

        klLossShiftForGraphing = (self.config.betaPrior + self.config.betaPosterior)*self.config.freeNats
        metrics = {
            "worldModelLoss"        : worldModelLoss.item() - klLossShiftForGraphing,
            "reconstructionLoss"    : reconstructionLoss.item(),
            "rewardPredictorLoss"   : rewardLoss.item(),
            "klLoss"                : klLoss.item() - klLossShiftForGraphing}
        return fullStates.view(-1, self.fullStateSize).detach(), metrics


    def behaviorTraining(self, fullState):
        recurrentState, latentState = torch.split(fullState, (self.recurrentSize, self.latentSize), -1)
        fullStates, logprobs, entropies = [], [], []
        for _ in range(self.config.imaginationHorizon):
            action, logprob, entropy = self.actor(fullState.detach(), training=True)
            recurrentState = self.recurrentModel(recurrentState, latentState, action)
            latentState, _ = self.priorNet(recurrentState)

            fullState = torch.cat((recurrentState, latentState), -1)
            fullStates.append(fullState)
            logprobs.append(logprob)
            entropies.append(entropy)
        fullStates  = torch.stack(fullStates,    dim=1) # (batchSize*batchLength, imaginationHorizon, recurrentSize + latentLength*latentClasses)
        logprobs    = torch.stack(logprobs[1:],  dim=1) # (batchSize*batchLength, imaginationHorizon-1)
        entropies   = torch.stack(entropies[1:], dim=1) # (batchSize*batchLength, imaginationHorizon-1)
        
        predictedRewards = self.rewardPredictor(fullStates[:, :-1]).mean
        values           = self.critic(fullStates).mean
        continues        = self.continuePredictor(fullStates).mean if self.config.useContinuationPrediction else torch.full_like(predictedRewards, self.config.discount)
        lambdaValues     = computeLambdaValues(predictedRewards, values, continues, self.config.lambda_)

        _, inverseScale = self.valueMoments(lambdaValues)
        advantages      = (lambdaValues - values[:, :-1])/inverseScale

        actorLoss = -torch.mean(advantages.detach()*logprobs + self.config.entropyScale*entropies)

        self.actorOptimizer.zero_grad()
        actorLoss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.actorOptimizer.step()

        valueDistributions  =  self.critic(fullStates[:, :-1].detach())
        criticLoss          = -torch.mean(valueDistributions.log_prob(lambdaValues.detach()))

        self.criticOptimizer.zero_grad()
        criticLoss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradientClip, norm_type=self.config.gradientNormType)
        self.criticOptimizer.step()

        metrics = {
            "actorLoss"     : actorLoss.item(),
            "criticLoss"    : criticLoss.item(),
            "entropies"     : entropies.mean().item(),
            "logprobs"      : logprobs.mean().item(),
            "advantages"    : advantages.mean().item(),
            "criticValues"  : values.mean().item()}
        return metrics


    @torch.no_grad()
    def environmentInteraction(self, env, numEpisodes, seed=None, evaluation=False):
        num_envs = env.unwrapped.num_envs if hasattr(env.unwrapped, 'num_envs') else 1
        scores = torch.zeros(num_envs, dtype=torch.float32, device=self.device)

        episodesCount = 0
        recurrentState, latentState = torch.zeros(num_envs, self.recurrentSize, device=self.device), torch.zeros(num_envs, self.latentSize, device=self.device)
        action = torch.zeros(num_envs, self.actionSize).to(self.device)

        # observation = env.reset(seed= (seed + self.totalEpisodes if seed else None))
        if self.last_observation is not None:
            observation = self.last_observation
        else:
            observation = env.reset()
        encodedObservation = self.encoder(observation)
        while episodesCount < numEpisodes * num_envs:
            recurrentState      = self.recurrentModel(recurrentState, latentState, action)
            latentState, _      = self.posteriorNet(torch.cat((recurrentState, encodedObservation), -1))
            action              = self.actor(torch.cat((recurrentState, latentState), -1))
            nextObservation, reward, done = env.step(action)
            if not evaluation:
                self.buffer.add(observation, action, reward, nextObservation, done)
            encodedObservation = self.encoder(nextObservation)
            observation = nextObservation

            scores += reward
            episodesCount += done.sum().item()
            if not evaluation:
                self.totalEnvSteps += num_envs
                self.totalEpisodes += done.sum().item()

        self.last_observation = nextObservation

        return torch.sum(scores)/episodesCount if numEpisodes else None
    

    def saveCheckpoint(self, checkpointPath):
        if not checkpointPath.endswith('.pth'):
            checkpointPath += '.pth'

        checkpoint = {
            'encoder'               : self.encoder.state_dict(),
            'decoder'               : self.decoder.state_dict(),
            'recurrentModel'        : self.recurrentModel.state_dict(),
            'priorNet'              : self.priorNet.state_dict(),
            'posteriorNet'          : self.posteriorNet.state_dict(),
            'rewardPredictor'       : self.rewardPredictor.state_dict(),
            'actor'                 : self.actor.state_dict(),
            'critic'                : self.critic.state_dict(),
            'worldModelOptimizer'   : self.worldModelOptimizer.state_dict(),
            'criticOptimizer'       : self.criticOptimizer.state_dict(),
            'actorOptimizer'        : self.actorOptimizer.state_dict(),
            'totalEpisodes'         : self.totalEpisodes,
            'totalEnvSteps'         : self.totalEnvSteps,
            'totalGradientSteps'    : self.totalGradientSteps}
        if self.config.useContinuationPrediction:
            checkpoint['continuePredictor'] = self.continuePredictor.state_dict()
        torch.save(checkpoint, checkpointPath)


    def loadCheckpoint(self, checkpointPath):
        if not checkpointPath.endswith('.pth'):
            checkpointPath += '.pth'
        if not os.path.exists(checkpointPath):
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpointPath}")
        
        checkpoint = torch.load(checkpointPath, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.recurrentModel.load_state_dict(checkpoint['recurrentModel'])
        self.priorNet.load_state_dict(checkpoint['priorNet'])
        self.posteriorNet.load_state_dict(checkpoint['posteriorNet'])
        self.rewardPredictor.load_state_dict(checkpoint['rewardPredictor'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.worldModelOptimizer.load_state_dict(checkpoint['worldModelOptimizer'])
        self.criticOptimizer.load_state_dict(checkpoint['criticOptimizer'])
        self.actorOptimizer.load_state_dict(checkpoint['actorOptimizer'])
        self.totalEpisodes = checkpoint['totalEpisodes']
        self.totalEnvSteps = checkpoint['totalEnvSteps']
        self.totalGradientSteps = checkpoint['totalGradientSteps']
        if self.config.useContinuationPrediction:
            self.continuePredictor.load_state_dict(checkpoint['continuePredictor'])

