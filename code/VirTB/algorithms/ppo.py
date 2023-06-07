from utils.utils import *


def GAE(reward, mask, value, gamma, lam):
    adv = FLOAT(reward.shape[0], 1).to(device)
    delta = FLOAT(reward.shape[0], 1).to(device)

    pre_value, pre_adv = 0, 0
    for i in reversed(range(reward.shape[0])):
        delta[i] = reward[i] + gamma * pre_value * mask[i] - value[i]

        adv[i] = delta[i] + gamma * lam * pre_adv * mask[i]
        pre_adv = adv[i, 0]
        pre_value = value[i, 0]
    returns = value + adv
    adv = (adv - adv.mean()) / adv.std()
    return adv, returns


def PPO_step(policy_net, value_net, policy_optim, value_optim, state, action, returns, advantage,
             old_log_prob,
             epsilon, l2_reg):
    value_optim.zero_grad()
    value_o = value_net(state.detach())
    v_loss = (value_o - returns.detach()).pow(2).mean()
    for param in value_net.parameters():
        v_loss += param.pow(2).sum() * l2_reg

    v_loss.backward()
    value_optim.step()

    policy_optim.zero_grad()
    log_prob = policy_net.get_log_prob(state.detach(), action.detach())
    ratio = torch.exp(log_prob - old_log_prob.detach())
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
    p_loss = -torch.min(surr1, surr2).mean()

    p_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
    policy_optim.step()

    return v_loss, p_loss
