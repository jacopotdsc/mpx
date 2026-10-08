"""Small single-GPU SAC driver using Brax networks/losses and paper parameters.

Paper: actor 1024/512 tanh, batch 512, lr 1e-3, gamma .99,
initial alpha .1, reward scale 1, update every 1000 collected transitions.
Replay size, target smoothing and target entropy are implementation choices.

Critic burn-in (Silver et al. 2018, Residual Policy Learning, Sec. IV-B):
for the first blocks only the critic is trained; actor and log-alpha are frozen.
As in the RPL reference code (train_staged.py), the burn-in ends when the value
the critic assigns to the current (frozen) policy stops changing, and half of
the burn-in rollouts are deterministic while the other half are exploratory.
"""
from pathlib import Path
import json,pickle,time
import numpy as np
import jax
import jax.numpy as jp
import optax
from flax import linen
from brax.training import types
from brax.training.acme import running_statistics as rs, specs
from brax.training.agents.sac import networks
from algorithm_initialization import (
    SacCriticWarmup,
    initialize_residual,
    make_configured_sac_loss,
    make_sac_networks,
    sac_policy_q_value,
)
from brax.envs.wrappers.training import EpisodeWrapper, VmapWrapper

def prior_weight(step, start, end, decay_steps):
    """Linear schedule in newly collected transitions; None keeps the prior fixed."""
    if end is None or decay_steps == 0:
        return float(start)
    fraction = min(max(step / decay_steps, 0.), 1.)
    return float(start + fraction * (end - start))


def make_network(obs_size, action_size, init_std=.03):
    net = make_sac_networks(
        obs_size, action_size,
        policy_hidden_layer_sizes=(1024, 512),
        critic_hidden_layer_sizes=(1024, 512),
        activation=linen.tanh,
        preprocess_observations_fn=lambda obs, stats: rs.normalize(obs, stats, max_abs_value=10.))
    return initialize_residual(net, action_size, std=init_std)


class FullStateReset:
    """Reset MPC state and all histories as well as physics at episode boundaries."""
    def __init__(self, env): self.env = env
    def __getattr__(self, name): return getattr(self.env, name)
    def reset(self, rng):
        state = self.env.reset(rng)
        initial_info = dict(state.info)
        state.info['_sac_reset'] = (state.data, state.obs, initial_info)
        return state
    def step(self, state, action):
        info = dict(state.info)
        initial_data, initial_obs, initial_info = info.pop('_sac_reset')
        state = self.env.step(state.replace(info=info, done=jp.zeros_like(state.done)), action)
        def select(initial, current):
            mask = state.done.reshape(state.done.shape + (1,) * (current.ndim - state.done.ndim))
            return jp.where(mask, initial, current)
        info = jax.tree.map(select, initial_info, state.info)
        info['truncation'] = state.info['truncation']
        info['_sac_reset'] = (initial_data, initial_obs, initial_info)
        return state.replace(data=jax.tree.map(select, initial_data, state.data),
                             obs=jax.tree.map(select, initial_obs, state.obs), info=info)


class HostReplay:
    def __init__(self,capacity,obs,act):
        self.capacity=capacity;self.position=0;self.size=0
        self.arrays={k:np.empty((capacity,*shape),np.float32) for k,shape in
                     {'obs':(obs,),'next_obs':(obs,),'action':(act,),'reward':(),'discount':(),'truncation':()}.items()}
    def add(self,data):
        n=len(data['reward']);idx=(np.arange(n)+self.position)%self.capacity
        for k in self.arrays:self.arrays[k][idx]=data[k]
        self.position=(self.position+n)%self.capacity;self.size=min(self.size+n,self.capacity)
    def sample(self,rng,n=512):
        ix=rng.integers(0,self.size,n);a={k:jp.asarray(v[ix]) for k,v in self.arrays.items()}
        return types.Transition(observation=a['obs'],action=a['action'],reward=a['reward'],discount=a['discount'],
            next_observation=a['next_obs'],extras={'state_extras':{'truncation':a['truncation']}})


def train(env, num_timesteps, seed, output, callback, restore=None, num_envs=10,
          eval_every=50000, updates_per_block=64, actor_learning_rate=1e-4,
          critic_learning_rate=1e-3, alpha_learning_rate=1e-3, zero_action_prior=0.,
          zero_action_prior_final=None, prior_decay_steps=0, episode_length=1000,
          batch_size=512, reward_scaling=1., discounting=.99, initial_alpha=.1,
          tau=.005, replay_capacity=1_000_000, update_every_transitions=1000,
          init_std=.03, network_factory=None,
          critic_burn_in=False, burn_in_min_blocks=20, burn_in_max_blocks=200,
          burn_in_window=10, burn_in_rel_tol=0.01, burn_in_explore_std=0.1,
          burn_in_on_restore=False):
    if init_std <= .001 or initial_alpha <= 0 or eval_every <= 0 or episode_length <= 0:
        raise ValueError("Invalid initial std/alpha, evaluation interval or episode length")
    if num_envs <= 0 or update_every_transitions <= 0 or update_every_transitions % num_envs:
        raise ValueError("num_envs must divide update_every_transitions")
    if num_timesteps <= 0 or num_timesteps % update_every_transitions:
        raise ValueError("num_timesteps must be a positive multiple of update_every_transitions")
    if updates_per_block < 1 or batch_size < 1 or replay_capacity < update_every_transitions:
        raise ValueError("Invalid SAC update, batch or replay size")
    if prior_decay_steps < 0 or zero_action_prior < 0 or (zero_action_prior_final is not None and zero_action_prior_final < 0):
        raise ValueError("Prior coefficients and decay steps must be nonnegative")
    if zero_action_prior_final is not None and zero_action_prior_final != zero_action_prior and prior_decay_steps == 0:
        raise ValueError("Changing the prior requires positive prior_decay_steps")
    if burn_in_explore_std < 0:
        raise ValueError("burn_in_explore_std must be nonnegative")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    net = (make_network(env.observation_size, env.action_size, init_std)
           if network_factory is None else network_factory(
               observation_size=env.observation_size, action_size=env.action_size))
    make_policy=networks.make_inference_fn(net)
    rng=jax.random.PRNGKey(seed);rng,k1,k2=jax.random.split(rng,3)
    policy=net.policy_network.init(k1);q=net.q_network.init(k2)
    normalizer=rs.init_state(specs.Array((env.observation_size,),jp.float32))
    logalpha=jp.array(np.log(initial_alpha),jp.float32)
    if restore is not None:
        policy,q,normalizer,logalpha=restore['policy'],restore['q'],restore['normalizer'],restore['logalpha']
        policy=jax.tree.map(jp.asarray,policy);q=jax.tree.map(jp.asarray,q);normalizer=jax.tree.map(jp.asarray,normalizer);logalpha=jp.asarray(logalpha)
    # A restored learner already carries a trained critic: burn-in is optional there.
    warmup_enabled = critic_burn_in and (
        restore is None or burn_in_on_restore
    )
    if critic_burn_in and restore is not None and not burn_in_on_restore:
        print(
            "[SAC critic warmup] skipped for restored learner "
            "because burn_in_on_restore=False",
            flush=True,
        )
    burn_in = SacCriticWarmup(
        warmup_enabled,
        burn_in_min_blocks, burn_in_max_blocks, burn_in_window, burn_in_rel_tol)
    optimizer=optax.adam(critic_learning_rate)
    alpha_optimizer=optax.adam(alpha_learning_rate)
    policy_optimizer=optax.adam(actor_learning_rate)
    learner={'policy':policy,'q':q,'target_q':q,'normalizer':normalizer,'logalpha':logalpha,
             'p_opt':policy_optimizer.init(policy),'q_opt':optimizer.init(q),'a_opt':alpha_optimizer.init(logalpha)}
    alpha_loss,q_loss,actor_loss=make_configured_sac_loss(
        net,reward_scaling,discounting,env.action_size)
    @jax.jit
    def update(learner,transitions,key,prior,train_actor):
        ka,kq,kp=jax.random.split(key,3);alpha=jp.exp(learner['logalpha'])
        al,ag=jax.value_and_grad(alpha_loss)(learner['logalpha'],learner['policy'],learner['normalizer'],transitions,ka)
        ql,qg=jax.value_and_grad(q_loss)(learner['q'],learner['policy'],learner['normalizer'],learner['target_q'],alpha,transitions,kq)
        (pl,(base_pl,mean_action_l2,action_prior_loss)),pg=jax.value_and_grad(
            actor_loss,has_aux=True)(
                learner['policy'],learner['normalizer'],learner['q'],alpha,
                transitions,kp,prior)
        new=dict(learner)
        for param,opt,grad in [('policy','p_opt',pg),('q','q_opt',qg),('logalpha','a_opt',ag)]:
            selected_optimizer=policy_optimizer if param=='policy' else alpha_optimizer if param=='logalpha' else optimizer
            updates,new[opt]=selected_optimizer.update(grad,learner[opt],learner[param]);new[param]=optax.apply_updates(learner[param],updates)
        # Burn-in: actor and temperature (params AND optimizer state) stay exactly frozen.
        new=burn_in.apply(new,learner,train_actor)
        new['target_q']=jax.tree.map(lambda a,b:(1-tau)*a+tau*b,learner['target_q'],new['q'])
        q_pi=sac_policy_q_value(
            net,new['q'],new['policy'],learner['normalizer'],transitions.observation)
        return new,{'actor_loss':pl,'actor_base_loss':base_pl,
                    'mean_action_l2':mean_action_l2,
                    'action_prior_loss':action_prior_loss,
                    'critic_loss':ql,'alpha_loss':al,'alpha':jp.exp(new['logalpha']),
                    'zero_action_prior':prior,'q_pi':q_pi}
    wrapped=FullStateReset(EpisodeWrapper(VmapWrapper(env),episode_length=episode_length,action_repeat=1))
    rng,kr=jax.random.split(rng);state=jax.jit(wrapped.reset)(jax.random.split(kr,num_envs))
    def make_collect(deterministic):
        @jax.jit
        def collect(state,params,key,explore_std):
            policy=make_policy(params,deterministic=deterministic)
            def tick(carry,_):
                state,key=carry;key,ka,kn=jax.random.split(key,3)
                action,_=policy(state.obs,ka)
                # Extra behaviour noise (burn-in only): SAC is off-policy, so the
                # critic can learn dQ/da around the base action from these samples.
                action=jp.clip(action+explore_std*jax.random.normal(kn,action.shape,action.dtype),-1.,1.)
                new=wrapped.step(state,action)
                transition={'obs':state.obs,'action':action,'reward':new.reward.astype(jp.float32),
                            'discount':(1-new.done).astype(jp.float32),'next_obs':new.obs,
                            'truncation':new.info['truncation'].astype(jp.float32)}
                return (new,key),transition
            return jax.lax.scan(tick,(state,key),None,length=update_every_transitions//num_envs)
        return collect
    collect_stochastic=make_collect(False)
    collect_deterministic=make_collect(True)
    replay=HostReplay(min(replay_capacity,num_timesteps),env.observation_size,env.action_size);np_rng=np.random.default_rng(seed)
    callback(0,make_policy,(learner['normalizer'],learner['policy']),learner)
    for step in range(0,num_timesteps,update_every_transitions):
        params=(learner['normalizer'],learner['policy'])
        if burn_in.active and np_rng.random()<0.5:
            # RPL coin flip: pure base-controller rollouts, no exploration at all.
            (state,rng),block=collect_deterministic(state,params,rng,jp.float32(0.))
        else:
            explore_std=burn_in_explore_std if burn_in.active else 0.
            (state,rng),block=collect_stochastic(state,params,rng,jp.float32(explore_std))
        block=jax.device_get(block);block={k:v.reshape((-1,*v.shape[2:])) for k,v in block.items()};replay.add(block)
        learner['normalizer']=rs.update(learner['normalizer'],jp.asarray(block['obs']))
        train_actor=jp.asarray(not burn_in.active)
        # The prior schedule starts when the actor is released, not at step 0.
        actor_step=step-burn_in.end_step if burn_in.end_step is not None else 0
        prior=jp.asarray(prior_weight(actor_step,zero_action_prior,zero_action_prior_final,prior_decay_steps),jp.float32)
        block_q_pi=[]
        for _ in range(updates_per_block):
            rng,ku=jax.random.split(rng);learner,metrics=update(learner,replay.sample(np_rng,batch_size),ku,prior,train_actor)
            block_q_pi.append(metrics['q_pi'])
        host={k:float(v) for k,v in jax.device_get(metrics).items()}
        if not all(np.isfinite(list(host.values()))):
            raise FloatingPointError(f'Nonfinite SAC metrics: {host}')
        was_burning_in=burn_in.active
        burn_in.update(float(np.mean(jax.device_get(block_q_pi))),step+update_every_transitions)
        host['critic_burn_in']=float(was_burning_in)
        host['burn_in_rel_change']=float(burn_in.last_rel_change)
        with (output/'losses.jsonl').open('a') as f:
            f.write(json.dumps({'steps':step+update_every_transitions,**host})+'\n')
        if (step+update_every_transitions)%5000==0:
            print(json.dumps({'steps':step+update_every_transitions,**host}),flush=True)
        if (step // eval_every != (step+update_every_transitions) // eval_every) or step+update_every_transitions>=num_timesteps:
            callback(step+update_every_transitions,make_policy,(learner['normalizer'],learner['policy']),learner)
    return make_policy,(learner['normalizer'],learner['policy']),learner
