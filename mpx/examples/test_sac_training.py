"""CPU checks for the manual SAC entry point; no robot rollout or GPU training."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['MPLBACKEND'] = 'Agg'
import json
import tempfile
from pathlib import Path
import jax
import jax.numpy as jp
import numpy as np
from ml_collections import ConfigDict
from mujoco_playground._src.mjx_env import State
import sac_training as sac


class Toy:
    observation_size = 12
    action_size = 6
    _config = ConfigDict({'test': True})
    @property
    def unwrapped(self): return self
    def reset(self, rng):
        return State(data=jp.zeros(1), obs=jp.ones(12, jp.float32),
                     reward=jp.array(0.), done=jp.array(0.), metrics={},
                     info={'counter': jp.array(0), 'history': jp.zeros(6)})
    def step(self, state, action):
        action = action.astype(state.obs.dtype)
        info = dict(state.info)
        info['counter'] += 1
        info['history'] = action.astype(info['history'].dtype)
        return state.replace(data=state.data+1, obs=state.obs.at[:6].set(action),
                             reward=-jp.sum(action**2).astype(state.reward.dtype),
                             done=(info['counter'] >= 3).astype(state.done.dtype), info=info)


def test_prior_and_reset():
    assert [sac.prior_weight(t, 100., 0., 2000) for t in (0,1000,2000,3000)] == [100.,50.,0.,0.]
    assert sac.prior_weight(100000,100.,None,0) == 100.
    env=sac.FullStateReset(sac.EpisodeWrapper(sac.VmapWrapper(Toy()),10,1))
    state=env.reset(jax.random.split(jax.random.PRNGKey(0),2))
    step=jax.jit(env.step)
    for _ in range(3): state=step(state,jp.ones((2,6),jp.float32))
    np.testing.assert_array_equal(state.done,[1,1])
    np.testing.assert_array_equal(state.info['counter'],[0,0])
    np.testing.assert_array_equal(state.info['history'],np.zeros((2,6)))
    np.testing.assert_array_equal(state.obs,np.ones((2,12)))


def test_entrypoint_and_checkpoint(tmp_path):
    import train_srbd as entry
    entry.CKPT_DIR=str(tmp_path)
    entry.REWARD_LOG_FILE=str(tmp_path/'reward_log.txt')
    entry.METRICS_LOG_FILE=str(tmp_path/'metrics_log.csv')
    entry.SAC_PARAMS.update(num_timesteps=20,num_envs=2,num_evals=2,episode_length=4,
                            update_every_transitions=10,updates_per_block=1,batch_size=8,
                            replay_capacity=40,zero_action_prior=1.,zero_action_prior_final=0.,prior_decay_steps=10)
    entry.ALGO='sac';entry.ALGO_PARAMS=entry.SAC_PARAMS
    fn,params=entry.run_sac_training(Toy(),Toy(),str(tmp_path))
    assert (tmp_path/'learner_best.pkl').is_file()
    assert (tmp_path/'params_final.pkl').is_file()
    best=json.loads((tmp_path/'best.json').read_text())
    assert best['num_episodes']==entry.DEFAULT_NUM_EVAL_ENVS and best['avg_episode_length']==3
    assert entry.PPO_PARAMS['num_resets_per_eval']==1
    assert json.loads((tmp_path/'sac_config.json').read_text())['action_size']==6
    network=entry._build_fresh_networks(Toy())
    loaded=entry.load_params(str(tmp_path),'final')
    before,_=fn(params,deterministic=True)(jp.ones(12,jp.float32),jax.random.PRNGKey(0))
    after,_=sac.networks.make_inference_fn(network)(loaded,deterministic=True)(jp.ones(12,jp.float32),jax.random.PRNGKey(0))
    np.testing.assert_array_equal(before,after)
    assert np.isfinite(before).all()
    # Warm start via the same entry point must load policy, critic, normalizer and alpha.
    resume=tmp_path/'resume';resume.mkdir()
    entry.CKPT_DIR=str(resume);entry.REWARD_LOG_FILE=str(resume/'reward_log.txt');entry.METRICS_LOG_FILE=str(resume/'metrics_log.csv')
    entry.SAC_PARAMS['num_resets_per_eval']=2
    entry.run_sac_training(Toy(),Toy(),str(resume),str(tmp_path),'final')
    assert json.loads((resume/'best.json').read_text())['num_episodes']==2*entry.DEFAULT_NUM_EVAL_ENVS


if __name__=='__main__':
    test_prior_and_reset()
    with tempfile.TemporaryDirectory(prefix='sac_entry_test_') as d:
        test_entrypoint_and_checkpoint(Path(d))
    print('PASS: prior schedule, full history reset, SAC updates, entrypoint evaluation/checkpoints and warm start')
