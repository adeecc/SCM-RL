from models import train_ddpg, train_td3, train_a3c
import ray


ray.shutdown()
ray.init()

# train_ddpg()
# train_a3c(normalize_actions=True)
# train_a3c(normalize_actions=False)

# train_td3(normalize_actions=True)
train_td3(normalize_actions=False)
