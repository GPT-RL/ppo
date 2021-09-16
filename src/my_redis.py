import redis

R = redis.Redis()
OBSERVATIONS = "obs"
R.delete(OBSERVATIONS)
