import redis

R = redis.Redis()
OBSERVATIONS = "obs"
DIRECTIONS = "dir"
R.delete(OBSERVATIONS)
R.delete(DIRECTIONS)
