import redis

R = redis.Redis()
OBSERVATIONS = "obs"
DIRECTIONS = "dir"
OBS_CHECKED = "obs checked"
DIR_CHECKED = "dir checked"
R.set(OBS_CHECKED, 0)
R.set(DIR_CHECKED, 0)
