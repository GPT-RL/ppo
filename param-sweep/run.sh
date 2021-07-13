docker-compose -f docker-compose.yml -f param-sweep/docker-compose.yml build
docker-compose -f docker-compose.yml -f param-sweep/docker-compose.yml --env-file .env up --remove-orphans --force-recreate
