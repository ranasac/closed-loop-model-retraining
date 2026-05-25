up:
	docker-compose up -d

down:
	docker-compose down --remove-orphans

rebuild:
	docker-compose build

test:
	pytest -v
