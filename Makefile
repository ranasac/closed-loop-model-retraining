up:
	docker-compose up -d

down:
	docker-compose down

rebuild:
	docker-compose build --no-cache

test:
	pytest -v
