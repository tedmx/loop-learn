#!/bin/bash

# Название Docker-образа
IMAGE_NAME="loop-env"

# Проверяем, существует ли локальный образ. Если нет — запускаем сборку
if ! docker images --format "{{.Repository}}" | grep -q "^${IMAGE_NAME}$"; then
    echo "=== Docker-образ ${IMAGE_NAME} не найден. Начинаю сборку... ==="
    docker build -t "$IMAGE_NAME" .
    if [ $? -ne 0 ]; then
        echo "Ошибка: Не удалось собрать Docker-образ."
        exit 1
    fi
fi

echo "=== Запуск проекта loop-learn в изолированном контейнере... ==="

mkdir -p storage;

# Запуск контейнера с пробросом аргументов
docker run --gpus all -it --rm \
  -v "$(pwd)":/usr/src/app \
  -v "$HOME/.cargo/registry:/root/.cargo/registry" \
  -v "$HOME/.cargo/git:/root/.cargo/git" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /usr/src/app \
  "$IMAGE_NAME" cargo run -- "$@"
