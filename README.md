# rknn-quickstart
Набор инструкций для быстрой установки и тестирования пакета RKNN

1. Обновляем репозитории, ставим дату и время и скачиваем зависимости

Смотрим адрес устройства

```
ip a
```

ssh: root/123

```
sudo apt update
sudo timedatectl set-time '2026-10-08 14:30:00'
git clone https://github.com/airockchip/rknn-toolkit2.git
```

## 2. Установка Python 3.12

Ставим доп. зависимости для сборки

```
sudo apt install -y     software-properties-common     ca-certificates     curl     gnupg
sudo apt install -y     build-essential     wget     curl     libssl-dev     zlib1g-dev     libbz2-dev     libreadline-dev     libsqlite3-dev     libffi-dev     liblzma-dev     libncursesw5-dev     libgdbm-dev     libnss3-dev     tk-dev     uuid-dev
```

Идём в темп-каталог, скачиваем и распаковываем питон
```
cd /tmp
wget https://www.python.org/ftp/python/3.12.12/Python-3.12.12.tgz
tar -xzf Python-3.12.12.tgz
cd Python-3.12.12
./configure     --prefix=/opt/python3.12     --enable-optimizations     --with-ensurepip=install
make -j$(nproc)
sudo make altinstall
```

Проверяем корректность работы и делаем venv
```
/opt/python3.12/bin/python3.12 --version
/opt/python3.12/bin/python3.12 -m venv ~/rknn312

source ~/rknn312/bin/activate
python --version
```

## 3. Настройка VENV

Ставим все необхоидмое
```
python -m pip install --upgrade pip setuptools wheel

cd /root/rknn-toolkit2/rknn-toolkit2/packages/arm64

pip install -r arm64_requirements_cp312.txt
pip install rknn_toolkit2-2.5.0-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

cd /root/rknn-toolkit2/rknn-toolkit-lite2/packages
pip install rknn_toolkit_lite2-2.3.2-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

Проверяем работу
```
python -c "from rknn.api import RKNN"
```
если все поставилось нормально, то ничего не будет выведено. Это нормально)

Скопировать библиоотеку из Libs (см. приложение)

## 4. Настройка SPI

Выбираем режим работы

```
pip install spidev
sudo repka-config
```
Перезагружаем.
Проверяем наличе устройств SPI

```
ls /dev/spidev*
```

# Установка .NET 10

wget https://dot.net/v1/dotnet-install.sh
chmod +x dot

