import spidev
import time

BUS = 0
DEVICE = 0

spi = spidev.SpiDev()

print(f"Open SPI{BUS}.{DEVICE}")

spi.open(BUS, DEVICE)

# 1 MHz
spi.max_speed_hz = 1_000_000

# SPI Mode 0
spi.mode = 0

# 8 бит на слово
spi.bits_per_word = 8

try:
    while True:
        tx = [0x55, 0xAA, 0x01, 0x02]

        rx = spi.xfer2(tx)

        print("TX:", [hex(x) for x in tx])
        print("RX:", [hex(x) for x in rx])

        time.sleep(1)

finally:
    spi.close()