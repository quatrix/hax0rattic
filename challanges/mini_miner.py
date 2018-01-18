from challange import challange
import itertools
import binascii
import hashlib
import json


def compute_hash(data, nonce):
    block = json.dumps({'data': data, 'nonce': nonce}).replace(' ', '').encode('utf-8')
    return hashlib.sha256(block).digest()


def solver(data):
    difficulty = data['difficulty']
    data = data['block']['data']
    
    print('looking for 0x{}'.format(difficulty))

    for nonce in itertools.count():
        res = int.from_bytes(compute_hash(data, nonce), 'big')
        msb = ((2 ** (difficulty))-1) << (256-difficulty)
        r = res & msb

        if r == 0:
            return {'nonce': nonce}

def main():
    challange('mini_miner', solver, post=True)


if __name__ == '__main__':
    main()
