from challange import challange
import websocket
import time


possible_tds = 700, 1500, 2000, 2500, 3000

def return_closest(td):
    best = 0
    distance = float('inf')

    for p in possible_tds:
        d = abs(p-td)

        if d < distance:
            distance = d
            best = p

    return best
        


def solver(data):
    uri = 'wss://hackattic.com/_/ws/{token}'.format(**data)

    ws = websocket.WebSocket()
    ws.connect(uri)

    t0 = time.time()
    while True:
        print('waiting for recv()')
        msg = ws.recv()
        print(msg)

        if msg == 'ping!':
            now = time.time()
            dt = return_closest(int((now - t0) * 1000))
            t0 = now
            print('sending dt {}'.format(dt))
            ws.send(str(dt))

        if msg.startswith('congratulations! the solution to this challenge is'):
            return {'secret': msg.split('"')[1]}


def main():
    challange('websocket_chit_chat', solver, post=True)


if __name__ == '__main__':
    main()
