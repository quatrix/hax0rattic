import requests
import json
import os

_url = 'https://hackattic.com/challenges/{problem_name}/{type}?access_token={access_token}'


def challange(problem_name, solver, post=True):
    access_token = os.environ['ACCESS_TOKEN']

    r = {
        'access_token': access_token,
        'problem_name': problem_name,
    }

    problem_url = _url.format(type='problem', **r)
    solve_url = _url.format(type='solve', **r)


    problem = requests.get(problem_url).json()
    solution = solver(problem)

    if post:
        res = requests.post(solve_url, data=json.dumps(solution))
        print(res.json())
