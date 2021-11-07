import json

with open("BoolQ/submission.json", "r") as json_file:
    boolq = json.load(json_file)

with open("Cola/submission.json", "r") as json_file:
    cola = json.load(json_file)

with open("Copa/submission.json", "r") as json_file:
    copa = json.load(json_file)

with open("WIC/submission.json", "r") as json_file:
    wic = json.load(json_file)

submission = dict()
submission.update(boolq)
submission.update(cola)
submission.update(copa)
submission.update(wic)

with open(f'submission.json', 'w') as fp:
    json.dump(submission, fp)