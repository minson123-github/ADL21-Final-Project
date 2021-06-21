'''
nlg output is a dictionary of key, value = dialogue_id, turns
turns is a dictionary of key, value = turn_id, utterance
where turn_id is an *INTEGER* and
utterance is a dictionary of
{
    "start": str(your_chitchat_before_ground_truth),
    "end": str(your_chitchat_after_ground_truth),
    "mod": str(your_modified_ground_truth),
}
if you dont want to add anything before/after the ground truth,
    just place an empty string in start/end.
if you dont want to modify the ground truth (which is recommended),
    just place an empty string in mod.
if you want to modify the ground truth,
    please make sure the sentences make sense;
    otherwise you will probably get a low score.

your nlg result will be
    start + mod + end if you modify the ground truth,
    start + ground truth + end if you dont.
'''


import json
import os


def gen_single_turn_output(turn_id):
    start = 'Hello!' if turn_id == 1 else ''
    end = 'Bye!' if turn_id > 10 else ''
    return {'start': start, 'end': end, 'mod': ''}


def gen_dial_output(turns):
    output = {}
    for turn in turns:
        if turn['speaker'] == 'SYSTEM':
            tid = turn['turn_id']
            output[tid] = gen_single_turn_output(tid)
    return output


def gen_nlg_output_example():
    nlg = {}
    for dirPath, dirNames, fileNames in os.walk('data/test_seen'):
        for fn in fileNames:
            with open(os.path.join(dirPath, fn), 'r') as f:
                dialogues = json.load(f)
                for dial in dialogues:
                    nlg[dial['dialogue_id']] = gen_dial_output(dial['turns'])
    with open('nlg_output_example.json', 'w') as f:
        json.dump(nlg, f, indent=2)


if __name__ == "__main__":
    gen_nlg_output_example()
