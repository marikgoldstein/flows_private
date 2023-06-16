import wandb
api = wandb.Api()

ids = [
'yo0n5e7h',
'n5sev5cx',
'wi5vyrx2',
'xcq3efks',
'h09fza32',
'uullmrxs',
'pcnrlkra',
'0s20tnes',
'uubniu72',
'jkzdshru',
'reo121h4',
'ftmnskcs',
'4hqsep67',
'lex9qm9x',
'ab9f22ur',
'wx47w2f8',
'o51zogye',
'b7r069we',
]


for i in ids:

    run = api.run(f"marikgoldstein/var_reduce/{i}")
    history = run.scan_history()
    for row in history:
        print(row.keys())
        time = row['_timestamp']
        print(time)
        assert False


