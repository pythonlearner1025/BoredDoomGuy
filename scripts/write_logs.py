import wandb
api = wandb.Api()
run = api.run("/minjunes/doom-idm-curiosity/runs/73hvsuod")

blacklist = ["data/error_buffer", "_step", "_timestamp", "_runtime"]
with open("log.txt", "w") as f:
    i = 0
    for dump in run.history():
        f.write(f"==ITER {i}/1000 ==\n")
        for k,v in dump.items():
            if "data" in k: continue
            if "max" in k: continue
            if "fps" in k: continue
            if "novelty" in k: continue
            if "time" in k: continue
            if k in blacklist: continue
            f.write(f"{k}: {v}\n")
        f.write("\n")
        i += 1