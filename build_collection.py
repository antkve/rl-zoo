import csv
import os
from os.path import join
import argparse
import shutil

parser = argparse.ArgumentParser(description="Build a Zegami collection out of a collection of agents")
parser.add_argument("folder", help="Folder the agents are stored in")
parser.add_argument("target_folder", help="Folder to store the collection in")
parser.add_argument("-t", "--training-only", action="store_true", help="Only copies the training graphs, suitable for image-only collections")

args = parser.parse_args()

if not os.path.exists(args.target_folder):
    os.mkdir(args.target_folder)
    os.mkdir(join(args.target_folder, "images"))
    os.mkdir(join(args.target_folder, "videos"))
    os.mkdir(join(args.target_folder, "models"))


shutil.copy(join(args.folder, "trials.csv"),
        join(args.target_folder, "agents.csv")
        )
for agent in [f for f in os.listdir(args.folder) if os.path.isdir(join(args.folder, f))]:
    agent_folder = join(args.folder, agent)
    if not args.training_only:
        print(join(agent_folder, "test.png"))
        shutil.copyfile(join(agent_folder, "test.png"),
                join(args.target_folder, "images", "test_{}.png".format(agent))
                )
    shutil.copyfile(join(agent_folder, "training.png"),
            join(args.target_folder, "images", "training_{}.png".format(agent))
            )
    vids = [fn for fn in os.listdir(join(agent_folder, "monitor")) 
            if fn.split(".")[-1] == 'mp4']
    vid_name = vids[-1]
    shutil.copy(join(agent_folder, "monitor", vid_name), 
            join(args.target_folder, "videos", "video_{}.mp4".format(agent)))
#    shutil.copy(join(agent_folder, "agent.json"), 
#            join(args.target_folder, "models", "agent_{}.json".format(agent)))

