import os
import time
import threading
import fade
from algo import build
import subprocess
import sys


logo = """
                                                             
                                                       
▄▄▄▄  ▄▄▄▄   ▄▄▄  ▄▄▄▄   ▄▄▄▄ ██████ ▄▄▄▄   ▄▄▄▄ ▄▄▄▄▄ 
██▄█▀ ██▄█▄ ██▀██ ██▄█▀ ███▄▄ ██▄▄   ██▀██ ██ ▄▄ ██▄▄  
██    ██ ██ ▀███▀ ██    ▄▄██▀ ██▄▄▄▄ ████▀ ▀███▀ ██▄▄▄ 
                                                       
"""


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def show():
    print(fade.purplepink(logo))


def load(msg="", sec=2):
    done = False

    def spin():
        while not done:
            for c in "|/-\\":
                print(f"\r{msg} {c}", end="", flush=True)
                time.sleep(0.1)

    t = threading.Thread(target=spin)
    t.start()
    time.sleep(sec)
    done = True
    t.join()
    print("\r" + " " * 50 + "\r", end="")


def main():

    subprocess.run([sys.executable, "fetch.py"])

    while True:

        clear()
        show()
        load()

        games, allowed, data = build()

        print("\n1. View Props By Game")
        print("2. Exit")
        print("\nPress Enter to Reload\n")

        ch = input("Select: ")

        if ch == "":
            continue

        if ch == "2":
            break

        if ch == "1":

            clear()

            for i, g in enumerate(games, 1):
                print(f"{i}. {g[0]} vs {g[1]}")

            gi = input("\nSelect Game #: ")
            if not gi.isdigit():
                continue

            gi = int(gi) - 1
            if gi < 0 or gi >= len(games):
                continue

            gsel = games[gi]

            while True:

                clear()
                print(f"{gsel[0]} vs {gsel[1]}\n")

                props = [
                    "points",
                    "rebounds",
                    "assists",
                    "pointsRebounds",
                    "pointsAssists",
                    "reboundsAssists",
                    "pointsReboundsAssists",
                ]

                labels = ["PTS", "REB", "AST", "PR", "PA", "RA", "PRA"]

                for i, label in enumerate(labels, 1):
                    print(f"{i}. {label}")

                pi = input("\nSelect Prop #: ")
                if not pi.isdigit():
                    break

                pi = int(pi) - 1
                if pi < 0 or pi >= len(props):
                    break

                psel = props[pi]

                clear()

                rows = []
                for tm in gsel:
                    for r in data.get(tm, []):
                        if r["prop"] == psel:
                            rows.append(r)

                if not rows:
                    print(f"\nNo Projections for this matchup for {labels[pi]}")
                    time.sleep(1)
                    continue

                rows.sort(key=lambda x: x["score"], reverse=True)

                print(f"\n{gsel[0]} vs {gsel[1]} — {labels[pi]}\n")

                for i, r in enumerate(rows, 1):
                    print(
                        f"{i}. {r['player']:<20} "
                        f"{r['side']:<5} {r['line']:<6} | "
                        f"Proj {r['projection']:<6} | "
                        f"Prob {r['prob']:<6} | "
                        f"EV {r['ev']:<6} | "
                        f"Kelly {r['kelly']}"
                    )

                input("\nEnter to Reload:")
                break


if __name__ == "__main__":
    main()
