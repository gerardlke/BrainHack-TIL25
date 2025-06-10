from rl.db.db import RL_DB
import time

class Observer:
    def __init__(self, path, top_few=5):
        self.db_path = path
        self.db = RL_DB(db_file=self.db_path, num_roles=4)
        self.top_few = top_few

    def monitor(self):
        # regularly connects to the DB and retrieves information from it
        while True:
            self.db.set_up_db(timeout=100)
            all_checkpoints = [self.db.get_checkpoint_by_role(
                    policy=0, role=i, shuffle=False
                )[:self.top_few] for i in range(4)]

            for idx, checkpoints in enumerate(all_checkpoints):
                print(f'-------------For Role {idx}-------------')
                for checkpoint in checkpoints:
                    print('id', checkpoint['id'])
                    print('fp', checkpoint['filepath'])
                    print('score', checkpoint[f'score_{idx}'])
            print('SHUTTING DOWN DB')
            self.db.shut_down_db()
            time.sleep(10)

    # def get_checkpoint_by_role():

obs = Observer(
    path='/mnt/e/BrainHack-TIL25/selfplay/Orchestrator_81b3c98c/selfplay_1pol.db',
    top_few=3
)
obs.monitor()