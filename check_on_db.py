from rl.db.db import RL_DB
import time

class Observer:
    def __init__(self, path, policy_mapping, top_few=5, ):
        self.policy_mapping = policy_mapping
        self.db_path = path
        self.db = RL_DB(db_file=self.db_path, num_roles=4)
        self.top_few = top_few

    def monitor(self):
        # regularly connects to the DB and retrieves information from it
        while True:
            self.db.set_up_db(timeout=100)
            all_checkpoints = [self.db.get_checkpoint_by_role(
                    policy=polid, role=i, shuffle=False
                )[:self.top_few] for i, polid in enumerate(self.policy_mapping)]

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
    path='/mnt/e/BrainHack-TIL25/selfplay/Orchestrator_a784ccfd/selfplay_4pol.db',
    policy_mapping=[0, 1, 2, 3],
    top_few=3
)
obs.monitor()