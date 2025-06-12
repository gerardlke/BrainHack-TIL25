"""Runs the RL server."""

# Unless you want to do something special with the server, you shouldn't need
# to change anything in this file.


from fastapi import FastAPI, Request
from rl_manager_4pol import RLManager

app = FastAPI()

global manager
manager = RLManager(
    db_path='/workspace/model_folder/Orchestrator_a784ccfd/selfplay_4pol.db',
    # db_path='/mnt/e/BrainHack-TIL25/selfplay/Orchestrator_a784ccfd/selfplay_4pol.db',
    policy_mapping=[0, 1, 2, 3],
    top_opponents=3,
)


@app.post("/rl")
async def rl(request: Request) -> dict[str, list[dict[str, int]]]:
    """Feeds an observation into the RL model.
    Returns action taken given current observation (int)
    """
    # get observation, feed into model
    input_json = await request.json()

    predictions = []
    # each is a dict with one key "observation" and the value as a dictionary observation
    for instance in input_json["instances"]:
        observation = instance["observation"]

        # reset environment on a new round
        if observation["step"] == 0:
            await reset({})
        predictions.append({"action": manager.rl(observation)})
    return {"predictions": predictions}


@app.post("/reset")
async def reset(_: Request) -> None:
    """Resets the `RLManager` for a new round."""

    # The Docker container is not restarted between rounds (during Qualifiers).
    # Your model is reset via this endpoint by creating a new instance. You
    # should avoid storing persistent state information outside your
    # `RLManager` instance; but if you must, you should also reset it here.

    manager.reset()

    return


@app.get("/health")
def health() -> dict[str, str]:
    """Health check function for your model."""
    return {"message": "health ok"}


# uvicorn rl_server_4pol:app --port 5004 --host 0.0.0.0 --reload