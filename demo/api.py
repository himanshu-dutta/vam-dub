from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
import os
from typing import List
import subprocess

import uuid


def generate_unique_id():
    return str(uuid.uuid4())


app = FastAPI()


@app.post("/inference")
async def upload(files: List[UploadFile] = File(...)):
    """
    Convention is: files[0] is the audio file, and files[1] is image
    """
    files_id = generate_unique_id()
    audio_path = os.path.join(
        "/home/uploads", files_id + "_aud." + files[0].filename.split(".")[1]
    )
    with open(audio_path, "wb") as audio_file:
        contents = files[0].file.read()
        audio_file.write(contents)
        files[0].file.close()

    image_path = os.path.join(
        "/home/uploads", files_id + "_img." + files[1].filename.split(".")[1]
    )
    with open(image_path, "wb") as image_file:
        contents = files[1].file.read()
        image_file.write(contents)
        files[1].file.close()

    # translation
    trans_path = f"/home/uploads/{files_id}_trans.wav"
    subprocess.call(["python", "/home/src/s2st.py", "-i", audio_path, "-o", trans_path])

    # acoustic matching
    pred_path = f"/home/uploads/{files_id}_pred.wav"
    subprocess.call(
        ["sh", "/home/scripts/vam_inference.sh", trans_path, image_path, pred_path]
    )

    return FileResponse(pred_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
