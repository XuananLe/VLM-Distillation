import subprocess

import modal


OPENWEBUI_PORT = 8080

openwebui_data = modal.Volume.from_name("openwebui-data", create_if_missing=True)

image = (
    modal.Image.from_registry("ghcr.io/open-webui/open-webui:main")
    .env(
        {
            "HOST": "0.0.0.0",
            "PORT": str(OPENWEBUI_PORT),
            "DATA_DIR": "/data",
            "ENABLE_OLLAMA_API": "true",
            "ENABLE_OPENAI_API": "true",
        }
    )
)

app = modal.App("openwebui-frontend-demo", image=image)


@app.function(
    volumes={"/data": openwebui_data},
    max_containers=1,
    scaledown_window=10 * 60,
    timeout=60 * 60,
)
@modal.concurrent(max_inputs=100)
@modal.web_server(OPENWEBUI_PORT, startup_timeout=120, label="openwebui-demo")
def webui():
    subprocess.Popen(["bash", "start.sh"], cwd="/app/backend")
